# distributed/worker/checkpoint_io.py
# ============================================================================
# 체크포인트 입출력 관리
# ============================================================================
# 서버에서 체크포인트를 다운로드하고, 학습 완료 후 가중치를 저장합니다.
# 로컬 캐시를 관리하여 불필요한 재다운로드를 방지합니다.

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch

from ..common.model import GPT, GPTConfig
from ..common.serialization import serialize_state_dict
from .client import CoordinatorClient

logger = logging.getLogger(__name__)


class CheckpointIO:
    """
    체크포인트 다운로드/저장/캐시 관리

    【역할】
    - 서버에서 글로벌 체크포인트를 다운로드
    - 학습 완료 후 가중치 파일을 로컬에 저장
    - 캐시 디렉토리를 관리하여 디스크 공간 절약

    【사용법】
    ckpt_io = CheckpointIO(cache_dir=Path(".fai-worker"))
    model = await ckpt_io.download_and_load(client, checkpoint_url, config)
    saved_path = ckpt_io.save_trained_weights(model, task_id)
    """

    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir
        self._checkpoints_dir = cache_dir / "checkpoints"
        self._weights_dir = cache_dir / "weights"

        # 디렉토리 생성
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._weights_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 체크포인트 다운로드 및 모델 로드
    # ========================================================================
    async def download_and_load(
        self,
        client: CoordinatorClient,
        checkpoint_url: str,
        config: GPTConfig,
        device: torch.device,
    ) -> GPT:
        """
        서버에서 체크포인트를 다운로드하고 모델에 로드합니다.

        Args:
            client: Coordinator 서버 클라이언트
            checkpoint_url: 체크포인트 다운로드 URL (상대 경로)
            config: GPT 모델 설정
            device: 학습 디바이스

        Returns:
            체크포인트가 로드된 GPT 모델
        """
        # 다운로드 경로 결정 (URL에서 파일명 추출)
        filename = checkpoint_url.split("/")[-1] if "/" in checkpoint_url else "latest.pt"
        save_path = self._checkpoints_dir / filename

        # 서버에서 다운로드
        logger.info(f"체크포인트 다운로드 중: {checkpoint_url}")
        await client.download_checkpoint(checkpoint_url, save_path)
        logger.info(f"체크포인트 저장 완료: {save_path} ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # 모델 생성 및 가중치 로드
        model = GPT(config)
        checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)

        # 체크포인트 형식 처리 (dict에 'model' 키가 있는 경우)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model = model.to(device)

        logger.info(f"모델 로드 완료 (파라미터: {sum(p.numel() for p in model.parameters()):,})")
        return model

    # ========================================================================
    # 학습 완료 후 가중치 저장
    # ========================================================================
    def save_trained_weights(
        self,
        model: GPT,
        task_id: int,
    ) -> Path:
        """
        학습 완료된 모델 가중치를 로컬 파일로 저장합니다.

        Args:
            model: 학습 완료된 GPT 모델
            task_id: 작업 ID (파일명에 사용)

        Returns:
            저장된 파일 경로 (서버 업로드용)
        """
        save_path = self._weights_dir / f"task_{task_id}_weights.pt"

        # state_dict만 저장 (옵티마이저 제외 — 서버에서 필요 없음)
        state_dict = model.state_dict()
        torch.save(state_dict, save_path)

        size_mb = save_path.stat().st_size / 1024 / 1024
        logger.info(f"학습 가중치 저장: {save_path} ({size_mb:.1f} MB)")

        return save_path

    # ========================================================================
    # 캐시 정리
    # ========================================================================
    def cleanup_old_checkpoints(self, keep_latest: int = 3) -> None:
        """
        오래된 체크포인트를 삭제하여 디스크 공간을 확보합니다.

        Args:
            keep_latest: 유지할 최신 파일 수
        """
        # 수정 시간 기준 정렬
        files = sorted(
            self._checkpoints_dir.glob("*.pt"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        # 오래된 파일 삭제
        for old_file in files[keep_latest:]:
            old_file.unlink()
            logger.debug(f"오래된 체크포인트 삭제: {old_file.name}")

    def cleanup_task_weights(self, task_id: int) -> None:
        """업로드 완료 후 임시 가중치 파일을 삭제합니다."""
        weight_file = self._weights_dir / f"task_{task_id}_weights.pt"
        if weight_file.exists():
            weight_file.unlink()
            logger.debug(f"임시 가중치 삭제: {weight_file.name}")

    def cleanup_all(self) -> None:
        """전체 캐시를 삭제합니다."""
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
            logger.info(f"캐시 전체 삭제: {self._cache_dir}")
