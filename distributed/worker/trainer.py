# distributed/worker/trainer.py
# ============================================================================
# 로컬 학습 루프 (N 스텝 학습)
# ============================================================================
# 서버에서 받은 글로벌 모델을 기반으로 N 스텝만큼 로컬 학습을 수행합니다.
# scripts/train_gpt.py의 학습 루프를 분산 학습용으로 재구성했습니다.
#
# 【핵심 차이점 — train_gpt.py vs trainer.py】
# - train_gpt.py: 전체 학습 (수천 스텝, 체크포인트 관리 포함)
# - trainer.py: 부분 학습 (N 스텝만, 서버가 체크포인트 관리)

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..common.model import GPT

logger = logging.getLogger(__name__)


# ============================================================================
# 학습 결과 데이터
# ============================================================================
@dataclass
class TrainResult:
    """로컬 학습 결과"""

    steps_trained: int  # 실제 학습한 스텝 수
    train_loss: float  # 평균 학습 loss
    val_loss: float  # 검증 loss
    duration_s: float  # 학습 소요 시간 (초)
    batch_size_used: int  # 사용한 배치 크기
    learning_rate_used: float  # 사용한 학습률


# ============================================================================
# 로컬 트레이너
# ============================================================================
class LocalTrainer:
    """
    로컬 학습 수행자

    【역할】
    서버에서 받은 글로벌 모델을 로컬 데이터로 N 스텝 학습합니다.
    학습 중 주기적으로 heartbeat 콜백을 호출하여 서버에 생존을 알립니다.

    【사용법】
    trainer = LocalTrainer(
        data_dir="data",
        device=torch.device("mps"),
        block_size=256,
    )
    result = await trainer.train(
        model=model,
        local_steps=100,
        batch_size=16,
        learning_rate=3e-4,
        heartbeat_fn=heartbeat_callback,
    )
    """

    def __init__(
        self,
        data_dir: str,
        device: torch.device,
        block_size: int = 256,
        grad_clip: float = 1.0,
    ):
        self._device = device
        self._block_size = block_size
        self._grad_clip = grad_clip

        # 학습/검증 데이터 로드 (numpy memmap — 메모리 효율적)
        train_path = Path(data_dir) / "train.bin"
        val_path = Path(data_dir) / "val.bin"

        if not train_path.exists():
            raise FileNotFoundError(
                f"학습 데이터를 찾을 수 없습니다: {train_path}\n"
                f"먼저 `uv run python scripts/build_bin_dataset.py`를 실행하세요."
            )

        self._train_data = np.memmap(str(train_path), dtype=np.uint16, mode="r")
        self._val_data = np.memmap(str(val_path), dtype=np.uint16, mode="r") if val_path.exists() else self._train_data

        logger.info(
            f"데이터 로드 완료: "
            f"train={len(self._train_data):,} 토큰, "
            f"val={len(self._val_data):,} 토큰"
        )

    # ========================================================================
    # 배치 생성 (train_gpt.py의 get_batch 재사용)
    # ========================================================================
    def _get_batch(
        self,
        split: str,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        학습/검증 데이터에서 랜덤 배치를 생성합니다.

        【동작 원리】
        1. 데이터에서 랜덤한 시작 위치 선택 (batch_size개)
        2. 각 위치에서 block_size 길이의 시퀀스 추출
        3. 입력(x)과 타겟(y) 생성 (y는 x를 한 칸 shift)

        Args:
            split: "train" 또는 "val"
            batch_size: 배치 크기

        Returns:
            x: 입력 토큰 (batch_size, block_size)
            y: 타겟 토큰 (batch_size, block_size)
        """
        data = self._train_data if split == "train" else self._val_data

        # 랜덤 시작 위치 선택
        max_start = len(data) - self._block_size - 1
        ix = torch.randint(max_start, (batch_size,))

        # 입력과 타겟 생성
        x = torch.stack([
            torch.from_numpy(data[i : i + self._block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(data[i + 1 : i + 1 + self._block_size].astype(np.int64))
            for i in ix
        ])

        return x.to(self._device), y.to(self._device)

    # ========================================================================
    # 검증 Loss 측정
    # ========================================================================
    @torch.no_grad()
    def _estimate_val_loss(
        self,
        model: GPT,
        batch_size: int,
        eval_iters: int = 20,
    ) -> float:
        """
        검증 데이터에서 평균 Loss를 측정합니다.

        Args:
            model: GPT 모델
            batch_size: 배치 크기
            eval_iters: 평가 반복 횟수

        Returns:
            평균 검증 loss
        """
        model.eval()
        losses = []

        for _ in range(eval_iters):
            x, y = self._get_batch("val", batch_size)
            _, loss = model(x, y)
            losses.append(loss.item())

        model.train()
        return sum(losses) / len(losses)

    # ========================================================================
    # 메인 학습 루프
    # ========================================================================
    async def train(
        self,
        model: GPT,
        local_steps: int,
        batch_size: int,
        learning_rate: float = 3e-4,
        heartbeat_fn=None,
        heartbeat_interval: int = 10,
    ) -> TrainResult:
        """
        로컬 N 스텝 학습을 수행합니다.

        【학습 흐름】
        1. AdamW 옵티마이저 생성
        2. N 스텝 반복 (배치 로드 → 순전파 → 역전파 → 업데이트)
        3. 주기적으로 heartbeat 전송
        4. 학습 완료 후 검증 loss 측정

        Args:
            model: 학습할 GPT 모델 (글로벌 체크포인트 로드 상태)
            local_steps: 학습할 스텝 수 (서버가 지정)
            batch_size: 배치 크기 (서버가 추천)
            learning_rate: 학습률
            heartbeat_fn: heartbeat 콜백 (async 함수, 선택)
            heartbeat_interval: heartbeat 전송 간격 (스텝 단위)

        Returns:
            TrainResult: 학습 결과 (loss, 소요 시간 등)
        """
        model.train()
        start_time = time.time()

        # AdamW 옵티마이저 (train_gpt.py와 동일 설정)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        # 학습 루프
        total_loss = 0.0
        actual_steps = 0

        logger.info(
            f"로컬 학습 시작: "
            f"steps={local_steps}, batch_size={batch_size}, "
            f"lr={learning_rate}"
        )

        for step in range(local_steps):
            # 1. 배치 로드
            x, y = self._get_batch("train", batch_size)

            # 2. 순전파
            _, loss = model(x, y)

            # 3. 역전파
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 4. Gradient Clipping (학습 안정화)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)

            # 5. 가중치 업데이트
            optimizer.step()

            total_loss += loss.item()
            actual_steps += 1

            # 진행 상황 로깅 (10스텝마다)
            if (step + 1) % 10 == 0:
                avg_loss = total_loss / actual_steps
                logger.info(
                    f"  step {step + 1}/{local_steps} — "
                    f"loss={loss.item():.4f} (avg={avg_loss:.4f})"
                )

            # Heartbeat 전송 (서버에 생존 알림)
            if heartbeat_fn and (step + 1) % heartbeat_interval == 0:
                try:
                    await heartbeat_fn()
                except Exception as e:
                    logger.warning(f"heartbeat 실패 (무시): {e}")

        # 학습 완료
        duration = time.time() - start_time
        avg_train_loss = total_loss / max(actual_steps, 1)

        # 검증 Loss 측정
        val_loss = self._estimate_val_loss(model, batch_size)

        logger.info(
            f"로컬 학습 완료: "
            f"steps={actual_steps}, "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"소요 시간={duration:.1f}초"
        )

        return TrainResult(
            steps_trained=actual_steps,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            duration_s=duration,
            batch_size_used=batch_size,
            learning_rate_used=learning_rate,
        )
