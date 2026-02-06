# distributed/worker/client.py
# ============================================================================
# 서버 통신 클라이언트 (httpx 기반)
# ============================================================================
# Coordinator 서버와 통신하는 HTTP 클라이언트입니다.
# 워커 등록, heartbeat, 작업 요청/완료, 체크포인트 다운로드를 담당합니다.

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import httpx

from ..common.protocol import WORKER_KEY_HEADER, Endpoints

logger = logging.getLogger(__name__)


class CoordinatorClient:
    """
    Coordinator 서버 통신 클라이언트

    【사용법】
    async with CoordinatorClient("http://localhost:8000") as client:
        result = await client.register("철수의 맥북", "mps", ...)
        task = await client.request_task(worker_uid, experiment_id=1)
    """

    def __init__(self, server_url: str, timeout: float = 30.0):
        self._base_url = server_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _headers(self, worker_uid: uuid.UUID | None = None) -> dict:
        """인증 헤더 생성"""
        headers = {}
        if worker_uid:
            headers[WORKER_KEY_HEADER] = str(worker_uid)
        return headers

    # ========================================================================
    # 워커 등록
    # ========================================================================
    async def register(
        self,
        name: str,
        device_type: str,
        device_name: str | None = None,
        gpu_memory_mb: int | None = None,
        ram_mb: int | None = None,
        cpu_cores: int | None = None,
    ) -> dict:
        """
        워커를 서버에 등록합니다.

        Returns:
            {"worker_uid": UUID, "recommended_batch_size": int, ...}
        """
        resp = await self._client.post(
            Endpoints.WORKERS_REGISTER,
            json={
                "name": name,
                "device_type": device_type,
                "device_name": device_name,
                "gpu_memory_mb": gpu_memory_mb,
                "ram_mb": ram_mb,
                "cpu_cores": cpu_cores,
            },
        )
        resp.raise_for_status()
        return resp.json()

    # ========================================================================
    # Heartbeat
    # ========================================================================
    async def heartbeat(
        self,
        worker_uid: uuid.UUID,
        status: str = "online",
    ) -> dict:
        """heartbeat 전송 (30초마다 호출)"""
        resp = await self._client.post(
            Endpoints.WORKERS_HEARTBEAT,
            json={"worker_uid": str(worker_uid), "status": status},
        )
        resp.raise_for_status()
        return resp.json()

    # ========================================================================
    # 작업 요청
    # ========================================================================
    async def request_task(
        self,
        worker_uid: uuid.UUID,
        experiment_id: int = 1,
    ) -> dict:
        """
        서버에 학습 작업을 요청합니다.

        Returns:
            {"task_id": int, "checkpoint_url": str, "local_steps": int, ...}
        """
        resp = await self._client.post(
            Endpoints.TASKS_REQUEST,
            json={
                "worker_uid": str(worker_uid),
                "experiment_id": experiment_id,
            },
        )
        resp.raise_for_status()
        return resp.json()

    # ========================================================================
    # 학습 완료 보고
    # ========================================================================
    async def complete_task(
        self,
        task_id: int,
        worker_uid: uuid.UUID,
        steps_trained: int,
        local_train_loss: float,
        local_val_loss: float,
        training_duration_s: float,
        device_type: str,
        batch_size_used: int,
        learning_rate_used: float,
        weights_path: Path,
    ) -> dict:
        """학습 완료 결과 및 가중치 파일을 업로드합니다."""
        with open(weights_path, "rb") as f:
            resp = await self._client.post(
                Endpoints.tasks_complete(task_id),
                data={
                    "steps_trained": str(steps_trained),
                    "local_train_loss": str(local_train_loss),
                    "local_val_loss": str(local_val_loss),
                    "training_duration_s": str(training_duration_s),
                    "device_type": device_type,
                    "batch_size_used": str(batch_size_used),
                    "learning_rate_used": str(learning_rate_used),
                },
                files={"weights": (weights_path.name, f, "application/octet-stream")},
                headers=self._headers(worker_uid),
                timeout=120.0,  # 파일 업로드는 타임아웃 여유
            )
        resp.raise_for_status()
        return resp.json()

    # ========================================================================
    # 체크포인트 다운로드
    # ========================================================================
    async def download_checkpoint(
        self,
        checkpoint_url: str,
        save_path: Path,
    ) -> Path:
        """
        체크포인트를 다운로드하여 로컬에 저장합니다.

        Args:
            checkpoint_url: 체크포인트 다운로드 URL (상대 경로)
            save_path: 저장 경로

        Returns:
            저장된 파일 경로
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        async with self._client.stream("GET", checkpoint_url) as resp:
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)

        logger.info(f"체크포인트 다운로드 완료: {save_path}")
        return save_path

    # ========================================================================
    # 워커 이탈
    # ========================================================================
    async def leave(self, worker_uid: uuid.UUID) -> None:
        """서버에 이탈을 통보합니다."""
        try:
            await self._client.post(
                Endpoints.WORKERS_LEAVE,
                headers=self._headers(worker_uid),
            )
        except Exception as e:
            logger.warning(f"이탈 통보 실패 (무시): {e}")
