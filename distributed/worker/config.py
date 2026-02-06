# distributed/worker/config.py
# ============================================================================
# 워커 클라이언트 설정
# ============================================================================

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorkerConfig:
    """
    워커 클라이언트 설정

    【설정 소스】
    - CLI 인자 (최우선)
    - 서버 응답 (batch_size, local_steps)
    """

    # 필수 설정
    name: str = "FAI 워커"
    server_url: str = "http://localhost:8000"
    experiment_id: int = 1

    # 디바이스 설정 (None이면 자동 감지)
    device: str | None = None
    batch_size: int | None = None
    local_steps: int | None = None
    learning_rate: float = 3e-4

    # 워커 ID (등록 후 저장)
    worker_uid: uuid.UUID | None = None

    # 로컬 데이터 경로
    data_dir: str = "data"
    cache_dir: str = ".fai-worker"

    # 동작 설정
    max_rounds: int | None = None    # None이면 무한 반복
    verbose: bool = False

    @property
    def cache_path(self) -> Path:
        """워커 캐시 디렉토리"""
        return Path(self.cache_dir)

    @property
    def state_file(self) -> Path:
        """워커 상태 파일 (worker_uid 저장)"""
        return self.cache_path / "worker_state.json"
