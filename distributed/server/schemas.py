# distributed/server/schemas.py
# ============================================================================
# Pydantic v2 스키마 (API 요청/응답)
# ============================================================================
# FastAPI 엔드포인트의 요청 및 응답 데이터를 정의합니다.

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ============================================================================
# 워커 관련 스키마
# ============================================================================
class WorkerRegisterRequest(BaseModel):
    """워커 등록 요청"""
    name: str = Field(..., description="워커 이름 (예: '철수의 맥북')")
    hostname: str | None = None
    device_type: str = Field(..., description="디바이스 유형: 'cuda', 'mps', 'cpu'")
    device_name: str | None = Field(None, description="GPU 모델명 (예: 'RTX 4090')")
    gpu_memory_mb: int | None = None
    ram_mb: int | None = None
    cpu_cores: int | None = None


class WorkerRegisterResponse(BaseModel):
    """워커 등록 응답"""
    worker_uid: uuid.UUID
    recommended_batch_size: int
    recommended_local_steps: int
    message: str = "워커 등록 완료"


class WorkerHeartbeatRequest(BaseModel):
    """워커 heartbeat 요청"""
    worker_uid: uuid.UUID
    status: str = "online"     # online, training, uploading
    current_step: int | None = None
    current_loss: float | None = None


class WorkerHeartbeatResponse(BaseModel):
    """워커 heartbeat 응답"""
    ok: bool = True
    server_time: datetime


class WorkerBenchmarkRequest(BaseModel):
    """벤치마크 결과 보고"""
    worker_uid: uuid.UUID
    score: float
    steps_per_sec: float


class WorkerBenchmarkResponse(BaseModel):
    """벤치마크 결과 기반 권장 설정"""
    recommended_batch_size: int
    recommended_local_steps: int


class WorkerInfo(BaseModel):
    """워커 정보 조회 응답"""
    id: int
    worker_uid: uuid.UUID
    name: str
    device_type: str
    device_name: str | None
    status: str
    total_contributions: int
    total_steps_trained: int
    trust_score: float
    earned_tokens: int = 0
    used_tokens: int = 0
    remaining_tokens: int = 0
    first_seen: datetime
    last_seen: datetime


# ============================================================================
# 작업(Task) 관련 스키마
# ============================================================================
class TaskRequestBody(BaseModel):
    """작업 요청"""
    worker_uid: uuid.UUID
    experiment_id: int = 1


class TaskRequestResponse(BaseModel):
    """작업 할당 응답"""
    task_id: int                       # contribution ID (임시)
    experiment_id: int
    checkpoint_id: int
    checkpoint_url: str                # 체크포인트 다운로드 URL
    base_global_step: int
    local_steps: int
    batch_size: int
    learning_rate: float
    config: dict                       # GPTConfig 딕셔너리


class TaskCompleteRequest(BaseModel):
    """학습 완료 보고 (파일은 multipart로 별도 전송)"""
    steps_trained: int
    local_train_loss: float
    local_val_loss: float
    training_duration_s: float
    device_type: str
    batch_size_used: int
    learning_rate_used: float


class TaskCompleteResponse(BaseModel):
    """학습 완료 응답"""
    ok: bool = True
    contribution_id: int
    status: str                        # pending, rejected
    stale_gap: int
    message: str


# ============================================================================
# 실험(Experiment) 관련 스키마
# ============================================================================
class ExperimentStatusResponse(BaseModel):
    """실험 상태 조회 응답"""
    id: int
    name: str
    status: str
    current_global_step: int
    current_train_loss: float | None
    current_val_loss: float | None
    best_val_loss: float | None
    active_workers: int
    total_contributions: int
    local_steps_per_round: int


# ============================================================================
# 체크포인트 관련 스키마
# ============================================================================
class CheckpointInfo(BaseModel):
    """체크포인트 정보"""
    id: int
    global_step: int
    round_number: int
    train_loss: float | None
    val_loss: float | None
    num_contributors: int
    is_latest: bool
    is_best: bool
    file_size_bytes: int | None
    created_at: datetime


class CheckpointHistoryResponse(BaseModel):
    """체크포인트 히스토리"""
    checkpoints: list[CheckpointInfo]
    total: int


# ============================================================================
# 메트릭 관련 스키마
# ============================================================================
class MetricsSummaryResponse(BaseModel):
    """전체 메트릭 요약"""
    experiment_name: str
    global_step: int
    current_train_loss: float | None
    current_val_loss: float | None
    best_val_loss: float | None
    active_workers: int
    total_workers: int
    total_contributions: int
    total_steps_trained: int


class LossHistoryEntry(BaseModel):
    """Loss 히스토리 항목"""
    global_step: int
    train_loss: float | None
    val_loss: float | None
    num_contributors: int
    recorded_at: datetime


class LeaderboardEntry(BaseModel):
    """기여도 리더보드 항목"""
    rank: int
    worker_name: str
    total_contributions: int
    total_steps_trained: int
    earned_tokens: int
    trust_score: float


class WorkerStatusEntry(BaseModel):
    """활성 워커 목록 항목"""
    name: str
    device_type: str
    status: str
    last_seen: datetime
    total_steps_trained: int


# ============================================================================
# 공통 응답 스키마
# ============================================================================
class ErrorResponse(BaseModel):
    """에러 응답"""
    detail: str
    error_code: str | None = None
