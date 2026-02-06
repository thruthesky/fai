# distributed/server/routes/workers.py
# ============================================================================
# 워커 관리 API (5개 엔드포인트)
# ============================================================================

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import ApiKey, AuditLog, Worker
from ..schemas import (
    WorkerBenchmarkRequest,
    WorkerBenchmarkResponse,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerInfo,
    WorkerRegisterRequest,
    WorkerRegisterResponse,
)
from ...common.constants import AuditEventType, Defaults, WorkerStatus

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# 워커 인증 헬퍼
# ============================================================================
async def get_worker_by_uid(
    worker_uid: uuid.UUID,
    session: AsyncSession,
) -> Worker:
    """worker_uid로 워커를 조회합니다. 없으면 404 에러."""
    result = await session.execute(
        select(Worker).where(Worker.worker_uid == worker_uid)
    )
    worker = result.scalar_one_or_none()
    if worker is None:
        raise HTTPException(status_code=404, detail="등록되지 않은 워커입니다")
    if worker.is_banned:
        raise HTTPException(status_code=403, detail="차단된 워커입니다")
    return worker


# ============================================================================
# POST /register — 워커 등록
# ============================================================================
@router.post("/register", response_model=WorkerRegisterResponse)
async def register_worker(
    req: WorkerRegisterRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    새 워커를 등록합니다.

    【동작】
    1. workers 테이블에 INSERT
    2. worker_uid(UUID) 자동 발급
    3. 하드웨어 기반 권장 batch_size/local_steps 계산
    """
    # 권장 설정 계산 (하드웨어 기반)
    batch_size, local_steps = _recommend_settings(
        device_type=req.device_type,
        gpu_memory_mb=req.gpu_memory_mb,
    )

    worker = Worker(
        name=req.name,
        hostname=req.hostname,
        device_type=req.device_type,
        device_name=req.device_name,
        gpu_memory_mb=req.gpu_memory_mb,
        ram_mb=req.ram_mb,
        cpu_cores=req.cpu_cores,
        recommended_batch_size=batch_size,
        recommended_local_steps=local_steps,
        status=WorkerStatus.ONLINE,
    )
    session.add(worker)
    await session.flush()  # ID 할당

    # 감사 로그
    session.add(AuditLog(
        event_type=AuditEventType.WORKER_JOINED,
        actor_id=worker.id,
        details={"name": req.name, "device_type": req.device_type},
    ))

    logger.info(f"워커 등록: {req.name} ({req.device_type}) → uid={worker.worker_uid}")

    return WorkerRegisterResponse(
        worker_uid=worker.worker_uid,
        recommended_batch_size=batch_size,
        recommended_local_steps=local_steps,
    )


# ============================================================================
# POST /heartbeat — 생존 신호
# ============================================================================
@router.post("/heartbeat", response_model=WorkerHeartbeatResponse)
async def heartbeat(
    req: WorkerHeartbeatRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    워커 heartbeat (30초마다 호출)

    【동작】
    workers.last_seen과 status를 업데이트합니다.
    """
    worker = await get_worker_by_uid(req.worker_uid, session)

    worker.last_seen = datetime.utcnow()
    worker.status = req.status

    return WorkerHeartbeatResponse(
        ok=True,
        server_time=datetime.utcnow(),
    )


# ============================================================================
# POST /benchmark — 벤치마크 결과 보고
# ============================================================================
@router.post("/benchmark", response_model=WorkerBenchmarkResponse)
async def report_benchmark(
    req: WorkerBenchmarkRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    벤치마크 결과를 보고하고 최적 설정을 받습니다.

    【동작】
    1. benchmark_score 저장
    2. steps_per_sec 기반으로 batch_size/local_steps 재계산
    """
    worker = await get_worker_by_uid(req.worker_uid, session)

    worker.benchmark_score = req.score

    # 벤치마크 결과 기반 설정 재계산
    batch_size, local_steps = _recommend_settings(
        device_type=worker.device_type,
        gpu_memory_mb=worker.gpu_memory_mb,
        benchmark_score=req.score,
    )
    worker.recommended_batch_size = batch_size
    worker.recommended_local_steps = local_steps

    return WorkerBenchmarkResponse(
        recommended_batch_size=batch_size,
        recommended_local_steps=local_steps,
    )


# ============================================================================
# GET /me — 내 정보 조회
# ============================================================================
@router.get("/me", response_model=WorkerInfo)
async def get_my_info(
    x_worker_key: str = Header(..., alias="X-Worker-Key"),
    session: AsyncSession = Depends(get_session),
):
    """현재 워커의 정보를 조회합니다."""
    worker = await get_worker_by_uid(uuid.UUID(x_worker_key), session)

    # API 키 잔액 조회
    result = await session.execute(
        select(ApiKey).where(
            ApiKey.worker_id == worker.id,
            ApiKey.is_active == True,
        )
    )
    api_key = result.scalar_one_or_none()

    earned = api_key.earned_tokens if api_key else 0
    used = api_key.used_tokens if api_key else 0

    return WorkerInfo(
        id=worker.id,
        worker_uid=worker.worker_uid,
        name=worker.name,
        device_type=worker.device_type,
        device_name=worker.device_name,
        status=worker.status,
        total_contributions=worker.total_contributions,
        total_steps_trained=worker.total_steps_trained,
        trust_score=worker.trust_score,
        earned_tokens=earned,
        used_tokens=used,
        remaining_tokens=earned - used,
        first_seen=worker.first_seen,
        last_seen=worker.last_seen,
    )


# ============================================================================
# POST /leave — 명시적 이탈
# ============================================================================
@router.post("/leave")
async def leave(
    x_worker_key: str = Header(..., alias="X-Worker-Key"),
    session: AsyncSession = Depends(get_session),
):
    """
    워커가 명시적으로 이탈합니다.
    Ctrl+C 등으로 종료 시 호출됩니다.
    """
    worker = await get_worker_by_uid(uuid.UUID(x_worker_key), session)

    worker.status = WorkerStatus.OFFLINE
    worker.last_seen = datetime.utcnow()

    # 감사 로그
    session.add(AuditLog(
        event_type=AuditEventType.WORKER_LEFT,
        actor_id=worker.id,
        details={"name": worker.name},
    ))

    logger.info(f"워커 이탈: {worker.name}")

    return {"ok": True, "message": f"워커 '{worker.name}' 이탈 완료"}


# ============================================================================
# 헬퍼 함수
# ============================================================================
def _recommend_settings(
    device_type: str,
    gpu_memory_mb: int | None = None,
    benchmark_score: float | None = None,
) -> tuple[int, int]:
    """
    하드웨어 기반 권장 batch_size와 local_steps를 계산합니다.

    【기준】
    - GPU 24GB+: batch_size=64, local_steps=100
    - GPU 12GB:  batch_size=32, local_steps=100
    - GPU 8GB:   batch_size=16, local_steps=50
    - MPS (Apple Silicon): batch_size=16, local_steps=50
    - CPU: batch_size=4, local_steps=25
    """
    if device_type == "cuda" and gpu_memory_mb:
        if gpu_memory_mb >= 24000:
            return 64, 100
        elif gpu_memory_mb >= 12000:
            return 32, 100
        elif gpu_memory_mb >= 8000:
            return 16, 50
        else:
            return 8, 50
    elif device_type == "mps":
        if gpu_memory_mb and gpu_memory_mb >= 16000:
            return 16, 50
        return 8, 50
    else:  # cpu
        return 4, 25
