# distributed/server/routes/metrics.py
# ============================================================================
# 메트릭 API (4개 엔드포인트)
# ============================================================================

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import (
    ApiKey,
    Contribution,
    Experiment,
    TrainingMetric,
    Worker,
)
from ..schemas import (
    LeaderboardEntry,
    LossHistoryEntry,
    MetricsSummaryResponse,
    WorkerStatusEntry,
)
from ...common.constants import WorkerStatus

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# GET /summary — 전체 메트릭 요약
# ============================================================================
@router.get("/summary", response_model=MetricsSummaryResponse)
async def metrics_summary(
    experiment_id: int = 1,
    session: AsyncSession = Depends(get_session),
):
    """실험의 전체 메트릭 요약을 반환합니다."""
    # 실험 조회
    result = await session.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail="실험을 찾을 수 없습니다")

    # 활성 워커 수
    active_count = (await session.execute(
        select(func.count()).select_from(Worker).where(
            Worker.status.in_([WorkerStatus.ONLINE, WorkerStatus.TRAINING])
        )
    )).scalar() or 0

    # 전체 워커 수
    total_workers = (await session.execute(
        select(func.count()).select_from(Worker)
    )).scalar() or 0

    # 전체 기여 수
    total_contribs = (await session.execute(
        select(func.count()).select_from(Contribution).where(
            Contribution.experiment_id == experiment_id
        )
    )).scalar() or 0

    # 전체 학습 스텝 합
    total_steps = (await session.execute(
        select(func.sum(Worker.total_steps_trained))
    )).scalar() or 0

    return MetricsSummaryResponse(
        experiment_name=exp.name,
        global_step=exp.current_global_step or 0,
        current_train_loss=exp.current_train_loss,
        current_val_loss=exp.current_val_loss,
        best_val_loss=exp.best_val_loss,
        active_workers=active_count,
        total_workers=total_workers,
        total_contributions=total_contribs,
        total_steps_trained=total_steps,
    )


# ============================================================================
# GET /loss-history — Loss 추이
# ============================================================================
@router.get("/loss-history", response_model=list[LossHistoryEntry])
async def loss_history(
    experiment_id: int = 1,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
):
    """학습/검증 Loss 히스토리를 반환합니다."""
    result = await session.execute(
        select(TrainingMetric)
        .where(TrainingMetric.experiment_id == experiment_id)
        .order_by(desc(TrainingMetric.global_step))
        .limit(limit)
    )
    metrics = result.scalars().all()

    return [
        LossHistoryEntry(
            global_step=m.global_step,
            train_loss=m.train_loss,
            val_loss=m.val_loss,
            num_contributors=m.num_contributions or 0,
            recorded_at=m.recorded_at,
        )
        for m in reversed(metrics)  # 시간순 정렬
    ]


# ============================================================================
# GET /leaderboard — 기여도 리더보드
# ============================================================================
@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def leaderboard(
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """기여도 상위 워커 리더보드를 반환합니다."""
    result = await session.execute(
        select(Worker)
        .where(Worker.total_contributions > 0)
        .order_by(desc(Worker.total_steps_trained))
        .limit(limit)
    )
    workers = result.scalars().all()

    entries = []
    for rank, w in enumerate(workers, 1):
        # API 키 잔액 조회
        ak_result = await session.execute(
            select(ApiKey).where(
                ApiKey.worker_id == w.id,
                ApiKey.is_active == True,
            ).limit(1)
        )
        ak = ak_result.scalar_one_or_none()

        entries.append(LeaderboardEntry(
            rank=rank,
            worker_name=w.name,
            total_contributions=w.total_contributions,
            total_steps_trained=w.total_steps_trained,
            earned_tokens=ak.earned_tokens if ak else 0,
            trust_score=w.trust_score,
        ))

    return entries


# ============================================================================
# GET /workers — 활성 워커 목록
# ============================================================================
@router.get("/workers", response_model=list[WorkerStatusEntry])
async def active_workers(
    session: AsyncSession = Depends(get_session),
):
    """현재 활성 워커 목록을 반환합니다."""
    result = await session.execute(
        select(Worker)
        .where(Worker.status != WorkerStatus.OFFLINE)
        .order_by(desc(Worker.last_seen))
    )
    workers = result.scalars().all()

    return [
        WorkerStatusEntry(
            name=w.name,
            device_type=w.device_type,
            status=w.status,
            last_seen=w.last_seen,
            total_steps_trained=w.total_steps_trained,
        )
        for w in workers
    ]
