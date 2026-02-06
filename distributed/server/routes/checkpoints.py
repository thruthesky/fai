# distributed/server/routes/checkpoints.py
# ============================================================================
# 체크포인트 관리 API (3개 엔드포인트)
# ============================================================================

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import Checkpoint
from ..schemas import CheckpointHistoryResponse, CheckpointInfo

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# GET /latest — 최신 체크포인트 다운로드
# ============================================================================
@router.get("/latest")
async def download_latest(
    experiment_id: int = 1,
    session: AsyncSession = Depends(get_session),
):
    """
    최신 체크포인트를 다운로드합니다.

    워커가 학습 시작 전 최신 글로벌 모델을 받는 데 사용합니다.
    """
    result = await session.execute(
        select(Checkpoint)
        .where(
            Checkpoint.experiment_id == experiment_id,
            Checkpoint.is_latest == True,
        )
        .order_by(Checkpoint.global_step.desc())
        .limit(1)
    )
    checkpoint = result.scalar_one_or_none()

    if checkpoint is None:
        raise HTTPException(status_code=404, detail="체크포인트가 없습니다")

    file_path = Path(checkpoint.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="체크포인트 파일을 찾을 수 없습니다")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


# ============================================================================
# GET /{checkpoint_id}/download — 특정 체크포인트 다운로드
# ============================================================================
@router.get("/{checkpoint_id}/download")
async def download_checkpoint(
    checkpoint_id: int,
    session: AsyncSession = Depends(get_session),
):
    """특정 체크포인트를 다운로드합니다."""
    result = await session.execute(
        select(Checkpoint).where(Checkpoint.id == checkpoint_id)
    )
    checkpoint = result.scalar_one_or_none()

    if checkpoint is None:
        raise HTTPException(status_code=404, detail="체크포인트를 찾을 수 없습니다")

    file_path = Path(checkpoint.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="체크포인트 파일을 찾을 수 없습니다")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


# ============================================================================
# GET /history — 체크포인트 히스토리
# ============================================================================
@router.get("/history", response_model=CheckpointHistoryResponse)
async def checkpoint_history(
    experiment_id: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """체크포인트 버전 히스토리를 조회합니다."""
    result = await session.execute(
        select(Checkpoint)
        .where(Checkpoint.experiment_id == experiment_id)
        .order_by(Checkpoint.global_step.desc())
        .limit(limit)
    )
    checkpoints = result.scalars().all()

    items = [
        CheckpointInfo(
            id=c.id,
            global_step=c.global_step,
            round_number=c.round_number,
            train_loss=c.train_loss,
            val_loss=c.val_loss,
            num_contributors=c.num_contributors or 0,
            is_latest=c.is_latest or False,
            is_best=c.is_best or False,
            file_size_bytes=c.file_size_bytes,
            created_at=c.created_at,
        )
        for c in checkpoints
    ]

    return CheckpointHistoryResponse(
        checkpoints=items,
        total=len(items),
    )
