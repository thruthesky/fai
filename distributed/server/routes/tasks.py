# distributed/server/routes/tasks.py
# ============================================================================
# 작업 관리 API (3개 엔드포인트)
# ============================================================================

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_config
from ..database import get_session
from ..models import Checkpoint, Contribution, Experiment, Worker
from ..schemas import (
    ExperimentStatusResponse,
    TaskCompleteResponse,
    TaskRequestBody,
    TaskRequestResponse,
)
from ...common.constants import (
    ContributionStatus,
    Defaults,
    ExperimentStatus,
    WorkerStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# GET /experiments/{id}/status — 실험 상태 조회
# ============================================================================
@router.get("/experiments/{experiment_id}/status", response_model=ExperimentStatusResponse)
async def experiment_status(
    experiment_id: int,
    session: AsyncSession = Depends(get_session),
):
    """실험의 현재 상태를 조회합니다."""
    result = await session.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail="실험을 찾을 수 없습니다")

    # 활성 워커 수
    active_count = await session.execute(
        select(func.count()).select_from(Worker).where(
            Worker.status.in_([WorkerStatus.ONLINE, WorkerStatus.TRAINING])
        )
    )

    # 총 기여 수
    contrib_count = await session.execute(
        select(func.count()).select_from(Contribution).where(
            Contribution.experiment_id == experiment_id
        )
    )

    return ExperimentStatusResponse(
        id=exp.id,
        name=exp.name,
        status=exp.status,
        current_global_step=exp.current_global_step or 0,
        current_train_loss=exp.current_train_loss,
        current_val_loss=exp.current_val_loss,
        best_val_loss=exp.best_val_loss,
        active_workers=active_count.scalar() or 0,
        total_contributions=contrib_count.scalar() or 0,
        local_steps_per_round=exp.local_steps_per_round,
    )


# ============================================================================
# POST /tasks/request — 작업 요청
# ============================================================================
@router.post("/tasks/request", response_model=TaskRequestResponse)
async def request_task(
    req: TaskRequestBody,
    session: AsyncSession = Depends(get_session),
):
    """
    워커에게 학습 작업을 할당합니다.

    【동작】
    1. 실험 상태 확인 (active 여부)
    2. 최신 체크포인트 조회
    3. 워커 권장 설정 조회
    4. Contribution 레코드 생성 (작업 할당 추적)
    """
    # 워커 조회
    result = await session.execute(
        select(Worker).where(Worker.worker_uid == req.worker_uid)
    )
    worker = result.scalar_one_or_none()
    if worker is None:
        raise HTTPException(status_code=404, detail="등록되지 않은 워커입니다")

    # 실험 조회
    result = await session.execute(
        select(Experiment).where(Experiment.id == req.experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail="실험을 찾을 수 없습니다")
    if exp.status != ExperimentStatus.ACTIVE:
        raise HTTPException(status_code=400, detail=f"실험이 '{exp.status}' 상태입니다")

    # 최신 체크포인트 조회
    result = await session.execute(
        select(Checkpoint)
        .where(Checkpoint.experiment_id == exp.id, Checkpoint.is_latest == True)
        .order_by(Checkpoint.global_step.desc())
        .limit(1)
    )
    checkpoint = result.scalar_one_or_none()

    if checkpoint is None:
        # 초기 체크포인트 자동 생성 (첫 라운드)
        checkpoint = await _create_initial_checkpoint(session, exp)

    # 워커 상태 업데이트
    worker.status = WorkerStatus.TRAINING
    worker.last_seen = datetime.utcnow()

    # 체크포인트 다운로드 URL 생성
    config = get_config()
    checkpoint_url = f"/api/v1/checkpoints/{checkpoint.id}/download"

    # 워커별 설정
    batch_size = worker.recommended_batch_size or Defaults.DEFAULT_BATCH_SIZE
    local_steps = worker.recommended_local_steps or exp.local_steps_per_round

    # Contribution 레코드 생성 (작업 할당 추적)
    contribution = Contribution(
        experiment_id=exp.id,
        worker_id=worker.id,
        base_checkpoint_id=checkpoint.id,
        base_global_step=exp.current_global_step or 0,
        steps_trained=0,  # 완료 시 업데이트
        status="assigned",
        device_type=worker.device_type,
        batch_size_used=batch_size,
        learning_rate_used=Defaults.DEFAULT_LEARNING_RATE,
    )
    session.add(contribution)
    await session.flush()

    return TaskRequestResponse(
        task_id=contribution.id,
        experiment_id=exp.id,
        checkpoint_id=checkpoint.id,
        checkpoint_url=checkpoint_url,
        base_global_step=exp.current_global_step or 0,
        local_steps=local_steps,
        batch_size=batch_size,
        learning_rate=Defaults.DEFAULT_LEARNING_RATE,
        config=exp.config or {},
    )


# ============================================================================
# POST /tasks/{task_id}/complete — 학습 완료 보고
# ============================================================================
@router.post("/tasks/{task_id}/complete", response_model=TaskCompleteResponse)
async def complete_task(
    task_id: int,
    steps_trained: int = Form(...),
    local_train_loss: float = Form(...),
    local_val_loss: float = Form(...),
    training_duration_s: float = Form(...),
    device_type: str = Form(...),
    batch_size_used: int = Form(...),
    learning_rate_used: float = Form(...),
    weights: UploadFile = File(..., description="학습된 가중치 파일 (.pt)"),
    x_worker_key: str = Header(..., alias="X-Worker-Key"),
    session: AsyncSession = Depends(get_session),
):
    """
    학습 완료 결과를 업로드합니다.

    【동작】
    1. 업로드된 가중치 파일을 로컬 스토리지에 저장
    2. contributions 테이블 업데이트
    3. stale_gap 계산
    4. 상태를 'pending'으로 설정 (병합 대기)
    """
    # Contribution 조회
    result = await session.execute(
        select(Contribution).where(Contribution.id == task_id)
    )
    contribution = result.scalar_one_or_none()
    if contribution is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    # 실험 조회 (현재 글로벌 스텝 확인)
    result = await session.execute(
        select(Experiment).where(Experiment.id == contribution.experiment_id)
    )
    exp = result.scalar_one_or_none()

    # 가중치 파일 저장
    config = get_config()
    upload_dir = Path(config.storage_path) / "contributions"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_filename = f"contrib_{task_id}_{uuid.uuid4().hex[:8]}.pt"
    upload_path = upload_dir / upload_filename

    content = await weights.read()
    upload_path.write_bytes(content)

    # stale_gap 계산
    current_step = exp.current_global_step or 0
    stale_gap = current_step - contribution.base_global_step

    # 기여 업데이트
    contribution.steps_trained = steps_trained
    contribution.local_train_loss = local_train_loss
    contribution.local_val_loss = local_val_loss
    contribution.training_duration_s = training_duration_s
    contribution.device_type = device_type
    contribution.batch_size_used = batch_size_used
    contribution.learning_rate_used = learning_rate_used
    contribution.upload_path = str(upload_path)
    contribution.upload_size_bytes = len(content)
    contribution.stale_gap = stale_gap
    contribution.submitted_at = datetime.utcnow()

    # stale_gap 판정
    max_stale = exp.max_stale_gap or Defaults.MAX_STALE_GAP
    if stale_gap > max_stale:
        contribution.status = ContributionStatus.REJECTED
        contribution.rejection_reason = f"stale_gap({stale_gap}) > max({max_stale})"
        message = f"기여 거부: stale_gap이 너무 큽니다 ({stale_gap}). 최신 체크포인트를 다운로드하세요."
    else:
        contribution.status = ContributionStatus.PENDING
        message = "기여 접수 완료. 병합 대기 중입니다."

    # 워커 상태 업데이트
    result = await session.execute(
        select(Worker).where(Worker.id == contribution.worker_id)
    )
    worker = result.scalar_one_or_none()
    if worker:
        worker.status = WorkerStatus.ONLINE
        worker.last_seen = datetime.utcnow()

    logger.info(
        f"기여 접수: task_id={task_id}, worker={worker.name if worker else '?'}, "
        f"steps={steps_trained}, loss={local_val_loss:.4f}, "
        f"stale_gap={stale_gap}, status={contribution.status}"
    )

    return TaskCompleteResponse(
        ok=True,
        contribution_id=contribution.id,
        status=contribution.status,
        stale_gap=stale_gap,
        message=message,
    )


# ============================================================================
# 헬퍼: 초기 체크포인트 자동 생성
# ============================================================================
async def _create_initial_checkpoint(
    session: AsyncSession,
    experiment: Experiment,
) -> Checkpoint:
    """
    초기 체크포인트가 없을 때 새 GPT 모델을 생성하여 저장합니다.
    첫 번째 워커가 작업 요청 시 자동으로 호출됩니다.
    """
    import torch
    from ...common.model import GPT, GPTConfig

    # 모델 생성
    gpt_config = GPTConfig.from_dict(experiment.config)
    model = GPT(gpt_config)

    # 파일 저장
    config = get_config()
    ckpt_dir = Path(config.storage_path) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"exp{experiment.id}_initial.pt"
    torch.save(model.state_dict(), ckpt_path)

    # DB 레코드 생성
    checkpoint = Checkpoint(
        experiment_id=experiment.id,
        global_step=0,
        round_number=0,
        file_path=str(ckpt_path),
        file_size_bytes=ckpt_path.stat().st_size,
        is_latest=True,
        is_best=False,
        num_contributors=0,
    )
    session.add(checkpoint)
    await session.flush()

    logger.info(
        f"초기 체크포인트 생성: {ckpt_path.name} "
        f"({ckpt_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    return checkpoint
