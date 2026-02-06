# distributed/server/services/merger.py
# ============================================================================
# FedAvg 병합 엔진
# ============================================================================
# 워커들의 로컬 학습 결과를 가중 평균(Federated Averaging)으로 병합합니다.
#
# 【동작 원리】
# 1. pending 상태의 기여가 MERGE_THRESHOLD(3) 이상 쌓이면 병합 시작
# 2. pg_advisory_lock으로 동시 병합 방지
# 3. 각 기여를 검증(validator) → 유효한 것만 병합
# 4. FedAvg: 가중 평균으로 글로벌 모델 업데이트
# 5. 새 체크포인트 저장 + 메트릭 기록
#
# 【PostgreSQL SKIP LOCKED 패턴】
# FOR UPDATE SKIP LOCKED로 기여를 잠금 → 다른 프로세스와 충돌 방지
# (Redis 분산 락 대체)

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_config
from ..database import get_engine
from ..models import (
    ApiKey,
    AuditLog,
    Checkpoint,
    Contribution,
    Experiment,
    TokenTransaction,
    TrainingMetric,
    Worker,
)
from ...common.constants import (
    ContributionStatus,
    Defaults,
    ExperimentStatus,
    WorkerStatus,
)
from .validator import ContributionValidator, ValidationResult

logger = logging.getLogger(__name__)


# ============================================================================
# FedAvg 병합
# ============================================================================
def fedavg_merge(
    base_state_dict: dict[str, torch.Tensor],
    contributions: list[tuple[dict[str, torch.Tensor], float]],
) -> OrderedDict:
    """
    Federated Averaging 가중 병합

    【알고리즘】
    global_weights = Σ (w_i * local_weights_i) / Σ w_i

    여기서 w_i는 각 기여의 merge_weight (steps × trust × stale_weight)

    Args:
        base_state_dict: 현재 글로벌 모델 가중치 (폴백용)
        contributions: [(state_dict, weight), ...] 목록

    Returns:
        병합된 새 글로벌 가중치
    """
    if not contributions:
        return OrderedDict(base_state_dict)

    # 가중치 합계 계산
    total_weight = sum(w for _, w in contributions)
    if total_weight <= 0:
        return OrderedDict(base_state_dict)

    # 가중 평균 계산
    merged = OrderedDict()
    for key in base_state_dict:
        # 각 기여의 해당 레이어를 가중 합산
        weighted_sum = torch.zeros_like(base_state_dict[key], dtype=torch.float32)
        for state_dict, weight in contributions:
            if key in state_dict:
                weighted_sum += state_dict[key].float() * (weight / total_weight)

        merged[key] = weighted_sum.to(base_state_dict[key].dtype)

    logger.info(
        f"FedAvg 완료: {len(contributions)}개 기여, "
        f"총 가중치={total_weight:.3f}"
    )

    return merged


# ============================================================================
# 병합 루프 (백그라운드 태스크)
# ============================================================================
class MergeLoop:
    """
    자동 병합 루프

    【동작 방식】
    1. 30초마다 pending 기여 확인
    2. MERGE_THRESHOLD 이상이면 병합 시작
    3. pg_advisory_lock으로 단일 실행 보장
    4. 검증 → 병합 → 체크포인트 저장 → 메트릭 기록 → 크레딧 적립

    【사용법 — app.py lifespan에서 시작】
    merge_loop = MergeLoop()
    asyncio.create_task(merge_loop.run())
    """

    def __init__(
        self,
        interval: int = 30,
        merge_threshold: int = Defaults.MERGE_THRESHOLD,
    ):
        self._interval = interval
        self._threshold = merge_threshold
        self._running = True
        self._validator = ContributionValidator()

    def stop(self):
        """루프 중지"""
        self._running = False

    async def run(self):
        """메인 병합 루프"""
        logger.info(
            f"병합 루프 시작 (간격: {self._interval}초, "
            f"임계값: {self._threshold}개)"
        )

        while self._running:
            try:
                await self._try_merge()
            except Exception as e:
                logger.error(f"병합 루프 오류: {e}", exc_info=True)

            await asyncio.sleep(self._interval)

    async def _try_merge(self):
        """
        병합 시도 (advisory_lock + SKIP LOCKED)

        【SQL 흐름】
        1. pg_try_advisory_lock(42) — 다른 프로세스가 병합 중이면 건너뜀
        2. SELECT contributions WHERE status='pending' FOR UPDATE SKIP LOCKED
        3. 기여 검증 + FedAvg 병합
        4. 새 체크포인트 저장
        5. pg_advisory_unlock(42)
        """
        from ..database import get_session_factory

        session_factory = get_session_factory()

        async with session_factory() as session:
            # Advisory Lock 획득 시도 (lock_id = 42)
            lock_result = await session.execute(
                text("SELECT pg_try_advisory_lock(42)")
            )
            got_lock = lock_result.scalar()

            if not got_lock:
                logger.debug("다른 프로세스가 병합 중 — 건너뜀")
                return

            try:
                await self._execute_merge(session)
            finally:
                # 락 해제
                await session.execute(text("SELECT pg_advisory_unlock(42)"))

    async def _execute_merge(self, session: AsyncSession):
        """실제 병합 실행"""
        # 활성 실험 조회
        exp_result = await session.execute(
            select(Experiment).where(
                Experiment.status == ExperimentStatus.ACTIVE
            )
        )
        experiments = exp_result.scalars().all()

        for experiment in experiments:
            await self._merge_experiment(session, experiment)

    async def _merge_experiment(
        self,
        session: AsyncSession,
        experiment: Experiment,
    ):
        """하나의 실험에 대해 병합을 수행합니다."""
        # pending 기여 조회 (SKIP LOCKED로 잠금)
        contrib_result = await session.execute(
            select(Contribution)
            .where(
                Contribution.experiment_id == experiment.id,
                Contribution.status == ContributionStatus.PENDING,
            )
            .with_for_update(skip_locked=True)
            .order_by(Contribution.submitted_at)
        )
        pending = contrib_result.scalars().all()

        if len(pending) < self._threshold:
            return  # 아직 부족

        logger.info(
            f"병합 시작: experiment={experiment.id}, "
            f"pending={len(pending)}개 기여"
        )

        # 현재 글로벌 모델 로드
        config = get_config()
        base_state_dict = await self._load_base_checkpoint(
            session, experiment, config.storage_path
        )

        # 각 기여 검증 및 가중치 로드
        valid_contributions: list[tuple[dict[str, torch.Tensor], float]] = []
        merged_from = []
        total_steps = 0

        for contrib in pending:
            # 기여 상태를 'validating'으로 변경
            contrib.status = ContributionStatus.VALIDATING

            # 가중치 파일 로드
            weight_path = Path(contrib.upload_path)
            if not weight_path.exists():
                contrib.status = ContributionStatus.REJECTED
                contrib.rejection_reason = "가중치 파일 없음"
                logger.warning(f"기여 {contrib.id}: 가중치 파일 없음 ({weight_path})")
                continue

            state_dict = torch.load(
                weight_path, map_location="cpu", weights_only=False
            )

            # state_dict 형식 처리
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            # 검증
            stale_gap = (experiment.current_global_step or 0) - (
                contrib.base_global_step or 0
            )
            contrib.stale_gap = max(stale_gap, 0)

            # 워커 trust_score 조회
            worker = await session.get(Worker, contrib.worker_id)
            trust_score = worker.trust_score if worker else 1.0

            result = self._validator.validate(
                state_dict=state_dict,
                local_train_loss=contrib.local_train_loss or 0.0,
                local_val_loss=contrib.local_val_loss or 0.0,
                stale_gap=contrib.stale_gap,
                current_global_loss=experiment.current_train_loss,
                trust_score=trust_score,
            )

            if result.is_valid:
                # 병합 가중치 = merge_weight × steps_trained
                effective_weight = result.merge_weight * contrib.steps_trained
                valid_contributions.append((state_dict, effective_weight))
                contrib.merge_weight = result.merge_weight
                contrib.validated_at = datetime.utcnow()

                merged_from.append({
                    "worker_id": contrib.worker_id,
                    "contribution_id": contrib.id,
                    "steps_trained": contrib.steps_trained,
                    "merge_weight": result.merge_weight,
                    "local_loss": contrib.local_train_loss,
                })
                total_steps += contrib.steps_trained
            else:
                contrib.status = ContributionStatus.REJECTED
                contrib.rejection_reason = result.rejection_reason
                logger.warning(
                    f"기여 {contrib.id} 거절: {result.rejection_reason}"
                )

        # 유효한 기여가 없으면 종료
        if not valid_contributions:
            await session.commit()
            logger.info("유효한 기여 없음 — 병합 건너뜀")
            return

        # FedAvg 병합
        merged_state_dict = fedavg_merge(base_state_dict, valid_contributions)

        # 새 체크포인트 저장
        new_global_step = (experiment.current_global_step or 0) + total_steps
        round_number = len(
            (await session.execute(
                select(Checkpoint).where(
                    Checkpoint.experiment_id == experiment.id
                )
            )).scalars().all()
        ) + 1

        # 파일 저장
        storage_path = Path(config.storage_path)
        ckpt_dir = storage_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_filename = f"exp{experiment.id}_step{new_global_step}_r{round_number}.pt"
        ckpt_path = ckpt_dir / ckpt_filename

        torch.save(merged_state_dict, ckpt_path)
        file_size = ckpt_path.stat().st_size

        # 기존 latest 플래그 해제
        await session.execute(
            update(Checkpoint)
            .where(
                Checkpoint.experiment_id == experiment.id,
                Checkpoint.is_latest == True,
            )
            .values(is_latest=False)
        )

        # 새 체크포인트 레코드
        # val_loss 계산 (기여들의 평균)
        avg_val_loss = sum(
            c.local_val_loss or 0 for c in pending
            if c.status != ContributionStatus.REJECTED
        ) / max(len(valid_contributions), 1)

        avg_train_loss = sum(
            c.local_train_loss or 0 for c in pending
            if c.status != ContributionStatus.REJECTED
        ) / max(len(valid_contributions), 1)

        is_best = (
            experiment.best_val_loss is None
            or avg_val_loss < experiment.best_val_loss
        )

        new_checkpoint = Checkpoint(
            experiment_id=experiment.id,
            global_step=new_global_step,
            round_number=round_number,
            file_path=str(ckpt_path),
            file_size_bytes=file_size,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            merged_from=merged_from,
            num_contributors=len(valid_contributions),
            is_latest=True,
            is_best=is_best,
        )
        session.add(new_checkpoint)

        # 실험 상태 업데이트
        experiment.current_global_step = new_global_step
        experiment.current_train_loss = avg_train_loss
        experiment.current_val_loss = avg_val_loss
        if is_best:
            experiment.best_val_loss = avg_val_loss

        # 기여 상태 업데이트 (merged)
        for contrib in pending:
            if contrib.status == ContributionStatus.VALIDATING:
                contrib.status = ContributionStatus.MERGED
                contrib.merged_at = datetime.utcnow()

        # 메트릭 기록
        active_workers = (await session.execute(
            select(Worker).where(
                Worker.status.in_([WorkerStatus.ONLINE, WorkerStatus.TRAINING])
            )
        )).scalars().all()

        metric = TrainingMetric(
            experiment_id=experiment.id,
            global_step=new_global_step,
            round_number=round_number,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            num_active_workers=len(active_workers),
            num_contributions=len(valid_contributions),
        )
        session.add(metric)

        # 워커 기여 통계 업데이트 + 크레딧 적립
        for info in merged_from:
            worker = await session.get(Worker, info["worker_id"])
            if worker:
                worker.total_contributions = (worker.total_contributions or 0) + 1
                worker.total_steps_trained = (
                    worker.total_steps_trained or 0
                ) + info["steps_trained"]

                # API 크레딧 적립 (1 학습 스텝 = 1 토큰 크레딧)
                await self._grant_credits(
                    session,
                    worker=worker,
                    contribution_id=info["contribution_id"],
                    steps_trained=info["steps_trained"],
                )

        # 감사 로그
        session.add(AuditLog(
            event_type="merge_completed",
            details={
                "experiment_id": experiment.id,
                "round_number": round_number,
                "global_step": new_global_step,
                "num_contributions": len(valid_contributions),
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "is_best": is_best,
            },
        ))

        await session.commit()

        logger.info(
            f"병합 완료: round={round_number}, "
            f"global_step={new_global_step}, "
            f"contributors={len(valid_contributions)}, "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"best={'Yes' if is_best else 'No'}"
        )

    # ========================================================================
    # 헬퍼: 기본 체크포인트 로드
    # ========================================================================
    async def _load_base_checkpoint(
        self,
        session: AsyncSession,
        experiment: Experiment,
        storage_path: str,
    ) -> dict[str, torch.Tensor]:
        """최신 글로벌 체크포인트를 로드합니다."""
        result = await session.execute(
            select(Checkpoint)
            .where(
                Checkpoint.experiment_id == experiment.id,
                Checkpoint.is_latest == True,
            )
            .order_by(Checkpoint.global_step.desc())
            .limit(1)
        )
        checkpoint = result.scalar_one_or_none()

        if checkpoint is None:
            # 첫 병합 — 새 모델의 초기 가중치
            from ...common.model import GPT, GPTConfig
            config = GPTConfig.from_dict(experiment.config)
            model = GPT(config)
            return model.state_dict()

        ckpt_path = Path(checkpoint.file_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트 파일 없음: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # dict에 'model' 키가 있는 형식 처리
        if isinstance(state_dict, dict) and "model" in state_dict:
            return state_dict["model"]

        return state_dict

    # ========================================================================
    # 헬퍼: 크레딧 적립
    # ========================================================================
    async def _grant_credits(
        self,
        session: AsyncSession,
        worker: Worker,
        contribution_id: int,
        steps_trained: int,
    ):
        """
        학습 기여에 대한 API 크레딧을 적립합니다.

        【크레딧 규칙】
        1 학습 스텝 = 1 API 토큰 크레딧
        """
        # 활성 API 키 조회 (없으면 자동 생성)
        ak_result = await session.execute(
            select(ApiKey).where(
                ApiKey.worker_id == worker.id,
                ApiKey.is_active == True,
            ).limit(1)
        )
        api_key = ak_result.scalar_one_or_none()

        if api_key is None:
            api_key = ApiKey(
                worker_id=worker.id,
                name=f"{worker.name} 기본 키",
            )
            session.add(api_key)
            await session.flush()  # ID 생성

        # 크레딧 적립
        api_key.earned_tokens = (api_key.earned_tokens or 0) + steps_trained
        balance = (api_key.earned_tokens or 0) - (api_key.used_tokens or 0)

        # 거래 기록
        session.add(TokenTransaction(
            worker_id=worker.id,
            api_key_id=api_key.id,
            type="earn",
            amount=steps_trained,
            balance_after=balance,
            contribution_id=contribution_id,
            steps_trained=steps_trained,
            description=f"학습 기여 ({steps_trained} 스텝)",
        ))
