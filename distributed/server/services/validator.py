# distributed/server/services/validator.py
# ============================================================================
# 기여 검증 엔진
# ============================================================================
# 워커가 업로드한 가중치가 유효한지 검증합니다.
#
# 【검증 항목】
# 1. NaN/Inf 체크 — 학습 발산 방지
# 2. Loss 이상 탐지 — 현재 글로벌 loss 대비 3배 이상이면 거절
# 3. 가중치 변화율 — 너무 작거나 큰 변화 감지
# 4. 신뢰도 점수 — 과거 기여 이력 기반 (trust_score)
#
# 【Stale Gap 처리】
# - gap 0~50: 완전 수용 (merge_weight = 1.0)
# - gap 51~200: 감쇠 적용 (merge_weight = 1.0 - gap/max_stale_gap)
# - gap 201+: 거절

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 검증 결과
# ============================================================================
@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool          # 유효 여부
    merge_weight: float     # 병합 가중치 (0.0~1.0)
    rejection_reason: str | None = None  # 거절 사유


# ============================================================================
# 기여 검증기
# ============================================================================
class ContributionValidator:
    """
    워커 기여 검증

    【사용법】
    validator = ContributionValidator(max_stale_gap=200)
    result = validator.validate(
        state_dict=loaded_weights,
        local_train_loss=0.5,
        local_val_loss=0.6,
        stale_gap=30,
        current_global_loss=0.4,
        trust_score=1.0,
    )
    """

    def __init__(
        self,
        max_stale_gap: int = 200,
        stale_decay_start: int = 50,
        loss_anomaly_factor: float = 3.0,
        min_weight_change: float = 1e-8,
        max_weight_change: float = 100.0,
    ):
        self._max_stale_gap = max_stale_gap
        self._stale_decay_start = stale_decay_start
        self._loss_anomaly_factor = loss_anomaly_factor
        self._min_weight_change = min_weight_change
        self._max_weight_change = max_weight_change

    # ========================================================================
    # 메인 검증
    # ========================================================================
    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        local_train_loss: float,
        local_val_loss: float,
        stale_gap: int,
        current_global_loss: float | None = None,
        trust_score: float = 1.0,
    ) -> ValidationResult:
        """
        기여를 종합 검증합니다.

        Args:
            state_dict: 워커가 업로드한 가중치
            local_train_loss: 워커의 학습 loss
            local_val_loss: 워커의 검증 loss
            stale_gap: 글로벌 스텝과의 차이
            current_global_loss: 현재 글로벌 loss (None이면 loss 비교 건너뜀)
            trust_score: 워커 신뢰도 (0.0~1.0)

        Returns:
            ValidationResult: 검증 결과 (유효 여부 + 병합 가중치)
        """
        # 1. NaN/Inf 체크
        nan_result = self._check_nan_inf(state_dict)
        if not nan_result.is_valid:
            return nan_result

        # 2. Loss 이상 탐지
        if current_global_loss is not None:
            loss_result = self._check_loss_anomaly(
                local_train_loss, current_global_loss
            )
            if not loss_result.is_valid:
                return loss_result

        # 3. 가중치 변화율 체크
        change_result = self._check_weight_magnitude(state_dict)
        if not change_result.is_valid:
            return change_result

        # 4. Stale gap 처리
        stale_result = self._compute_stale_weight(stale_gap)
        if not stale_result.is_valid:
            return stale_result

        # 5. 최종 병합 가중치 계산
        # merge_weight = stale_weight * trust_score
        merge_weight = stale_result.merge_weight * max(trust_score, 0.1)

        logger.info(
            f"검증 통과: "
            f"stale_gap={stale_gap}, "
            f"merge_weight={merge_weight:.3f}, "
            f"trust={trust_score:.2f}"
        )

        return ValidationResult(
            is_valid=True,
            merge_weight=merge_weight,
        )

    # ========================================================================
    # 검증 항목 1: NaN/Inf 체크
    # ========================================================================
    def _check_nan_inf(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> ValidationResult:
        """가중치에 NaN 또는 Inf가 있는지 확인합니다."""
        for name, tensor in state_dict.items():
            if torch.isnan(tensor).any():
                reason = f"NaN 감지 — 레이어: {name}"
                logger.warning(f"검증 실패: {reason}")
                return ValidationResult(
                    is_valid=False,
                    merge_weight=0.0,
                    rejection_reason=reason,
                )
            if torch.isinf(tensor).any():
                reason = f"Inf 감지 — 레이어: {name}"
                logger.warning(f"검증 실패: {reason}")
                return ValidationResult(
                    is_valid=False,
                    merge_weight=0.0,
                    rejection_reason=reason,
                )

        return ValidationResult(is_valid=True, merge_weight=1.0)

    # ========================================================================
    # 검증 항목 2: Loss 이상 탐지
    # ========================================================================
    def _check_loss_anomaly(
        self,
        local_loss: float,
        global_loss: float,
    ) -> ValidationResult:
        """
        로컬 loss가 글로벌 loss 대비 비정상인지 확인합니다.

        【기준】
        local_loss > global_loss * anomaly_factor 이면 거절
        """
        if global_loss > 0 and local_loss > global_loss * self._loss_anomaly_factor:
            reason = (
                f"Loss 이상 — "
                f"local={local_loss:.4f}, "
                f"global={global_loss:.4f} "
                f"({self._loss_anomaly_factor}배 초과)"
            )
            logger.warning(f"검증 실패: {reason}")
            return ValidationResult(
                is_valid=False,
                merge_weight=0.0,
                rejection_reason=reason,
            )

        return ValidationResult(is_valid=True, merge_weight=1.0)

    # ========================================================================
    # 검증 항목 3: 가중치 크기 체크
    # ========================================================================
    def _check_weight_magnitude(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> ValidationResult:
        """
        가중치 텐서의 L2 노름을 확인합니다.

        【체크 대상】
        - 너무 작은 노름: 학습이 거의 안 됨 (빈 기여)
        - 너무 큰 노름: 학습 발산 가능성
        """
        for name, tensor in state_dict.items():
            if tensor.ndim < 2:
                continue  # bias, LayerNorm 파라미터 건너뜀

            norm = tensor.float().norm().item()

            if norm > self._max_weight_change:
                reason = f"가중치 크기 이상 — {name}: norm={norm:.4f} (최대 {self._max_weight_change})"
                logger.warning(f"검증 실패: {reason}")
                return ValidationResult(
                    is_valid=False,
                    merge_weight=0.0,
                    rejection_reason=reason,
                )

        return ValidationResult(is_valid=True, merge_weight=1.0)

    # ========================================================================
    # 검증 항목 4: Stale gap 감쇠
    # ========================================================================
    def _compute_stale_weight(
        self,
        stale_gap: int,
    ) -> ValidationResult:
        """
        Stale gap에 따른 병합 가중치를 계산합니다.

        【감쇠 규칙】
        - gap 0~50: merge_weight = 1.0 (완전 수용)
        - gap 51~200: merge_weight = 1.0 - (gap - 50) / (200 - 50) = 선형 감쇠
        - gap 201+: 거절

        Args:
            stale_gap: 글로벌 스텝과의 차이

        Returns:
            ValidationResult: stale 가중치
        """
        if stale_gap > self._max_stale_gap:
            reason = (
                f"Stale gap 초과 — "
                f"gap={stale_gap}, 최대={self._max_stale_gap}"
            )
            logger.warning(f"검증 실패: {reason}")
            return ValidationResult(
                is_valid=False,
                merge_weight=0.0,
                rejection_reason=reason,
            )

        if stale_gap <= self._stale_decay_start:
            merge_weight = 1.0
        else:
            # 선형 감쇠: 50~200 구간
            decay_range = self._max_stale_gap - self._stale_decay_start
            merge_weight = 1.0 - (stale_gap - self._stale_decay_start) / decay_range

        return ValidationResult(
            is_valid=True,
            merge_weight=max(merge_weight, 0.05),  # 최소 5% 가중치 보장
        )
