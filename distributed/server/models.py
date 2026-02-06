# distributed/server/models.py
# ============================================================================
# SQLAlchemy ORM 모델 (9개 테이블)
# ============================================================================
# Supabase PostgreSQL에 생성되는 모든 테이블을 정의합니다.
# distributed-training-plan.md 섹션 4의 DDL을 ORM으로 변환했습니다.

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


# ============================================================================
# 1. workers — 워커(참여자) 관리
# ============================================================================
class Worker(Base):
    """
    참여자(워커) 정보 및 상태 관리

    【역할】
    - 워커 등록 시 하드웨어 정보 저장
    - heartbeat로 온라인 상태 추적
    - 기여도 및 신뢰도 관리
    """
    __tablename__ = "workers"

    id = Column(Integer, primary_key=True)
    worker_uid = Column(
        UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4
    )
    name = Column(Text, nullable=False)
    hostname = Column(Text)

    # 하드웨어 정보
    device_type = Column(Text, nullable=False)  # 'cuda', 'mps', 'cpu'
    device_name = Column(Text)                  # GPU 모델명 (예: 'RTX 4090')
    gpu_memory_mb = Column(Integer)
    ram_mb = Column(Integer)
    cpu_cores = Column(Integer)

    # 벤치마크 결과
    benchmark_score = Column(Float)
    recommended_batch_size = Column(Integer)
    recommended_local_steps = Column(Integer)

    # 상태
    status = Column(Text, default="offline")  # offline, online, training, uploading
    total_contributions = Column(Integer, default=0)
    total_steps_trained = Column(BigInteger, default=0)

    # 시간 기록
    first_seen = Column(DateTime, server_default=func.now())
    last_seen = Column(DateTime, server_default=func.now())

    # 신뢰도
    trust_score = Column(Float, default=1.0)
    is_banned = Column(Boolean, default=False)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 관계
    contributions = relationship("Contribution", back_populates="worker")
    api_keys = relationship("ApiKey", back_populates="worker")

    __table_args__ = (
        Index("idx_workers_status", "status"),
        Index("idx_workers_uid", "worker_uid"),
    )


# ============================================================================
# 2. experiments — 학습 실험 관리
# ============================================================================
class Experiment(Base):
    """
    학습 실험 설정 및 진행 상태

    【역할】
    - 모델 하이퍼파라미터 저장 (config JSONB)
    - 현재 글로벌 진행도 추적
    - 병합 전략 및 stale gap 설정
    """
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)          # 예: "fai-v1-small"
    description = Column(Text)
    status = Column(Text, default="active")      # active, paused, completed

    # 모델 설정 (GPTConfig.to_dict() 저장)
    config = Column(JSONB, nullable=False)

    # 학습 설정
    target_steps = Column(BigInteger)             # 목표 전체 스텝
    local_steps_per_round = Column(Integer, default=50)
    max_stale_gap = Column(Integer, default=200)
    merge_strategy = Column(Text, default="fedavg")

    # 현재 진행 상태
    current_global_step = Column(BigInteger, default=0)
    current_train_loss = Column(Float)
    current_val_loss = Column(Float)
    best_val_loss = Column(Float)

    # 데이터셋 무결성 확인용
    dataset_checksum = Column(Text)     # train.bin SHA256
    tokenizer_checksum = Column(Text)   # tokenizer.json SHA256

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 관계
    checkpoints = relationship("Checkpoint", back_populates="experiment")
    contributions = relationship("Contribution", back_populates="experiment")


# ============================================================================
# 3. checkpoints — 모델 체크포인트 버전 관리
# ============================================================================
class Checkpoint(Base):
    """
    모델 체크포인트 히스토리

    【역할】
    - 각 병합 후 새 체크포인트 저장
    - 어떤 워커들의 기여로 만들어졌는지 기록 (merged_from)
    - 최신/최고 체크포인트 플래그 관리
    """
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    global_step = Column(BigInteger, nullable=False)
    round_number = Column(Integer, nullable=False)

    # 파일 정보
    file_path = Column(Text, nullable=False)     # 저장 경로
    file_size_bytes = Column(BigInteger)
    file_checksum = Column(Text)                  # SHA256

    # 학습 지표
    train_loss = Column(Float)
    val_loss = Column(Float)

    # 병합 정보
    merged_from = Column(JSONB)                   # [{worker_id, steps, loss}, ...]
    num_contributors = Column(Integer, default=0)

    # 플래그
    is_latest = Column(Boolean, default=False)
    is_best = Column(Boolean, default=False)

    created_at = Column(DateTime, server_default=func.now())

    # 관계
    experiment = relationship("Experiment", back_populates="checkpoints")

    __table_args__ = (
        Index("idx_checkpoints_experiment", "experiment_id"),
        Index("idx_checkpoints_latest", "is_latest", postgresql_where=text("is_latest = TRUE")),
        Index("idx_checkpoints_step", "global_step"),
    )


# ============================================================================
# 4. contributions — 워커 학습 결과
# ============================================================================
class Contribution(Base):
    """
    워커가 제출한 학습 결과

    【역할】
    - 워커의 로컬 학습 결과 기록
    - FedAvg 병합의 입력 데이터
    - stale_gap으로 기여의 신선도 판단
    """
    __tablename__ = "contributions"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    worker_id = Column(Integer, ForeignKey("workers.id"), nullable=False)

    # 학습 기준점
    base_checkpoint_id = Column(Integer, ForeignKey("checkpoints.id"), nullable=False)
    base_global_step = Column(BigInteger, nullable=False)

    # 학습 결과
    steps_trained = Column(Integer, nullable=False)
    local_train_loss = Column(Float)
    local_val_loss = Column(Float)

    # 업로드 정보
    upload_path = Column(Text)
    upload_size_bytes = Column(BigInteger)
    upload_checksum = Column(Text)

    # 학습 환경
    device_type = Column(Text)
    batch_size_used = Column(Integer)
    learning_rate_used = Column(Float)
    training_duration_s = Column(Float)

    # 상태
    status = Column(Text, default="pending")  # pending, validating, merged, rejected
    rejection_reason = Column(Text)

    # 병합 정보
    merge_weight = Column(Float)
    stale_gap = Column(Integer)

    # 시간 기록
    submitted_at = Column(DateTime, server_default=func.now())
    validated_at = Column(DateTime)
    merged_at = Column(DateTime)

    # 관계
    experiment = relationship("Experiment", back_populates="contributions")
    worker = relationship("Worker", back_populates="contributions")

    __table_args__ = (
        Index("idx_contributions_experiment", "experiment_id"),
        Index("idx_contributions_worker", "worker_id"),
        Index("idx_contributions_status", "status"),
        Index(
            "idx_contributions_pending",
            "status",
            postgresql_where=text("status = 'pending'"),
        ),
    )


# ============================================================================
# 5. training_metrics — 학습 메트릭 히스토리
# ============================================================================
class TrainingMetric(Base):
    """학습 곡선 데이터 (대시보드/모니터링용)"""
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    global_step = Column(BigInteger, nullable=False)
    round_number = Column(Integer)
    train_loss = Column(Float)
    val_loss = Column(Float)
    num_active_workers = Column(Integer)
    num_contributions = Column(Integer)
    recorded_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_metrics_experiment_step", "experiment_id", "global_step"),
    )


# ============================================================================
# 6. audit_log — 시스템 감사 로그
# ============================================================================
class AuditLog(Base):
    """시스템 이벤트 감사 추적 (투명성 및 문제 추적)"""
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True)
    event_type = Column(Text, nullable=False)
    actor_id = Column(Integer)          # worker_id 또는 NULL (시스템)
    details = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_audit_event", "event_type"),
        Index("idx_audit_time", "created_at"),
    )


# ============================================================================
# 7. api_keys — API 키 관리 (크레딧 시스템)
# ============================================================================
class ApiKey(Base):
    """
    API 키 및 크레딧 관리

    【크레딧 계산】
    잔여 크레딧 = earned_tokens - used_tokens
    """
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True)
    worker_id = Column(Integer, ForeignKey("workers.id"), nullable=False)
    api_key = Column(
        UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4
    )
    name = Column(Text)                      # 키 이름 (예: "내 앱용")

    is_active = Column(Boolean, default=True)
    earned_tokens = Column(BigInteger, default=0)
    used_tokens = Column(BigInteger, default=0)

    rate_limit_rpm = Column(Integer, default=60)
    max_tokens_per_request = Column(Integer, default=256)

    created_at = Column(DateTime, server_default=func.now())
    last_used_at = Column(DateTime)
    expires_at = Column(DateTime)

    # 관계
    worker = relationship("Worker", back_populates="api_keys")

    __table_args__ = (
        Index("idx_api_keys_key", "api_key"),
        Index("idx_api_keys_worker", "worker_id"),
    )


# ============================================================================
# 8. api_usage_log — API 사용 이력
# ============================================================================
class ApiUsageLog(Base):
    """API 호출 로깅 (토큰 소비 추적, rate limiting)"""
    __tablename__ = "api_usage_log"

    id = Column(Integer, primary_key=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    worker_id = Column(Integer, ForeignKey("workers.id"), nullable=False)
    endpoint = Column(Text, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    model_version = Column(Text)
    response_time_ms = Column(Integer)
    status_code = Column(Integer, default=200)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_api_usage_key", "api_key_id"),
        Index("idx_api_usage_worker", "worker_id"),
        Index("idx_api_usage_time", "created_at"),
    )


# ============================================================================
# 9. token_transactions — 크레딧 거래 기록
# ============================================================================
class TokenTransaction(Base):
    """
    크레딧 적립/차감 거래 기록 (투명성 보장)

    【거래 유형】
    - earn: 학습 기여로 적립
    - spend: API 사용으로 차감
    - bonus: 보너스 (첫 참여, 연속 참여 등)
    - expire: 만료
    """
    __tablename__ = "token_transactions"

    id = Column(Integer, primary_key=True)
    worker_id = Column(Integer, ForeignKey("workers.id"), nullable=False)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"))  # 차감 시에만
    type = Column(Text, nullable=False)      # earn, spend, bonus, expire
    amount = Column(BigInteger, nullable=False)  # 양수(적립) 또는 음수(차감)
    balance_after = Column(BigInteger, nullable=False)
    contribution_id = Column(Integer, ForeignKey("contributions.id"))
    steps_trained = Column(Integer)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_token_tx_worker", "worker_id"),
        Index("idx_token_tx_type", "type"),
    )
