# distributed/common/constants.py
# ============================================================================
# 공통 상수 정의
# ============================================================================
# 서버와 워커가 공유하는 상수값들을 정의합니다.
# 상태 값, 이벤트 타입, 기본 설정 등이 포함됩니다.

# ============================================================================
# 워커 상태
# ============================================================================
class WorkerStatus:
    """워커의 현재 상태"""
    OFFLINE = "offline"       # 오프라인 (heartbeat 없음)
    ONLINE = "online"         # 온라인 (대기 중)
    TRAINING = "training"     # 학습 중
    UPLOADING = "uploading"   # 결과 업로드 중


# ============================================================================
# 기여(Contribution) 상태
# ============================================================================
class ContributionStatus:
    """워커가 제출한 기여의 처리 상태"""
    PENDING = "pending"         # 병합 대기 중
    VALIDATING = "validating"   # 검증 중
    MERGED = "merged"           # 병합 완료
    REJECTED = "rejected"       # 거부됨


# ============================================================================
# 실험(Experiment) 상태
# ============================================================================
class ExperimentStatus:
    """학습 실험의 상태"""
    ACTIVE = "active"       # 활성 (학습 진행 중)
    PAUSED = "paused"       # 일시 중지
    COMPLETED = "completed" # 완료


# ============================================================================
# 감사 로그 이벤트 타입
# ============================================================================
class AuditEventType:
    """시스템 이벤트 유형"""
    WORKER_JOINED = "worker_joined"
    WORKER_LEFT = "worker_left"
    WORKER_BANNED = "worker_banned"
    CONTRIBUTION_SUBMITTED = "contribution_submitted"
    CONTRIBUTION_MERGED = "contribution_merged"
    CONTRIBUTION_REJECTED = "contribution_rejected"
    CHECKPOINT_CREATED = "checkpoint_created"
    EXPERIMENT_CREATED = "experiment_created"
    EXPERIMENT_PAUSED = "experiment_paused"
    EXPERIMENT_COMPLETED = "experiment_completed"
    API_KEY_CREATED = "api_key_created"
    TOKENS_EARNED = "tokens_earned"
    TOKENS_SPENT = "tokens_spent"


# ============================================================================
# 토큰 거래 유형
# ============================================================================
class TokenTransactionType:
    """크레딧 적립/차감 유형"""
    EARN = "earn"       # 학습 기여로 적립
    SPEND = "spend"     # API 사용으로 차감
    BONUS = "bonus"     # 보너스 적립 (첫 참여, 연속 참여 등)
    EXPIRE = "expire"   # 만료


# ============================================================================
# 기본 설정값
# ============================================================================
class Defaults:
    """시스템 기본값"""

    # 병합 트리거
    MERGE_THRESHOLD = 3        # 대기 기여 수 (이 이상이면 병합 트리거)
    MERGE_TIMEOUT_SEC = 300    # 5분 (마지막 병합 후 이 시간 경과 시 병합)
    STEP_THRESHOLD = 100       # 대기 기여의 총 step 합 (이 이상이면 병합)

    # Stale 처리
    MAX_STALE_GAP = 200        # 허용 최대 stale gap (이 초과 시 기여 거부)

    # Heartbeat
    HEARTBEAT_INTERVAL_SEC = 30    # 워커 heartbeat 전송 간격
    OFFLINE_TIMEOUT_SEC = 60       # 이 시간 동안 heartbeat 없으면 오프라인

    # 학습 기본값
    DEFAULT_LOCAL_STEPS = 50       # 워커당 로컬 학습 스텝 수
    DEFAULT_BATCH_SIZE = 16        # 기본 배치 크기
    DEFAULT_LEARNING_RATE = 3e-4   # 기본 학습률
    DEFAULT_GRAD_CLIP = 1.0        # 기본 gradient clipping 값

    # API 크레딧
    DEFAULT_RATE_LIMIT_RPM = 60    # 분당 요청 제한
    DEFAULT_MAX_TOKENS_PER_REQ = 256  # 요청당 최대 토큰 수
    MAX_API_KEYS_PER_WORKER = 5    # 워커당 최대 API 키 수

    # 검증
    LOSS_ANOMALY_THRESHOLD = 3.0   # loss가 글로벌 평균의 3배 초과 시 의심
    WEIGHT_DELTA_THRESHOLD = 0.5   # 가중치 변화율 50% 초과 시 의심
    MIN_TRUST_SCORE = 0.3          # 이 미만이면 차단
    TRUST_SCORE_INCREMENT = 0.01   # 정상 기여 시 신뢰도 증가
    TRUST_SCORE_DECREMENT = 0.1    # 이상 기여 시 신뢰도 감소

    # 체크포인트 병합 루프
    MERGE_CHECK_INTERVAL_SEC = 10  # 병합 조건 확인 주기
