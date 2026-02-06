# distributed/common/protocol.py
# ============================================================================
# API 통신 프로토콜 정의
# ============================================================================
# 서버와 워커 간 통신에 사용하는 URL 경로, 헤더 키, 이벤트 채널 등을 정의합니다.
# 양쪽에서 동일한 상수를 사용하여 불일치를 방지합니다.


# ============================================================================
# API 기본 경로
# ============================================================================
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"


# ============================================================================
# 인증 헤더
# ============================================================================
WORKER_KEY_HEADER = "X-Worker-Key"  # 워커 인증용 헤더 키 (worker_uid 전달)


# ============================================================================
# API 엔드포인트 경로
# ============================================================================
class Endpoints:
    """서버 API 엔드포인트 경로"""

    # 워커 관리
    WORKERS_REGISTER = f"{API_PREFIX}/workers/register"
    WORKERS_HEARTBEAT = f"{API_PREFIX}/workers/heartbeat"
    WORKERS_BENCHMARK = f"{API_PREFIX}/workers/benchmark"
    WORKERS_ME = f"{API_PREFIX}/workers/me"
    WORKERS_LEAVE = f"{API_PREFIX}/workers/leave"

    # 작업
    TASKS_REQUEST = f"{API_PREFIX}/tasks/request"

    @staticmethod
    def tasks_complete(task_id: int) -> str:
        return f"{API_PREFIX}/tasks/{task_id}/complete"

    @staticmethod
    def experiment_status(experiment_id: int) -> str:
        return f"{API_PREFIX}/experiments/{experiment_id}/status"

    # 체크포인트
    CHECKPOINTS_LATEST = f"{API_PREFIX}/checkpoints/latest"
    CHECKPOINTS_HISTORY = f"{API_PREFIX}/checkpoints/history"

    @staticmethod
    def checkpoint_download(checkpoint_id: int) -> str:
        return f"{API_PREFIX}/checkpoints/{checkpoint_id}/download"

    # 메트릭
    METRICS_SUMMARY = f"{API_PREFIX}/metrics/summary"
    METRICS_LOSS_HISTORY = f"{API_PREFIX}/metrics/loss-history"
    METRICS_LEADERBOARD = f"{API_PREFIX}/metrics/leaderboard"
    METRICS_WORKERS = f"{API_PREFIX}/metrics/workers"

    # WebSocket
    WS_EVENTS = f"{API_PREFIX}/ws/events"


# ============================================================================
# PostgreSQL LISTEN/NOTIFY 채널
# ============================================================================
class NotifyChannels:
    """PostgreSQL 이벤트 알림 채널명"""
    CHECKPOINT_UPDATED = "checkpoint_updated"
    EXPERIMENT_CONTROL = "experiment_control"
    WORKER_STATUS = "worker_status"


# ============================================================================
# Supabase Storage 버킷
# ============================================================================
class StorageBuckets:
    """Supabase Storage 버킷명"""
    CHECKPOINTS = "checkpoints"
    DATASETS = "datasets"
