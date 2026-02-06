# distributed/server/services/heartbeat.py
# ============================================================================
# 워커 Heartbeat 모니터
# ============================================================================
# 주기적으로 DB를 확인하여 heartbeat가 끊긴 워커를 오프라인으로 전환합니다.
# PostgreSQL의 last_seen 컬럼 기반으로 동작합니다 (Redis 대체).

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy import text, update

from ..database import get_engine
from ..models import Worker
from ...common.constants import Defaults, WorkerStatus

logger = logging.getLogger(__name__)


class HeartbeatMonitor:
    """
    워커 생존 감시 (백그라운드 태스크)

    【동작 원리】
    - 60초마다 workers 테이블의 last_seen 확인
    - OFFLINE_TIMEOUT_SEC(60초) 이상 heartbeat 없는 워커를 'offline'으로 전환
    - 기존 Redis SETEX/TTL 패턴을 PostgreSQL last_seen 컬럼으로 대체
    """

    def __init__(self, interval: int = Defaults.OFFLINE_TIMEOUT_SEC):
        self._interval = interval
        self._running = True

    def stop(self):
        """모니터 중지"""
        self._running = False

    async def run(self):
        """
        메인 루프: 주기적으로 오프라인 워커 감지

        【SQL】
        UPDATE workers SET status = 'offline'
        WHERE status != 'offline'
          AND last_seen < NOW() - INTERVAL '60 seconds';
        """
        logger.info(f"Heartbeat 모니터 시작 (간격: {self._interval}초)")

        while self._running:
            try:
                await self._check_offline_workers()
            except Exception as e:
                logger.error(f"Heartbeat 모니터 오류: {e}")

            await asyncio.sleep(self._interval)

    async def _check_offline_workers(self):
        """오프라인 워커 감지 및 상태 업데이트"""
        engine = get_engine()

        async with engine.begin() as conn:
            # last_seen이 타임아웃 이전인 워커를 오프라인으로 전환
            # INTERVAL에는 바인드 변수를 직접 쓸 수 없으므로 make_interval 사용
            result = await conn.execute(
                text(
                    "UPDATE workers SET status = :offline_status, updated_at = NOW() "
                    "WHERE status != :offline_status "
                    "AND last_seen < NOW() - make_interval(secs => :timeout_sec)"
                ),
                {
                    "offline_status": WorkerStatus.OFFLINE,
                    "timeout_sec": Defaults.OFFLINE_TIMEOUT_SEC,
                },
            )

            if result.rowcount > 0:
                logger.info(f"오프라인 전환: {result.rowcount}대 워커")
