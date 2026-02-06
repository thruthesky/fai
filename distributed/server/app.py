# distributed/server/app.py
# ============================================================================
# FastAPI 앱 진입점 (Coordinator 서버)
# ============================================================================
#
# 【실행 방법】
# uv run uvicorn distributed.server.app:app --host 0.0.0.0 --port 8000 --reload
#
# 【Swagger 문서】
# http://localhost:8000/docs
#

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .config import ServerConfig, get_config, set_config
from .database import close_db, create_tables, init_db
from .routes import checkpoints, metrics, tasks, workers
from .services.heartbeat import HeartbeatMonitor
from .services.merger import MergeLoop

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fai.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 라이프사이클 관리

    【시작 시】
    1. 서버 설정 로드 (.environments)
    2. DB 엔진 초기화 (Supabase PostgreSQL 접속)
    3. 테이블 생성 (없는 경우만)
    4. 스토리지 디렉토리 생성
    5. Heartbeat 모니터 시작

    【종료 시】
    1. Heartbeat 모니터 중지
    2. DB 엔진 종료
    """
    # ── 시작 ──
    logger.info("=" * 60)
    logger.info("FAI 분산 학습 Coordinator 서버 시작")
    logger.info("=" * 60)

    # 1. 설정 로드
    config = ServerConfig.from_env_file()
    set_config(config)
    logger.info(f"Supabase 호스트: {config.supabase_host}")

    # 2. DB 초기화
    await init_db(config)

    # 3. 테이블 생성
    await create_tables()

    # 4. 스토리지 디렉토리 생성
    storage_path = Path(config.storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    (storage_path / "checkpoints").mkdir(exist_ok=True)
    (storage_path / "contributions").mkdir(exist_ok=True)
    logger.info(f"스토리지 경로: {storage_path.resolve()}")

    # 5. Heartbeat 모니터 시작
    heartbeat_monitor = HeartbeatMonitor()
    heartbeat_task = asyncio.create_task(heartbeat_monitor.run())

    # 6. 병합 루프 시작 (FedAvg 자동 병합)
    merge_loop = MergeLoop()
    merge_task = asyncio.create_task(merge_loop.run())

    logger.info("서버 준비 완료 — 워커 연결 대기 중")
    logger.info("=" * 60)

    yield  # ── 앱 실행 중 ──

    # ── 종료 ──
    logger.info("서버 종료 중...")
    heartbeat_monitor.stop()
    merge_loop.stop()
    heartbeat_task.cancel()
    merge_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
    try:
        await merge_task
    except asyncio.CancelledError:
        pass
    await close_db()
    logger.info("서버 종료 완료")


# ============================================================================
# FastAPI 앱 생성
# ============================================================================
app = FastAPI(
    title="FAI 분산 학습 Coordinator",
    description="자발적 참여자들의 분산 GPT 학습을 관리하는 중앙 서버",
    version="1.0.0",
    lifespan=lifespan,
)

# 라우터 등록
app.include_router(workers.router, prefix="/api/v1/workers", tags=["워커 관리"])
app.include_router(tasks.router, prefix="/api/v1", tags=["작업 관리"])
app.include_router(checkpoints.router, prefix="/api/v1/checkpoints", tags=["체크포인트"])
app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["메트릭"])


# ============================================================================
# 헬스체크
# ============================================================================
@app.get("/health", tags=["시스템"])
async def health_check():
    """서버 헬스체크"""
    return {"status": "ok", "service": "fai-coordinator"}
