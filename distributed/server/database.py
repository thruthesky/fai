# distributed/server/database.py
# ============================================================================
# Supabase PostgreSQL 연결, 세션 관리
# ============================================================================
# SQLAlchemy 비동기 엔진과 세션 팩토리를 관리합니다.
# FastAPI의 Depends를 통해 각 요청에 세션을 주입합니다.

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from .config import ServerConfig, get_config

logger = logging.getLogger(__name__)


# ============================================================================
# ORM 기반 클래스 (모든 모델이 상속)
# ============================================================================
class Base(DeclarativeBase):
    """SQLAlchemy ORM 기반 클래스"""
    pass


# ============================================================================
# 전역 엔진/세션 팩토리
# ============================================================================
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db(config: ServerConfig | None = None) -> None:
    """
    데이터베이스 엔진 초기화 (앱 시작 시 1회 호출)

    Args:
        config: 서버 설정 (None이면 .environments에서 자동 로드)
    """
    global _engine, _session_factory

    if config is None:
        config = get_config()

    logger.info(f"DB 연결 중: {config.supabase_host}:{config.postgres_port}")

    # Supavisor (transaction pooler) 사용 시 prepared statement 비활성화
    # pgbouncer/Supavisor는 prepared statement를 지원하지 않음
    connect_args = {}
    if config.pooler_tenant_id:
        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_cache_size"] = 0

    _engine = create_async_engine(
        config.database_url,
        pool_size=20,       # 동시 연결 수 (Supavisor 기본 풀 크기와 맞춤)
        max_overflow=10,    # 추가 허용 연결 수
        pool_timeout=30,    # 연결 대기 타임아웃 (초)
        pool_recycle=1800,  # 30분마다 연결 재생성 (stale 방지)
        echo=False,         # SQL 로그 출력 (디버그 시 True)
        connect_args=connect_args,
    )

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,  # 커밋 후 객체 재로드 방지 (성능)
    )

    logger.info("DB 엔진 초기화 완료")


async def close_db() -> None:
    """데이터베이스 엔진 종료 (앱 종료 시 호출)"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("DB 엔진 종료 완료")


async def create_tables() -> None:
    """
    모든 테이블 생성 (개발용)

    【주의】
    - 프로덕션에서는 Alembic 마이그레이션 사용 권장
    - 이 함수는 테이블이 없을 때만 생성 (기존 테이블은 건드리지 않음)
    """
    if _engine is None:
        raise RuntimeError("DB 엔진이 초기화되지 않았습니다. init_db()를 먼저 호출하세요.")

    # models.py에서 모든 ORM 모델을 임포트하여 Base.metadata에 등록
    from . import models  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("테이블 생성 완료 (이미 존재하는 테이블은 건너뜀)")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Depends용 세션 제공자

    사용법:
        @router.get("/example")
        async def example(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(Worker))
            ...
    """
    if _session_factory is None:
        raise RuntimeError("DB 세션 팩토리가 초기화되지 않았습니다.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_engine() -> AsyncEngine:
    """현재 엔진 반환 (LISTEN/NOTIFY 등 raw connection 필요 시)"""
    if _engine is None:
        raise RuntimeError("DB 엔진이 초기화되지 않았습니다.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """세션 팩토리 반환 (서비스 레이어에서 직접 세션 관리 시 사용)"""
    if _session_factory is None:
        raise RuntimeError("DB 세션 팩토리가 초기화되지 않았습니다.")
    return _session_factory
