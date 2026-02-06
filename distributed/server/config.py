# distributed/server/config.py
# ============================================================================
# 서버 설정 (.environments 파일 로드)
# ============================================================================
# .environments 파일에서 Supabase PostgreSQL 접속 정보를 로드합니다.
# 워커 설정은 distributed/worker/config.py에서 관리합니다.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_environments(env_path: str | None = None) -> dict[str, str]:
    """
    .environments 파일에서 KEY=VALUE 형태의 설정을 로드합니다.

    【파일 형식】
    - 빈 줄과 # 주석은 무시
    - KEY=VALUE 형식 (export 없음)
    - 따옴표 없이 값만 기재

    Args:
        env_path: 환경 변수 파일 경로 (None이면 프로젝트 루트의 .environments)

    Returns:
        환경 변수 딕셔너리
    """
    if env_path is None:
        # 프로젝트 루트에서 .environments 파일 탐색
        # distributed/server/config.py → 3단계 상위 = 프로젝트 루트
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = str(project_root / ".environments")

    env_vars: dict[str, str] = {}
    env_file = Path(env_path)

    if not env_file.exists():
        print(f"  [경고] 환경 변수 파일을 찾을 수 없습니다: {env_path}")
        return env_vars

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # 빈 줄 및 주석 무시
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            env_vars[key.strip()] = value.strip()

    return env_vars


@dataclass
class ServerConfig:
    """
    서버 설정 (Supabase PostgreSQL + Storage)

    【접속 방식】
    - 서버 → Supabase PostgreSQL: asyncpg 직접 접속 (포트 5432)
    - 워커 → 서버: REST API (포트 8000)
    - 워커는 DB에 직접 접속하지 않음
    """

    # Supabase 접속 정보
    supabase_host: str = "localhost"
    postgres_password: str = ""
    postgres_db: str = "postgres"
    postgres_port: int = 5432
    pooler_port: int = 6543
    pooler_tenant_id: str = ""

    # Supabase 인증 키
    anon_key: str = ""
    service_role_key: str = ""

    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8000

    # 로컬 스토리지 경로 (체크포인트/데이터셋 파일 저장)
    storage_path: str = "storage"

    @property
    def database_url(self) -> str:
        """
        asyncpg용 비동기 PostgreSQL 접속 URL

        【참고】
        - Supabase 자체 호스팅에서는 Supavisor pooler 포트(6543) 사용
        - 사용자명 형식: postgres.{tenant_id} (Supavisor 인증)
        - pooler_tenant_id가 있으면 Supavisor 경유, 없으면 직접 접속
        """
        if self.pooler_tenant_id:
            # Supavisor pooler 경유 접속
            user = f"postgres.{self.pooler_tenant_id}"
            port = self.pooler_port
        else:
            user = "postgres"
            port = self.postgres_port
        return (
            f"postgresql+asyncpg://{user}:{self.postgres_password}"
            f"@{self.supabase_host}:{port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """동기 PostgreSQL 접속 URL (Alembic 마이그레이션용)"""
        if self.pooler_tenant_id:
            user = f"postgres.{self.pooler_tenant_id}"
            port = self.pooler_port
        else:
            user = "postgres"
            port = self.postgres_port
        return (
            f"postgresql://{user}:{self.postgres_password}"
            f"@{self.supabase_host}:{port}/{self.postgres_db}"
        )

    @property
    def storage_url(self) -> str:
        """Supabase Storage REST API URL"""
        return f"http://{self.supabase_host}/storage/v1"

    @classmethod
    def from_env_file(cls, env_path: str | None = None) -> ServerConfig:
        """
        .environments 파일에서 설정을 로드하여 ServerConfig 인스턴스 생성

        Args:
            env_path: 환경 변수 파일 경로 (None이면 자동 탐색)

        Returns:
            ServerConfig 인스턴스
        """
        env = load_environments(env_path)

        return cls(
            supabase_host=env.get("SUPABASE_HOST", "localhost"),
            postgres_password=env.get("POSTGRES_PASSWORD", ""),
            postgres_db=env.get("POSTGRES_DB", "postgres"),
            postgres_port=int(env.get("POSTGRES_PORT", "5432")),
            pooler_port=int(env.get("POOLER_PROXY_PORT_TRANSACTION", "6543")),
            pooler_tenant_id=env.get("POOLER_TENANT_ID", ""),
            anon_key=env.get("ANON_KEY", ""),
            service_role_key=env.get("SERVICE_ROLE_KEY", ""),
            storage_path=os.environ.get("STORAGE_PATH", "storage"),
        )


# 전역 설정 인스턴스 (app.py에서 초기화)
_config: ServerConfig | None = None


def get_config() -> ServerConfig:
    """현재 서버 설정을 반환합니다."""
    global _config
    if _config is None:
        _config = ServerConfig.from_env_file()
    return _config


def set_config(config: ServerConfig) -> None:
    """서버 설정을 지정합니다 (테스트용)."""
    global _config
    _config = config
