# docker/Dockerfile
# ============================================================================
# FAI 분산 학습 시스템 — 컨테이너 이미지
# ============================================================================
#
# 【빌드】
# docker build -t fai-distributed -f docker/Dockerfile .
#
# 【용도】
# - Coordinator 서버 (uvicorn 실행)
# - 학습 워커 (python -m distributed.worker 실행)

FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지 (curl: 헬스체크용)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# uv 설치 (빠른 패키지 관리)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 의존성 설치 (캐시 레이어 활용)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 소스 코드 복사
COPY distributed/ distributed/
COPY data/ data/

# 기본 포트
EXPOSE 8000

# 기본 명령 (docker-compose에서 오버라이드)
CMD ["uv", "run", "uvicorn", "distributed.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
