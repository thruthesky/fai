#!/usr/bin/env python3
"""
migrate_crawl_data_to_db.py - 기존 크롤링 데이터를 PostgreSQL로 마이그레이션

기존 data/raw/ 폴더의 마크다운 파일과 data/crawling-index.json의 메타데이터를
PostgreSQL(Supabase)의 crawled_documents 테이블로 이관하는 일회성 스크립트.

실행:
    uv run python scripts/migrate_crawl_data_to_db.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from distributed.server.config import ServerConfig
from distributed.server.database import init_db, close_db, create_tables, get_session_factory, Base
from distributed.server.models import CrawledDocument


def parse_datetime(value: str | None) -> datetime | None:
    """
    ISO 8601 날짜 문자열을 naive datetime 객체로 변환한다.
    DB 컬럼이 TIMESTAMP WITHOUT TIME ZONE이므로 timezone 정보를 제거한다.
    """
    if not value:
        return None
    try:
        # 'Z' 접미사 처리 (UTC)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        # timezone 정보 제거 (asyncpg는 naive datetime만 허용)
        return dt.replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


async def migrate():
    """data/raw/ 파일과 crawling-index.json을 DB로 마이그레이션한다."""

    # 1. DB 초기화
    env_path = os.path.join(PROJECT_ROOT, ".environments")
    config = ServerConfig.from_env_file(env_path)
    await init_db(config)

    # crawled_documents 테이블 생성 (없으면)
    await create_tables()

    # 2. crawling-index.json 로드
    index_path = Path(PROJECT_ROOT) / "data" / "crawling-index.json"
    if not index_path.exists():
        print(f"[에러] 크롤링 인덱스를 찾을 수 없습니다: {index_path}")
        await close_db()
        return

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    urls_data = index_data.get("urls", {})
    print(f"크롤링 인덱스에서 {len(urls_data)}개 URL 발견")

    # 3. 각 URL에 대해 파일 읽기 + DB 저장
    session_factory = get_session_factory()

    migrated = 0
    skipped = 0
    failed = 0
    no_file = 0

    from sqlalchemy import select

    for url, info in urls_data.items():
        try:
            # 각 URL마다 별도 세션 사용 (에러 격리)
            async with session_factory() as session:
                # 이미 존재하는지 확인
                result = await session.execute(
                    select(CrawledDocument).where(CrawledDocument.url == url)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    skipped += 1
                    continue

                # 파일 경로 확인
                file_path_str = info.get("file_path", "")
                file_path = Path(PROJECT_ROOT) / file_path_str

                # 파일 내용 읽기
                content = None
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8")
                else:
                    no_file += 1
                    print(f"  [파일없음] {file_path_str}")

                # URL 파싱
                parsed = urlparse(url)
                domain = parsed.netloc
                url_path = parsed.path or "/"

                doc = CrawledDocument(
                    url=url,
                    domain=domain,
                    url_path=url_path,
                    content=content,
                    content_hash=info.get("content_hash"),
                    content_size=info.get("file_size", 0),
                    status=info.get("status", "completed"),
                    error=info.get("error"),
                    first_crawled=parse_datetime(info.get("first_crawled")),
                    last_crawled=parse_datetime(info.get("last_crawled")),
                    crawl_count=info.get("crawl_count", 1),
                )
                session.add(doc)
                await session.commit()
                migrated += 1

        except Exception as e:
            failed += 1
            print(f"  [에러] {url}: {e}")

    # 4. 결과 출력
    print(f"\n{'='*50}")
    print(f"마이그레이션 완료")
    print(f"  성공: {migrated}개")
    print(f"  건너뜀 (이미 존재): {skipped}개")
    print(f"  파일 없음: {no_file}개")
    print(f"  실패: {failed}개")
    print(f"  전체: {len(urls_data)}개")
    print(f"{'='*50}")

    await close_db()


if __name__ == "__main__":
    asyncio.run(migrate())
