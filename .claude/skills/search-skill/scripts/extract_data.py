#!/usr/bin/env python3
"""
extract_data.py - Dart/Flutter 공식 문서 크롤링 도구 (PostgreSQL 저장)

FAI 프로젝트 Stage 1 데이터 수집용 스크립트.
WebFetch 결과를 받아서 PostgreSQL(Supabase)의 crawled_documents 테이블에 직접 저장한다.
기존 data/raw/ 파일 시스템 저장 방식을 대체한다.

Usage:
    # 단일 페이지 저장 (DB에 직접 저장)
    uv run python .claude/skills/search-skill/scripts/extract_data.py \\
        --url "https://dart.dev/language/variables" --content "..."

    # stdin에서 JSON 입력
    echo '{"url": "...", "content": "..."}' | \\
        uv run python .claude/skills/search-skill/scripts/extract_data.py

    # 크롤링 시드 URL 생성
    uv run python .claude/skills/search-skill/scripts/extract_data.py --sitemap dart.dev

    # 수집 현황 확인 (DB 조회)
    uv run python .claude/skills/search-skill/scripts/extract_data.py --status
    uv run python .claude/skills/search-skill/scripts/extract_data.py --status --domain dart.dev
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone, UTC
from typing import Optional
from urllib.parse import urlparse

# ============================================================================
# 프로젝트 루트를 sys.path에 추가하여 distributed.server 패키지 import 가능하게 설정
# extract_data.py 위치: .claude/skills/search-skill/scripts/extract_data.py
# 프로젝트 루트: 4단계 상위 디렉토리
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# DB 초기화/종료 함수 (distributed.server 인프라 재사용)
# ============================================================================
async def init_crawl_db() -> None:
    """
    크롤링용 DB 연결을 초기화한다.
    distributed/server/의 기존 DB 인프라를 재사용한다.
    .environments 파일에서 Supabase 접속 정보를 로드한다.
    """
    from distributed.server.config import ServerConfig
    from distributed.server.database import init_db, create_tables

    # .environments 파일 경로를 명시적으로 지정
    env_path = os.path.join(PROJECT_ROOT, ".environments")
    config = ServerConfig.from_env_file(env_path)
    await init_db(config)

    # crawled_documents 테이블이 없으면 자동 생성
    await create_tables()


async def close_crawl_db() -> None:
    """DB 연결을 종료한다."""
    from distributed.server.database import close_db
    await close_db()


# ============================================================================
# 콘텐츠 처리 함수 (파일 시스템과 무관하므로 그대로 유지)
# ============================================================================
def extract_links_from_content(content: str, base_url: str) -> list[str]:
    """
    마크다운 콘텐츠에서 내부 링크를 추출한다.

    Args:
        content: 마크다운 콘텐츠
        base_url: 기준 URL (같은 도메인 링크 필터링용)

    Returns:
        추출된 내부 링크 목록
    """
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc

    # 마크다운 링크 패턴: [text](url) 또는 일반 URL
    link_patterns = [
        r'\[([^\]]*)\]\(([^)]+)\)',  # [text](url)
        r'href=["\']([^"\']+)["\']',  # href="url"
    ]

    links = []
    for pattern in link_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # 튜플이면 URL 부분만 추출
            url = match[1] if isinstance(match, tuple) else match

            # 상대 경로 처리
            if url.startswith("/"):
                url = f"{parsed_base.scheme}://{base_domain}{url}"
            elif not url.startswith("http"):
                continue  # 앵커나 javascript 등 스킵

            # 같은 도메인인지 확인
            parsed_url = urlparse(url)
            if parsed_url.netloc == base_domain:
                # 쿼리 파라미터와 앵커 제거
                clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                if clean_url not in links:
                    links.append(clean_url)

    return links


def format_markdown_content(content: str, url: str, title: Optional[str] = None) -> str:
    """
    WebFetch 결과를 정리된 마크다운 형식으로 변환한다.

    Args:
        content: WebFetch로 가져온 원본 콘텐츠
        url: 원본 URL
        title: 페이지 제목 (없으면 URL에서 추출)

    Returns:
        메타데이터가 추가된 정리된 마크다운
    """
    # 제목이 없으면 URL 경로에서 추출
    if not title:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if path:
            title = path.split("/")[-1].replace("-", " ").replace("_", " ").title()
        else:
            title = parsed.netloc

    # 콘텐츠 앞뒤 공백 정리
    content = content.strip()

    # 이미 제목이 있는지 확인
    if not content.startswith("# "):
        content = f"# {title}\n\n{content}"

    # 출처 메타데이터 추가
    fetch_date = datetime.now().strftime("%Y-%m-%d")
    source_section = f"""

---

## Source

- **URL**: {url}
- **Fetched**: {fetch_date}
"""

    return content + source_section


# ============================================================================
# DB 저장 함수 (기존 파일 시스템 저장을 대체)
# ============================================================================
async def save_page_to_db(url: str, content: str, title: Optional[str] = None) -> dict:
    """
    페이지 콘텐츠를 PostgreSQL에 저장한다 (UPSERT).
    기존 URL이 있으면 업데이트, 없으면 새로 삽입한다.

    Args:
        url: 페이지 URL
        content: 페이지 콘텐츠 (마크다운 원본)
        title: 페이지 제목 (선택)

    Returns:
        {"url": url, "status": "saved", "content_size": ...}
    """
    from sqlalchemy import select
    from distributed.server.database import get_session_factory
    from distributed.server.models import CrawledDocument

    parsed = urlparse(url)
    domain = parsed.netloc
    url_path = parsed.path or "/"

    # 콘텐츠 포맷팅 (출처 메타데이터 추가)
    formatted = format_markdown_content(content, url, title)
    content_hash = hashlib.md5(formatted.encode("utf-8")).hexdigest()
    content_size = len(formatted.encode("utf-8"))
    # DB 컬럼이 TIMESTAMP WITHOUT TIME ZONE이므로 naive datetime 사용
    now = datetime.now(UTC).replace(tzinfo=None)

    session_factory = get_session_factory()
    async with session_factory() as session:
        # 기존 레코드 확인
        result = await session.execute(
            select(CrawledDocument).where(CrawledDocument.url == url)
        )
        doc = result.scalar_one_or_none()

        if doc:
            # 기존 레코드 업데이트
            doc.content = formatted
            doc.content_hash = content_hash
            doc.content_size = content_size
            doc.title = title or doc.title
            doc.status = "completed"
            doc.last_crawled = now
            doc.crawl_count = (doc.crawl_count or 0) + 1
            doc.error = None
        else:
            # 신규 레코드 삽입
            doc = CrawledDocument(
                url=url,
                domain=domain,
                url_path=url_path,
                title=title,
                content=formatted,
                content_hash=content_hash,
                content_size=content_size,
                status="completed",
                first_crawled=now,
                last_crawled=now,
                crawl_count=1,
                error=None,
            )
            session.add(doc)

        await session.commit()

    return {"url": url, "status": "saved", "content_size": content_size}


async def process_batch_to_db(pages: list[dict]) -> dict:
    """
    여러 페이지를 배치로 DB에 저장한다.

    Args:
        pages: [{"url": "...", "content": "...", "title": "..."}] 형식의 페이지 목록

    Returns:
        처리 결과 요약
    """
    results = {
        "saved": [],
        "failed": [],
        "total": len(pages)
    }

    for page in pages:
        url = page.get("url")
        content = page.get("content")
        title = page.get("title")

        if not url or not content:
            results["failed"].append({"url": url, "error": "URL 또는 콘텐츠 누락"})
            continue

        try:
            result = await save_page_to_db(url, content, title)
            results["saved"].append(result)
        except Exception as e:
            results["failed"].append({"url": url, "error": str(e)})

    return results


# ============================================================================
# DB 조회 함수 (기존 파일 시스템 조회를 대체)
# ============================================================================
async def list_collected_docs(domain: Optional[str] = None) -> list[dict]:
    """
    DB에서 수집된 문서 목록을 조회한다.

    Args:
        domain: 특정 도메인만 필터 (선택)

    Returns:
        수집된 문서 정보 목록
    """
    from sqlalchemy import select
    from distributed.server.database import get_session_factory
    from distributed.server.models import CrawledDocument

    session_factory = get_session_factory()
    async with session_factory() as session:
        query = select(CrawledDocument).order_by(CrawledDocument.url)
        if domain:
            query = query.where(CrawledDocument.domain == domain)

        result = await session.execute(query)
        docs = result.scalars().all()

        return [
            {
                "url": doc.url,
                "domain": doc.domain,
                "status": doc.status,
                "content_size": doc.content_size,
                "crawl_count": doc.crawl_count,
                "last_crawled": doc.last_crawled.isoformat() if doc.last_crawled else None,
            }
            for doc in docs
        ]


async def get_collected_urls_from_db(domain: str) -> set[str]:
    """
    DB에서 특정 도메인의 수집 완료된 URL 목록을 조회한다.

    Args:
        domain: 도메인

    Returns:
        수집된 URL set
    """
    from sqlalchemy import select
    from distributed.server.database import get_session_factory
    from distributed.server.models import CrawledDocument

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(CrawledDocument.url).where(
                CrawledDocument.domain == domain,
                CrawledDocument.status == "completed",
            )
        )
        return {row[0] for row in result.all()}


async def get_status_from_db(domain: Optional[str] = None) -> dict:
    """
    DB에서 크롤링 현황 통계를 조회한다.

    Args:
        domain: 특정 도메인 필터 (선택)

    Returns:
        상태별 통계 정보
    """
    from sqlalchemy import select, func as sqlfunc
    from distributed.server.database import get_session_factory
    from distributed.server.models import CrawledDocument

    session_factory = get_session_factory()
    async with session_factory() as session:
        query = select(
            CrawledDocument.status,
            sqlfunc.count().label("count"),
            sqlfunc.coalesce(sqlfunc.sum(CrawledDocument.content_size), 0).label("total_size"),
        ).group_by(CrawledDocument.status)

        if domain:
            query = query.where(CrawledDocument.domain == domain)

        result = await session.execute(query)
        rows = result.all()

        stats = {}
        total_count = 0
        total_size = 0
        for row in rows:
            stats[row.status] = {"count": row.count, "total_size": row.total_size}
            total_count += row.count
            total_size += row.total_size

        return {
            "domain": domain or "all",
            "total": total_count,
            "total_size_bytes": total_size,
            "by_status": stats,
        }


# ============================================================================
# 시드 URL 생성 (변경 없음)
# ============================================================================
def generate_sitemap_urls(domain: str) -> list[str]:
    """
    도메인의 주요 URL 목록을 생성한다.
    Claude가 WebFetch로 크롤링할 URL 시드 목록.

    Args:
        domain: 크롤링할 도메인 (예: dart.dev)

    Returns:
        크롤링 시작점 URL 목록
    """
    # 도메인별 주요 섹션 정의
    sitemap = {
        "dart.dev": [
            "/",
            "/overview",
            "/language",
            "/language/variables",
            "/language/operators",
            "/language/comments",
            "/language/metadata",
            "/language/libraries",
            "/language/keywords",
            "/language/built-in-types",
            "/language/records",
            "/language/collections",
            "/language/generics",
            "/language/typedefs",
            "/language/type-system",
            "/language/patterns",
            "/language/functions",
            "/language/control-flow",
            "/language/loops",
            "/language/branches",
            "/language/error-handling",
            "/language/classes",
            "/language/constructors",
            "/language/methods",
            "/language/extend",
            "/language/mixins",
            "/language/enums",
            "/language/extension-methods",
            "/language/extension-types",
            "/language/callable-objects",
            "/language/class-modifiers",
            "/language/modifier-reference",
            "/language/concurrency",
            "/language/async",
            "/language/isolates",
            "/libraries",
            "/libraries/dart-core",
            "/libraries/dart-async",
            "/libraries/dart-math",
            "/libraries/dart-convert",
            "/libraries/dart-io",
            "/libraries/dart-html",
            "/tutorials",
            "/tutorials/server",
            "/tutorials/web",
            "/effective-dart",
            "/effective-dart/style",
            "/effective-dart/documentation",
            "/effective-dart/usage",
            "/effective-dart/design",
            "/tools",
            "/tools/dart-compile",
            "/tools/dart-create",
            "/tools/dart-doc",
            "/tools/dart-fix",
            "/tools/dart-format",
            "/tools/dart-info",
            "/tools/dart-pub",
            "/tools/dart-run",
            "/tools/dart-test",
            "/tools/dartaotruntime",
            "/tools/dartdevc",
            "/tools/pub",
            "/null-safety",
            "/null-safety/understanding-null-safety",
            "/null-safety/migration-guide",
            "/codelabs",
            "/resources/books",
            "/community",
        ],
        "docs.flutter.dev": [
            "/",
            "/get-started",
            "/get-started/install",
            "/get-started/editor",
            "/get-started/test-drive",
            "/get-started/codelab",
            "/ui",
            "/ui/widgets",
            "/ui/widgets/basics",
            "/ui/layout",
            "/ui/layout/constraints",
            "/ui/interactivity",
            "/ui/assets-and-images",
            "/ui/navigation",
            "/ui/navigation/deep-linking",
            "/ui/animations",
            "/ui/animations/overview",
            "/ui/animations/tutorial",
            "/development/data-and-backend",
            "/development/data-and-backend/state-mgmt",
            "/development/data-and-backend/state-mgmt/intro",
            "/development/data-and-backend/state-mgmt/options",
            "/development/data-and-backend/networking",
            "/development/data-and-backend/json",
            "/development/accessibility-and-localization",
            "/development/accessibility-and-localization/accessibility",
            "/development/accessibility-and-localization/internationalization",
            "/development/platform-integration",
            "/development/packages-and-plugins",
            "/development/packages-and-plugins/using-packages",
            "/development/packages-and-plugins/developing-packages",
            "/testing",
            "/testing/overview",
            "/testing/debugging",
            "/testing/integration-tests",
            "/perf",
            "/perf/rendering-performance",
            "/perf/best-practices",
            "/deployment",
            "/deployment/android",
            "/deployment/ios",
            "/deployment/web",
            "/resources/faq",
            "/resources/books",
            "/cookbook",
        ],
        "api.flutter.dev": [
            "/",
            "/flutter/widgets/widgets-library.html",
            "/flutter/material/material-library.html",
            "/flutter/cupertino/cupertino-library.html",
            "/flutter/widgets/StatelessWidget-class.html",
            "/flutter/widgets/StatefulWidget-class.html",
            "/flutter/widgets/State-class.html",
            "/flutter/widgets/BuildContext-class.html",
            "/flutter/widgets/Widget-class.html",
            "/flutter/widgets/Container-class.html",
            "/flutter/widgets/Text-class.html",
            "/flutter/widgets/Row-class.html",
            "/flutter/widgets/Column-class.html",
            "/flutter/widgets/Stack-class.html",
            "/flutter/widgets/ListView-class.html",
            "/flutter/widgets/GridView-class.html",
            "/flutter/widgets/GestureDetector-class.html",
            "/flutter/material/MaterialApp-class.html",
            "/flutter/material/Scaffold-class.html",
            "/flutter/material/AppBar-class.html",
            "/flutter/material/ElevatedButton-class.html",
            "/flutter/material/TextField-class.html",
            "/flutter/material/Card-class.html",
            "/flutter/material/ListTile-class.html",
            "/flutter/material/Drawer-class.html",
            "/flutter/material/BottomNavigationBar-class.html",
            "/flutter/material/TabBar-class.html",
            "/flutter/material/Theme-class.html",
            "/flutter/animation/Animation-class.html",
            "/flutter/animation/AnimationController-class.html",
            "/flutter/painting/painting-library.html",
        ],
        "api.dart.dev": [
            "/",
            "/stable/dart-core/dart-core-library.html",
            "/stable/dart-async/dart-async-library.html",
            "/stable/dart-collection/dart-collection-library.html",
            "/stable/dart-convert/dart-convert-library.html",
            "/stable/dart-io/dart-io-library.html",
            "/stable/dart-math/dart-math-library.html",
            "/stable/dart-core/String-class.html",
            "/stable/dart-core/int-class.html",
            "/stable/dart-core/double-class.html",
            "/stable/dart-core/List-class.html",
            "/stable/dart-core/Map-class.html",
            "/stable/dart-core/Set-class.html",
            "/stable/dart-async/Future-class.html",
            "/stable/dart-async/Stream-class.html",
        ],
        "pub.dev": [
            "/",
            "/packages/provider",
            "/packages/bloc",
            "/packages/flutter_bloc",
            "/packages/riverpod",
            "/packages/get",
            "/packages/dio",
            "/packages/http",
            "/packages/shared_preferences",
            "/packages/hive",
            "/packages/sqflite",
            "/packages/path_provider",
            "/packages/url_launcher",
            "/packages/image_picker",
            "/packages/camera",
            "/packages/permission_handler",
            "/packages/firebase_core",
            "/packages/firebase_auth",
            "/packages/cloud_firestore",
            "/packages/intl",
            "/packages/json_serializable",
            "/packages/freezed",
            "/packages/equatable",
            "/packages/flutter_hooks",
            "/packages/go_router",
            "/packages/auto_route",
        ],
    }

    base_url = f"https://{domain}"
    paths = sitemap.get(domain, ["/"])

    return [base_url + path for path in paths]


# ============================================================================
# CLI 메인 함수 (asyncio 래핑)
# ============================================================================
async def async_main():
    """비동기 메인 함수. DB 연결 초기화 후 CLI 명령을 처리한다."""
    parser = argparse.ArgumentParser(
        description="Dart/Flutter 공식 문서 크롤링 도구 (PostgreSQL 저장)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 페이지 DB에 저장
  uv run python extract_data.py --url "https://dart.dev/language" --content "..."

  # stdin에서 JSON 입력
  echo '{"url": "...", "content": "..."}' | uv run python extract_data.py

  # 배치 처리 (JSON 파일)
  uv run python extract_data.py --input pages.json

  # 크롤링 시드 URL 생성 (DB 연결 불필요)
  uv run python extract_data.py --sitemap dart.dev

  # 수집 현황 확인 (DB 조회)
  uv run python extract_data.py --status
  uv run python extract_data.py --status --domain dart.dev
        """
    )

    # 입력 옵션
    parser.add_argument("--url", "-u", help="저장할 페이지 URL")
    parser.add_argument("--content", "-c", help="페이지 콘텐츠 (마크다운)")
    parser.add_argument("--title", "-t", help="페이지 제목 (선택)")
    parser.add_argument("--input", "-i", help="입력 JSON 파일 (- for stdin)")

    # 유틸리티 옵션
    parser.add_argument("--sitemap", "-s", help="도메인의 시드 URL 목록 생성")
    parser.add_argument("--status", action="store_true", help="수집 현황 표시 (DB 조회)")
    parser.add_argument("--domain", "-d", help="특정 도메인 필터")
    parser.add_argument("--links", action="store_true", help="콘텐츠에서 링크 추출")
    parser.add_argument("--json", action="store_true", help="JSON 형식으로 출력")

    args = parser.parse_args()

    # --sitemap: DB 연결 불필요한 명령
    if args.sitemap:
        urls = generate_sitemap_urls(args.sitemap)
        if args.json:
            print(json.dumps(urls, ensure_ascii=False, indent=2))
        else:
            for url in urls:
                print(url)
        return

    # --links: DB 연결 불필요한 명령
    if args.links and args.content and args.url:
        links = extract_links_from_content(args.content, args.url)
        if args.json:
            print(json.dumps(links, ensure_ascii=False, indent=2))
        else:
            for link in links:
                print(link)
        return

    # ---------------------------------------------------------------
    # 이하 명령은 DB 연결이 필요함
    # ---------------------------------------------------------------
    await init_crawl_db()
    try:
        # --status: DB에서 수집 현황 조회
        if args.status:
            if args.domain:
                # 특정 도메인의 상세 현황
                collected = await get_collected_urls_from_db(args.domain)
                total_urls = set(generate_sitemap_urls(args.domain))
                remaining = total_urls - collected
                db_status = await get_status_from_db(args.domain)

                status_info = {
                    "domain": args.domain,
                    "collected": len(collected),
                    "total_seed": len(total_urls),
                    "remaining": len(remaining),
                    "progress": f"{len(collected) / len(total_urls) * 100:.1f}%" if total_urls else "N/A",
                    "db_stats": db_status,
                }

                if args.json:
                    status_info["remaining_urls"] = sorted(list(remaining))
                    print(json.dumps(status_info, ensure_ascii=False, indent=2, default=str))
                else:
                    print(f"도메인: {status_info['domain']}")
                    print(f"수집 완료: {status_info['collected']} / {status_info['total_seed']} ({status_info['progress']})")
                    print(f"미수집: {status_info['remaining']}")
                    print(f"DB 전체: {db_status['total']}개, {db_status['total_size_bytes']:,} bytes")
                    if remaining:
                        print("\n미수집 URL:")
                        for url in sorted(remaining)[:20]:
                            print(f"  {url}")
                        if len(remaining) > 20:
                            print(f"  ... 외 {len(remaining) - 20}개")
            else:
                # 전체 현황
                db_status = await get_status_from_db()
                if args.json:
                    print(json.dumps(db_status, ensure_ascii=False, indent=2, default=str))
                else:
                    print(f"전체 문서: {db_status['total']}개, {db_status['total_size_bytes']:,} bytes")
                    for status_name, info in db_status["by_status"].items():
                        print(f"  {status_name}: {info['count']}개 ({info['total_size']:,} bytes)")
            return

        # --url + --content: 단일 페이지 DB 저장
        if args.url and args.content:
            result = await save_page_to_db(args.url, args.content, args.title)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"DB 저장 완료: {result['url']} ({result['content_size']:,} bytes)")
            return

        # JSON 입력 처리 (stdin 또는 파일)
        input_source = args.input or "-"

        if input_source == "-":
            if sys.stdin.isatty():
                parser.print_help()
                return
            data = json.load(sys.stdin)
        else:
            with open(input_source, "r", encoding="utf-8") as f:
                data = json.load(f)

        # 단일 객체 또는 배열 처리
        if isinstance(data, dict):
            pages = [data]
        else:
            pages = data

        results = await process_batch_to_db(pages)

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(f"DB 저장 완료: {len(results['saved'])} / {results['total']}")
            for item in results["saved"]:
                print(f"  ✓ {item['url']} ({item['content_size']:,} bytes)")
            if results["failed"]:
                print(f"실패: {len(results['failed'])}")
                for item in results["failed"]:
                    print(f"  ✗ {item['url']}: {item['error']}")

    finally:
        await close_crawl_db()


def main():
    """동기 진입점. asyncio.run()으로 비동기 메인을 실행한다."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
