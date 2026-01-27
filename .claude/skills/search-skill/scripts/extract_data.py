#!/usr/bin/env python3
"""
extract_data.py - Dart/Flutter 공식 문서 크롤링 도구

FAI 프로젝트 Stage 1 데이터 수집용 스크립트.
WebFetch 결과를 받아서 URL 경로 구조 그대로 data/raw/ 폴더에 Markdown으로 저장한다.

Usage:
    # 단일 페이지 저장
    python3 extract_data.py --url "https://dart.dev/language/variables" --content "..." --output data/raw

    # 파이프라인으로 사용 (JSON 입력)
    echo '{"url": "https://dart.dev/overview", "content": "..."}' | python3 extract_data.py --output data/raw

    # 여러 페이지 배치 저장 (JSON 배열)
    python3 extract_data.py --input pages.json --output data/raw
"""

import argparse
import json
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional


def url_to_filepath(url: str, base_dir: str = "data/raw") -> str:
    """
    URL을 파일 경로로 변환한다.

    변환 규칙:
    1. https:// 제거
    2. 도메인을 최상위 폴더로 사용
    3. URL 경로를 하위 폴더/파일로 변환
    4. .html 확장자는 .md로 변경
    5. 경로 끝이 /이면 index.md로 저장

    예시:
    - https://dart.dev/ → data/raw/dart.dev/index.md
    - https://dart.dev/overview → data/raw/dart.dev/overview.md
    - https://dart.dev/language/variables → data/raw/dart.dev/language/variables.md
    - https://api.flutter.dev/flutter/widgets/Text-class.html → data/raw/api.flutter.dev/flutter/widgets/Text-class.md
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    # 경로 정규화
    if not path or path == "/":
        path = "/index"
    elif path.endswith("/"):
        path = path.rstrip("/") + "/index"

    # .html → .md 변환
    if path.endswith(".html"):
        path = path[:-5]  # .html 제거

    # 파일 경로 생성
    filepath = os.path.join(base_dir, domain, path.lstrip("/") + ".md")
    return filepath


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


def save_page(url: str, content: str, base_dir: str, title: Optional[str] = None) -> str:
    """
    페이지 콘텐츠를 파일로 저장한다.

    Args:
        url: 페이지 URL
        content: 페이지 콘텐츠 (마크다운)
        base_dir: 저장 기본 디렉토리
        title: 페이지 제목 (선택)

    Returns:
        저장된 파일 경로
    """
    filepath = url_to_filepath(url, base_dir)

    # 디렉토리 생성
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 콘텐츠 포맷팅
    formatted_content = format_markdown_content(content, url, title)

    # 파일 저장 (UTF-8)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formatted_content)

    return filepath


def process_batch(pages: list[dict], base_dir: str) -> dict:
    """
    여러 페이지를 배치로 처리한다.

    Args:
        pages: [{"url": "...", "content": "...", "title": "..."}] 형식의 페이지 목록
        base_dir: 저장 기본 디렉토리

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
            filepath = save_page(url, content, base_dir, title)
            results["saved"].append({"url": url, "filepath": filepath})
        except Exception as e:
            results["failed"].append({"url": url, "error": str(e)})

    return results


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


def list_collected_files(base_dir: str, domain: Optional[str] = None) -> list[str]:
    """
    이미 수집된 파일 목록을 반환한다.

    Args:
        base_dir: 기본 디렉토리 (data/raw)
        domain: 특정 도메인만 조회 (선택)

    Returns:
        수집된 .md 파일 경로 목록
    """
    search_path = Path(base_dir)
    if domain:
        search_path = search_path / domain

    if not search_path.exists():
        return []

    return sorted([str(p) for p in search_path.rglob("*.md")])


def get_collected_urls(base_dir: str, domain: str) -> set[str]:
    """
    이미 수집된 URL 목록을 반환한다.
    파일 경로를 역으로 URL로 변환.

    Args:
        base_dir: 기본 디렉토리
        domain: 도메인

    Returns:
        수집된 URL set
    """
    files = list_collected_files(base_dir, domain)
    urls = set()

    for filepath in files:
        # 파일 경로에서 URL 복원
        rel_path = os.path.relpath(filepath, base_dir)
        parts = rel_path.split(os.sep)

        if len(parts) < 1:
            continue

        file_domain = parts[0]
        path_parts = parts[1:]

        # index.md → /
        if path_parts and path_parts[-1] == "index.md":
            path_parts = path_parts[:-1]
            url_path = "/" + "/".join(path_parts) if path_parts else "/"
        else:
            # .md 제거
            if path_parts:
                path_parts[-1] = path_parts[-1].replace(".md", "")
            url_path = "/" + "/".join(path_parts)

        url = f"https://{file_domain}{url_path}"
        urls.add(url)

    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Dart/Flutter 공식 문서 크롤링 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 페이지 저장
  python3 extract_data.py --url "https://dart.dev/language" --content "..." --output data/raw

  # stdin에서 JSON 입력
  echo '{"url": "...", "content": "..."}' | python3 extract_data.py --output data/raw

  # 배치 처리 (JSON 파일)
  python3 extract_data.py --input pages.json --output data/raw

  # 크롤링 시드 URL 생성
  python3 extract_data.py --sitemap dart.dev

  # 수집 현황 확인
  python3 extract_data.py --status --output data/raw
  python3 extract_data.py --status --domain dart.dev --output data/raw
        """
    )

    # 입력 옵션
    parser.add_argument("--url", "-u", help="저장할 페이지 URL")
    parser.add_argument("--content", "-c", help="페이지 콘텐츠 (마크다운)")
    parser.add_argument("--title", "-t", help="페이지 제목 (선택)")
    parser.add_argument("--input", "-i", help="입력 JSON 파일 (- for stdin)")

    # 출력 옵션
    parser.add_argument("--output", "-o", default="data/raw", help="출력 기본 디렉토리")

    # 유틸리티 옵션
    parser.add_argument("--sitemap", "-s", help="도메인의 시드 URL 목록 생성")
    parser.add_argument("--status", action="store_true", help="수집 현황 표시")
    parser.add_argument("--domain", "-d", help="특정 도메인 필터")
    parser.add_argument("--links", action="store_true", help="콘텐츠에서 링크 추출")
    parser.add_argument("--json", action="store_true", help="JSON 형식으로 출력")

    args = parser.parse_args()

    # 시드 URL 생성
    if args.sitemap:
        urls = generate_sitemap_urls(args.sitemap)
        if args.json:
            print(json.dumps(urls, ensure_ascii=False, indent=2))
        else:
            for url in urls:
                print(url)
        return

    # 수집 현황 표시
    if args.status:
        if args.domain:
            collected = get_collected_urls(args.output, args.domain)
            total_urls = set(generate_sitemap_urls(args.domain))
            remaining = total_urls - collected

            status = {
                "domain": args.domain,
                "collected": len(collected),
                "total": len(total_urls),
                "remaining": len(remaining),
                "progress": f"{len(collected) / len(total_urls) * 100:.1f}%" if total_urls else "N/A"
            }

            if args.json:
                status["remaining_urls"] = sorted(list(remaining))
                print(json.dumps(status, ensure_ascii=False, indent=2))
            else:
                print(f"도메인: {status['domain']}")
                print(f"수집 완료: {status['collected']} / {status['total']} ({status['progress']})")
                print(f"미수집: {status['remaining']}")
                if remaining:
                    print("\n미수집 URL:")
                    for url in sorted(remaining)[:20]:  # 최대 20개만 표시
                        print(f"  {url}")
                    if len(remaining) > 20:
                        print(f"  ... 외 {len(remaining) - 20}개")
        else:
            files = list_collected_files(args.output)
            if args.json:
                print(json.dumps({"files": files, "total": len(files)}, ensure_ascii=False, indent=2))
            else:
                print(f"총 수집 파일: {len(files)}")
                for f in files[:30]:  # 최대 30개만 표시
                    print(f"  {f}")
                if len(files) > 30:
                    print(f"  ... 외 {len(files) - 30}개")
        return

    # 링크 추출
    if args.links and args.content and args.url:
        links = extract_links_from_content(args.content, args.url)
        if args.json:
            print(json.dumps(links, ensure_ascii=False, indent=2))
        else:
            for link in links:
                print(link)
        return

    # 단일 페이지 저장
    if args.url and args.content:
        filepath = save_page(args.url, args.content, args.output, args.title)
        result = {"url": args.url, "filepath": filepath, "status": "saved"}
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"저장 완료: {filepath}")
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

    results = process_batch(pages, args.output)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"저장 완료: {len(results['saved'])} / {results['total']}")
        for item in results["saved"]:
            print(f"  ✓ {item['filepath']}")
        if results["failed"]:
            print(f"실패: {len(results['failed'])}")
            for item in results["failed"]:
                print(f"  ✗ {item['url']}: {item['error']}")


if __name__ == "__main__":
    main()
