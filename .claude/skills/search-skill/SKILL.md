---
name: search-skill
description: Dart/Flutter 공식 문서 사이트를 크롤링하여 data/raw/ 폴더에 Markdown으로 저장하는 스킬. 다음 상황에서 사용: (1) "/search <사이트>" 또는 "/search <URL>" 명령 실행 시, (2) dart.dev, docs.flutter.dev 등 공식 문서 수집 요청 시, (3) Stage 1 데이터 수집 작업 시.
user-invocable: true
---

# Dart/Flutter 문서 수집 스킬 (Stage 1)

## 목적

FAI 프로젝트의 Stage 1 데이터 수집을 담당한다. WebFetch와 WebSearch 도구를 사용하여 공식 문서 사이트를 크롤링하고, URL 경로 구조 그대로 `data/raw/` 폴더에 Markdown으로 저장한다.

---

## 사용법

```
/search <사이트명>          # 사이트 전체 크롤링
/search <URL>               # 특정 페이지만 수집
```

**예시:**
```
/search dart.dev            # dart.dev 전체 크롤링
/search docs.flutter.dev    # Flutter 문서 전체 크롤링
/search https://dart.dev/language/variables   # 특정 페이지만
```

---

## URL → 파일 경로 매핑 규칙

| URL | 저장 경로 |
|-----|----------|
| `https://dart.dev/` | `data/raw/dart.dev/index.md` |
| `https://dart.dev/overview` | `data/raw/dart.dev/overview.md` |
| `https://dart.dev/language` | `data/raw/dart.dev/language.md` |
| `https://dart.dev/language/variables` | `data/raw/dart.dev/language/variables.md` |
| `https://docs.flutter.dev/` | `data/raw/docs.flutter.dev/index.md` |
| `https://docs.flutter.dev/ui/widgets` | `data/raw/docs.flutter.dev/ui/widgets.md` |
| `https://api.flutter.dev/flutter/widgets/StatefulWidget-class.html` | `data/raw/api.flutter.dev/flutter/widgets/StatefulWidget-class.md` |

**변환 규칙:**
1. `https://` 제거
2. 도메인을 최상위 폴더로 사용
3. URL 경로를 하위 폴더/파일로 변환
4. `.html` 확장자는 `.md`로 변경
5. 경로 끝이 `/`이면 `index.md`로 저장

---

## 수집 대상 사이트

| 사이트 | 명령어 | 저장 경로 | 우선순위 |
|--------|--------|----------|----------|
| dart.dev | `/search dart.dev` | `data/raw/dart.dev/` | 1 (최우선) |
| docs.flutter.dev | `/search docs.flutter.dev` | `data/raw/docs.flutter.dev/` | 2 |
| api.flutter.dev | `/search api.flutter.dev` | `data/raw/api.flutter.dev/` | 3 |
| api.dart.dev | `/search api.dart.dev` | `data/raw/api.dart.dev/` | 4 |
| pub.dev | `/search pub.dev` | `data/raw/pub.dev/` | 5 |

---

## 실행 절차

### 1단계: 시드 URL 목록 생성

`extract_data.py --sitemap` 명령으로 도메인별 크롤링 시작점 URL 목록을 생성한다:

```bash
python3 .claude/skills/search-skill/scripts/extract_data.py --sitemap dart.dev
```

### 2단계: WebFetch로 각 페이지 수집

각 URL에 대해 WebFetch 도구로 콘텐츠를 가져온다:

```
WebFetch: https://dart.dev/language/variables
Prompt: "이 페이지의 전체 내용을 마크다운 형식으로 추출해줘. 코드 예시와 설명을 모두 포함해."
```

### 3단계: extract_data.py로 저장

WebFetch 결과를 `extract_data.py`로 저장한다:

```bash
# 단일 페이지 저장
python3 .claude/skills/search-skill/scripts/extract_data.py \
  --url "https://dart.dev/language/variables" \
  --content "WebFetch 결과..." \
  --output data/raw

# 또는 JSON으로 배치 처리
echo '[{"url": "...", "content": "..."}]' | \
  python3 .claude/skills/search-skill/scripts/extract_data.py --output data/raw
```

### 4단계: 수집 현황 확인

```bash
# 전체 현황
python3 .claude/skills/search-skill/scripts/extract_data.py --status --output data/raw

# 특정 도메인 현황
python3 .claude/skills/search-skill/scripts/extract_data.py --status --domain dart.dev --output data/raw
```

---

## extract_data.py 스크립트

### 주요 기능

| 옵션 | 설명 |
|------|------|
| `--url, -u` | 저장할 페이지 URL |
| `--content, -c` | 페이지 콘텐츠 (마크다운) |
| `--title, -t` | 페이지 제목 (선택) |
| `--input, -i` | 입력 JSON 파일 (- for stdin) |
| `--output, -o` | 출력 기본 디렉토리 (기본: data/raw) |
| `--sitemap, -s` | 도메인의 시드 URL 목록 생성 |
| `--status` | 수집 현황 표시 |
| `--domain, -d` | 특정 도메인 필터 |
| `--links` | 콘텐츠에서 링크 추출 |
| `--json` | JSON 형식으로 출력 |

### 사용 예시

```bash
# 시드 URL 목록 생성
python3 extract_data.py --sitemap dart.dev

# 단일 페이지 저장
python3 extract_data.py --url "https://dart.dev/language" --content "..." --output data/raw

# stdin에서 JSON 입력
echo '{"url": "...", "content": "..."}' | python3 extract_data.py --output data/raw

# 수집 현황 (미수집 URL 포함)
python3 extract_data.py --status --domain dart.dev --json --output data/raw
```

---

## 저장 파일 형식

```markdown
# [Page Title]

[원본 문서 내용을 Markdown으로 변환]

## Overview
...

## Details
...

## Code Examples
```dart
// Example code from the page
void main() {
  print('Hello, Dart!');
}
```

---

## Source

- **URL**: https://dart.dev/language/variables
- **Fetched**: 2024-01-27
```

---

## 크롤링 워크플로우 (Claude 실행)

1. **시드 URL 확인**: `--sitemap` 또는 `--status --domain`으로 수집할 URL 목록 확인
2. **반복 크롤링**: 각 URL에 대해:
   - WebFetch로 콘텐츠 가져오기
   - extract_data.py로 저장
3. **링크 발견**: WebFetch 결과에서 새 링크 발견 시 추가 수집
4. **진행 상황 확인**: `--status`로 수집 현황 확인

---

## 크롤링 주의사항

1. **robots.txt 준수**: 각 사이트의 크롤링 정책 확인
2. **요청 간격**: 서버 부하 방지를 위해 적절한 딜레이
3. **중복 방지**: 이미 수집된 파일은 스킵 (덮어쓰기 옵션 제공)
4. **오류 처리**: 실패한 URL 로깅 및 재시도 로직

---

## 지원 도메인 URL 목록

### dart.dev (60+ URLs)

주요 섹션: `/language/*`, `/libraries/*`, `/tutorials/*`, `/effective-dart/*`, `/tools/*`

### docs.flutter.dev (45+ URLs)

주요 섹션: `/get-started/*`, `/ui/*`, `/development/*`, `/testing/*`, `/deployment/*`

### api.flutter.dev (30+ URLs)

주요 클래스: `StatelessWidget`, `StatefulWidget`, `Container`, `Text`, `Row`, `Column`, Material/Cupertino 위젯

### api.dart.dev (15+ URLs)

주요 라이브러리: `dart-core`, `dart-async`, `dart-collection`, `dart-convert`, `dart-io`

### pub.dev (25+ URLs)

인기 패키지: `provider`, `bloc`, `riverpod`, `dio`, `hive`, `firebase_*` 등

---

## 관련 스킬

- **fai-skill**: Stage 2 (전처리) 및 전체 프로젝트 관리
