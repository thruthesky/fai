---
name: search-skill
description: Dart/Flutter 공식 문서 사이트를 크롤링하여 data/raw/ 폴더에 Markdown으로 저장하는 스킬. 다음 상황에서 사용: (1) "/search <사이트>" 또는 "/search <URL>" 명령 실행 시, (2) dart.dev, docs.flutter.dev 등 공식 문서 수집 요청 시, (3) Stage 1 데이터 수집 작업 시.
context: fork
argument-hint: JSON
disable-model-invocation: false
allowed-tools: Read, Grep, Web Search, Web Fetch, Bash
user-invocable: true
license: Complete terms in LICENSE.txt
---

# Dart/Flutter 문서 수집 스킬 (Stage 1)

## 목적

FAI 프로젝트의 Stage 1 데이터 수집을 담당한다. 공식 문서 사이트를 크롤링하여 URL 경로 구조 그대로 `data/raw/` 폴더에 Markdown으로 저장한다.

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

### 1단계: 사이트맵 또는 네비게이션 분석

WebFetch로 메인 페이지에서 모든 내부 링크 수집:

```
1. 메인 페이지 fetch
2. 사이드바/네비게이션에서 모든 링크 추출
3. 중복 제거 및 외부 링크 필터링
4. 수집할 URL 목록 생성
```

### 2단계: 각 페이지 수집

각 URL에 대해:

```
1. WebFetch로 페이지 내용 가져오기
2. HTML → Markdown 변환
3. URL 경로에 맞는 폴더 구조 생성
4. .md 파일로 저장
```

### 3단계: 메타데이터 추가

각 파일 끝에 출처 정보 추가:

```markdown
---

## Source
- URL: https://dart.dev/language/variables
- Fetched: 2024-01-27
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
- URL: https://dart.dev/language/variables
- Fetched: 2024-01-27
```

---

## 크롤링 주의사항

1. **robots.txt 준수**: 각 사이트의 크롤링 정책 확인
2. **요청 간격**: 서버 부하 방지를 위해 적절한 딜레이
3. **중복 방지**: 이미 수집된 파일은 스킵 (덮어쓰기 옵션 제공)
4. **오류 처리**: 실패한 URL 로깅 및 재시도 로직

---

## dart.dev 크롤링 예시 (우선 수집 대상)

### 주요 섹션

| 섹션 | URL 패턴 | 예상 파일 수 |
|------|----------|-------------|
| Language | `/language/*` | ~30 |
| Libraries | `/libraries/*` | ~20 |
| Tutorials | `/tutorials/*` | ~15 |
| Effective Dart | `/effective-dart/*` | ~10 |
| Tools | `/tools/*` | ~25 |

### 크롤링 실행

```
/search dart.dev
```

**예상 결과:**
```
data/raw/dart.dev/
├── index.md
├── overview.md
├── language.md
├── language/
│   ├── variables.md
│   ├── operators.md
│   ├── functions.md
│   ├── classes.md
│   └── ...
├── libraries.md
├── libraries/
│   ├── dart-core.md
│   ├── dart-async.md
│   └── ...
└── ...
```

---

## 관련 스킬

- **fai-skill**: Stage 2 (전처리) 및 전체 프로젝트 관리
