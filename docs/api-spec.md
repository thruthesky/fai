# Center API 사용 가이드 (API Specification)

이 문서는 Center API를 **외부에서 사용하는 방법**에 대한 상세 가이드입니다.

> **참고:** API 개발자를 위한 문서는 [api.md](api.md)를 참조하세요.

---

## 🚨🚨🚨 최우선 규칙: function_list API 사용 필수 🚨🚨🚨

> **⛔⛔⛔ 절대 준수 사항 ⛔⛔⛔**
>
> **API 함수 목록은 자주 변경됩니다!**
>
> 본 문서에 기재된 API 함수 목록은 **참고용**일 뿐입니다.
> 실제 사용 시에는 **반드시 `function_list` API를 호출**하여 최신 함수 목록을 확인해야 합니다.

| 구분 | 설명 |
|------|------|
| **본 문서의 함수 목록** | ⚠️ 참고용 (구버전일 수 있음) |
| **function_list API** | ✅ 항상 최신 상태 (필수 사용) |

### 왜 function_list를 사용해야 하는가?

1. **API 함수는 자주 추가/변경/삭제됩니다**
2. **본 문서의 함수 목록은 작성 시점의 스냅샷**입니다
3. **function_list API는 항상 현재 서버의 최신 함수 목록을 반환**합니다
4. **DocBlock 기반으로 파라미터, 반환 타입까지 정확히 제공**합니다

### 권장 워크플로우

```
1단계: function_list API 호출 (필수!)
    └─> 최신 API 함수 목록 획득

2단계: 필요한 함수 검색
    └─> name, description 필드로 검색

3단계: API 호출
    └─> params 필드로 파라미터 확인 후 호출
```

---

## 문서 용도

| 문서 | 대상 | 내용 |
|------|------|------|
| **api-spec.md** (본 문서) | 외부 도구, AI, 자동화 시스템 | API 사용 방법, 토큰 생성, 호출 예시 |
| [api.md](api.md) | API 개발자 | 새 API 함수 추가 방법, 아키텍처 |

---

## API 기본 정보

| 항목 | 값 |
|------|-----|
| **엔드포인트** | `/api.php` |
| **HTTP 메서드** | POST (권장), GET (일부 조회 API) |
| **Content-Type** | `application/json` |
| **인증 방식** | API 토큰 (`apikey-{user_id}-{md5_hash}`) |
| **응답 형식** | JSON |

---

## 1. function_list API (시작점)

### 개요

`function_list`는 **모든 API 사용의 시작점**입니다. AllowedFunctions 클래스에 정의된 모든 API 함수 정보를 조회할 수 있습니다.

| 특징 | 설명 |
|------|------|
| **인증** | 불필요 (누구나 호출 가능) |
| **용도** | 사용 가능한 API 목록 확인 |
| **응답** | 함수명, 설명, 파라미터, 반환 타입 |

### 호출 방법

**curl 예시:**
```bash
curl -X POST "https://sonub.com/api.php" \
  -H "Content-Type: application/json" \
  -d '{"func": "function_list"}'
```

**GET 요청:**
```bash
curl "https://sonub.com/api.php?func=function_list"
```

### 응답 형식

```json
{
  "functions": [
    {
      "name": "create_post",
      "description": "게시글 생성\n@param array $input 입력 데이터\n- token: Firebase ID Token\n- category_id: 카테고리 ID (우선)\n- category_slug: 카테고리 슬러그\n- title: 제목\n- content: 내용\n@return Post Post Entity",
      "params": [
        {
          "name": "input",
          "type": "array",
          "required": true,
          "default": null
        }
      ],
      "return_type": "Center\\Entity\\Post"
    },
    {
      "name": "my",
      "description": "내 정보 조회\n@param array $input 입력 데이터\n- token: Firebase ID Token\n@return User|null 사용자 Entity 또는 null",
      "params": [...],
      "return_type": "?Center\\Entity\\User"
    }
  ],
  "count": 112
}
```

### 응답 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `functions` | array | API 함수 목록 |
| `functions[].name` | string | 함수명 (API 호출 시 `func` 파라미터 값) |
| `functions[].description` | string | DocBlock 전체 (줄바꿈 `\n`으로 구분) |
| `functions[].params` | array | 파라미터 정보 |
| `functions[].return_type` | string | 반환 타입 (예: `array`, `Center\Entity\User`) |
| `count` | int | 전체 함수 개수 |

---

## 2. 사용자 토큰 생성 (인증)

API 토큰은 `apikey-{user_id}-{md5_hash}` 형식입니다. `md5_hash`는 다음 필드들을 결합한 문자열의 MD5 해시입니다:

## 3. 게시글 생성 API (create_post)

### API 정보

| 항목 | 값 |
|------|-----|
| **함수명** | `create_post` |
| **인증** | 필수 (token) |
| **HTTP 메서드** | POST |

### 필수/선택 파라미터

| 파라미터 | 필수 | 설명 |
|----------|------|------|
| `token` | ✅ | 사용자 API 토큰 |
| `category_id` | ⭕ | 카테고리 ID (category_slug보다 우선) |
| `category_slug` | ⭕ | 카테고리 슬러그 |
| `title` | ✅ | 게시글 제목 |
| `content` | ✅ | 게시글 내용 |
| `urls` | ❌ | 첨부파일 URL 배열 (선택) |

> **참고:** `category_id` 또는 `category_slug` 중 하나는 필수입니다.

### curl 호출 예시

```bash
TOKEN="apikey-1-a1b2c3d4e5f6..."

curl -s -X POST 'https://sonub.com/api.php' \
  -H 'Content-Type: application/json' \
  -d '{
    "func": "create_post",
    "token": "'"$TOKEN"'",
    "category_slug": "free-board",
    "title": "테스트 게시글 제목",
    "content": "테스트 게시글 내용입니다."
  }' | jq .
```

### 성공 응답

**HTTP 상태 코드:** `200 OK`

```json
{
  "id": 12345,
  "branch_id": 14,
  "category_id": 6780,
  "user_id": 1,
  "title": "테스트 게시글 제목",
  "content": "테스트 게시글 내용입니다.",
  "urls": [],
  "view_count": 0,
  "comment_count": 0,
  "like_count": 0,
  "dislike_count": 0,
  "created_at": "2025-01-01T12:00:00+09:00",
  "updated_at": "2025-01-01T12:00:00+09:00",
  "deleted_at": null
}
```

**응답 필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | int | 게시글 고유 ID |
| `branch_id` | int | 가맹사(Branch) ID |
| `category_id` | int | 카테고리 ID |
| `user_id` | int | 작성자 사용자 ID |
| `title` | string | 게시글 제목 |
| `content` | string | 게시글 본문 내용 |
| `urls` | array | 첨부파일 URL 배열 (이미지, 동영상, 문서 등) |
| `view_count` | int | 조회수 |
| `comment_count` | int | 댓글 수 |
| `like_count` | int | 좋아요 수 |
| `dislike_count` | int | 싫어요 수 |
| `created_at` | string | 생성 시각 (ISO 8601 형식, 타임존 포함) |
| `updated_at` | string | 수정 시각 (ISO 8601 형식, 타임존 포함) |
| `deleted_at` | string\|null | 삭제 시각 (Soft Delete, null이면 삭제되지 않음) |
| `display_name` | string | 작성자 닉네임 (JOIN 시) |
| `photo_url` | string | 작성자 프로필 사진 URL (JOIN 시) |
| `uid` | string | 작성자 Firebase UID (JOIN 시, Hovercard용) |

### 에러 응답

**HTTP 상태 코드:** `400`, `401`, `403`, `404` 등 (에러 유형에 따라 다름)

**에러 응답 형식:**

```json
{
  "error": "에러코드/상세코드",
  "message": "사용자에게 표시할 에러 메시지"
}
```

**주요 에러 케이스:**

| HTTP 상태 | 에러 코드 | 메시지 | 원인 |
|-----------|-----------|--------|------|
| 400 | `assert-token/token-required` | API TOKEN이 필요합니다. | token 파라미터 누락 |
| 400 | `assert-token/invalid-api-token-hash` | API TOKEN 해시가 일치하지 않습니다. | 토큰 해시 불일치 (잘못된 토큰) |
| 400 | `post/title-required` | 제목을 입력하세요. | title 파라미터 누락 |
| 400 | `post/content-required` | 내용을 입력하세요. | content 파라미터 누락 |
| 400 | `category/id-or-slug-required` | 카테고리 ID 또는 슬러그가 필요합니다. | category_id, category_slug 모두 누락 |
| 404 | `category/not-found` | 카테고리를 찾을 수 없습니다. | 존재하지 않는 카테고리 |
| 404 | `user/not-found` | 사용자를 찾을 수 없습니다. | 토큰의 사용자가 존재하지 않음 |
| 403 | `permission-denied` | 권한이 없습니다. | 해당 카테고리에 글 작성 권한 없음 |

**에러 응답 예시:**

```json
{
  "error": "assert-token/invalid-api-token-hash",
  "message": "API TOKEN 해시가 일치하지 않습니다."
}
```

```json
{
  "error": "post/title-required",
  "message": "제목을 입력하세요."
}
```

### 첨부파일(이미지) URL 직접 입력

게시글에 이미지나 파일을 첨부하려면 `urls` 파라미터에 **이미지 URL 배열**을 전달합니다.

> **💡 핵심:** 파일 서버에 직접 업로드할 필요 없이, **외부 이미지 URL을 그대로 사용**할 수 있습니다.

**지원되는 URL 형식:**

| 형식 | 예시 | 설명 |
|------|------|------|
| 외부 이미지 URL | `https://example.com/image.jpg` | 외부 서버의 이미지 직접 사용 |
| Center 파일 서버 경로 | `/uploads/uid/file.jpg` | file_upload API로 업로드한 파일 경로 |

**curl 호출 예시 (외부 이미지 URL 사용):**

```bash
curl -s -X POST 'https://sonub.com/api.php' \
  -H 'Content-Type: application/json' \
  -d '{
    "func": "create_post",
    "token": "apikey-15966-c3bf931180822294d07b80759d914eed",
    "category_slug": "free-board",
    "title": "이미지가 포함된 게시글",
    "content": "본문 내용입니다.",
    "urls": [
      "https://example.com/photo1.jpg",
      "https://example.com/photo2.png"
    ]
  }' | jq .
```

**지원 파일 형식:**

| 분류 | 확장자 |
|------|--------|
| 이미지 | jpg, jpeg, png, gif, webp, avif |
| 동영상 | mp4 |
| 문서 | pdf, doc, docx, xls, xlsx, ppt, pptx |
| 압축 | zip, rar, 7z |

**URL 처리 방식:**

| URL 형식 | 예시 | 처리 방식 |
|----------|------|-----------|
| 전체 URL (https://) | `https://example.com/image.jpg` | 그대로 표시 |
| 전체 URL (http://) | `http://example.com/image.jpg` | 그대로 표시 |
| 상대 경로 | `/uploads/uid/file.jpg` | Center 파일 서버 URL과 결합하여 표시 |

**주의사항:**

- 전체 URL(`https://`, `http://`)은 **그대로 화면에 표시**됩니다
- 상대 경로(`/uploads/...`)는 Center 파일 서버 URL이 자동 추가됩니다
- 외부 URL은 **공개적으로 접근 가능**해야 합니다
- 이미지는 게시글 상단에 썸네일로 표시됩니다
- 여러 파일을 첨부하려면 배열에 URL을 추가하세요

### file_upload API로 파일 업로드 후 사용

외부 URL 대신 Center 파일 서버에 직접 업로드하려면 `file_upload` API를 사용합니다.

**1단계: 파일 업로드**

```bash
curl -X POST 'https://sonub.com/api.php' \
  -F "func=file_upload" \
  -F "token=apikey-15966-c3bf931180822294d07b80759d914eed" \
  -F "file=@./photo.jpg"
```

**업로드 응답:**

```json
{
  "url": "https://sonub.com/uploads/abc123/20250104_photo.jpg",
  "thumbnail_url": "https://sonub.com/thumbnail.php?src=abc123/20250104_photo.jpg&w=100&h=100",
  "path": "abc123/20250104_photo.jpg",
  "is_image": true,
  "is_video": false
}
```

**2단계: 업로드된 파일 URL로 게시글 생성**

```bash
curl -s -X POST 'https://sonub.com/api.php' \
  -H 'Content-Type: application/json' \
  -d '{
    "func": "create_post",
    "token": "apikey-15966-c3bf931180822294d07b80759d914eed",
    "category_slug": "free-board",
    "title": "업로드된 이미지가 포함된 게시글",
    "content": "본문 내용입니다.",
    "urls": [
      "/uploads/abc123/20250104_photo.jpg"
    ]
  }' | jq .
```

> **📖 파일 업로드 상세:** [file-upload.md](file-upload.md)

### 스크립트를 이용한 게시글 생성

```bash
# 단일 게시글 생성
./.claude/skills/center-skill/scripts/create_post.sh \
    --slug free-board \
    --title "제목" \
    --content "내용"

# 배치 모드 (여러 글 일괄 생성)
./.claude/skills/center-skill/scripts/create_post.sh \
    --batch \
    --auto-token \
    --user-id 1 \
    --slug free-board \
    --data-file ./posts.json
```

> **📖 배치 생성 상세:** [batch-post-creation.md](batch-post-creation.md)

---

## 4. 내 게시글 조회 API (list_my_posts)

> **🤖 AI/자동화 사용 지침**
>
> 다음과 같은 사용자 요청이 있을 때 `list_my_posts` API를 사용하세요:
> - "내 글 목록 보여줘", "내가 쓴 글 조회해줘"
> - "내 글 추출하기", "내 게시글 가져와줘"
> - "내가 작성한 게시글 리스트", "나의 포스트 목록"
> - "특정 카테고리에서 내가 쓴 글만 보여줘"
>
> **필수 조건:** 사용자 API 토큰(token)이 필요합니다.

### API 정보

| 항목 | 값 |
|------|-----|
| **함수명** | `list_my_posts` |
| **인증** | 필수 (token) |
| **HTTP 메서드** | POST |

### 파라미터

| 파라미터 | 필수 | 타입 | 설명 |
|----------|------|------|------|
| `token` | ✅ | string | 사용자 API 토큰 |
| `category_id` | ❌ | int | 카테고리 ID (필터링용) |
| `category_slug` | ❌ | string | 카테고리 슬러그 (category_id 대신 사용 가능) |
| `page` | ❌ | int | 페이지 번호 (기본 1) |
| `limit` | ❌ | int | 페이지당 개수 (기본 20, 최대 100) |

### curl 호출 예시

**기본 조회 (전체 게시글):**

```bash
TOKEN="apikey-15966-c3bf931180822294d07b80759d914eed"

curl -s -X POST 'https://sonub.com/api.php' \
  -H 'Content-Type: application/json' \
  -d '{
    "func": "list_my_posts",
    "token": "'"$TOKEN"'"
  }' | jq .
```

**카테고리 필터링:**

```bash
# category_id로 필터링
curl -s -X POST 'https://sonub.com/api.php' \
  -H 'Content-Type: application/json' \
  -d '{
    "func": "list_my_posts",
    "token": "'"$TOKEN"'",
    "category_id": 123
  }' | jq .

# category_slug로 필터링
curl -s -X POST 'https://sonub.com/api.php' \
  -H 'Content-Type: application/json' \
  -d '{
    "func": "list_my_posts",
    "token": "'"$TOKEN"'",
    "category_slug": "free-board"
  }' | jq .
```

**페이지네이션:**

```bash
curl -s -X POST 'https://sonub.com/api.php' \
  -H 'Content-Type: application/json' \
  -d '{
    "func": "list_my_posts",
    "token": "'"$TOKEN"'",
    "page": 2,
    "limit": 10
  }' | jq .
```

### 성공 응답

**HTTP 상태 코드:** `200 OK`

```json
{
  "data": [
    {
      "id": 12345,
      "branch_id": 14,
      "category_id": 6780,
      "user_id": 15966,
      "title": "게시글 제목",
      "content": "게시글 내용입니다.",
      "urls": [],
      "view_count": 42,
      "comment_count": 3,
      "like_count": 5,
      "dislike_count": 0,
      "created_at": "2025-01-01T12:00:00+09:00",
      "updated_at": "2025-01-01T12:00:00+09:00",
      "deleted_at": null,
      "category_name": "자유게시판",
      "category_slug": "free-board"
    }
  ],
  "total": 25,
  "page": 1,
  "limit": 20
}
```

**응답 필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `data` | array | 게시글 배열 |
| `data[].category_name` | string | 카테고리 이름 (추가 정보) |
| `data[].category_slug` | string | 카테고리 슬러그 (추가 정보) |
| `total` | int | 전체 게시글 수 |
| `page` | int | 현재 페이지 번호 |
| `limit` | int | 페이지당 개수 |

### 에러 응답

| HTTP 상태 | 에러 코드 | 메시지 | 원인 |
|-----------|-----------|--------|------|
| 400 | `assert-token/token-required` | API TOKEN이 필요합니다. | token 파라미터 누락 |
| 400 | `assert-token/invalid-api-token-hash` | API TOKEN 해시가 일치하지 않습니다. | 토큰 해시 불일치 |

---

## 5. 외부 도구/AI/자동화 활용

### 5.1 LLM (AI) 활용 패턴

AI가 Center API를 활용하는 권장 워크플로우입니다.

```
1단계: function_list API 호출
    └─> 전체 API 함수 목록 획득

2단계: 사용자 요청 분석
    └─> "내 정보를 조회하고 싶어" → description에서 관련 함수 탐색

3단계: 적절한 API 함수 선택
    └─> "내 정보 조회" → my 함수 선택

4단계: 파라미터 확인
    └─> params 배열에서 필수 파라미터 확인 (token 필요)

5. 사용자 토큰 생성 또는 확보
    └─> MD5 해시 공식으로 토큰 생성

6단계: API 호출 생성 및 실행
    └─> curl 또는 HTTP 클라이언트로 호출
```

**실제 예시:**

```python
import requests
import hashlib

# 1. API 목록 조회
response = requests.post(
    "https://sonub.com/api.php",
    json={"func": "function_list"}
)
functions = response.json()["functions"]

# 2. 필요한 함수 찾기
for func in functions:
    if "내 정보" in func["description"]:
        print(f"Found: {func['name']}")
        # my 함수 발견

# 3. 토큰 생성 (사용자 정보 필요)
user_id = 1
created_at_ts = 1735123456
email = "apple@test.com"
branch_id = ""

combined = f"{user_id}{created_at_ts}{email}{branch_id}"
hash_value = hashlib.md5(combined.encode()).hexdigest()
token = f"apikey-{user_id}-{hash_value}"

# 4. API 호출
response = requests.post(
    "https://sonub.com/api.php",
    json={"func": "my", "token": token}
)
user_info = response.json()
```

### 5.2 CI/CD 파이프라인 활용

GitHub Actions, Jenkins 등에서 API를 활용하는 예시입니다.

**GitHub Actions 예시:**

```yaml
name: Create Release Post

on:
  release:
    types: [published]

jobs:
  create-post:
    runs-on: ubuntu-latest
    steps:
      - name: Create Announcement Post
        run: |
          curl -X POST "${{ secrets.API_ENDPOINT }}/api.php" \
            -H "Content-Type: application/json" \
            -d '{
              "func": "create_post",
              "token": "${{ secrets.API_TOKEN }}",
              "category_slug": "announcements",
              "title": "v${{ github.event.release.tag_name }} 릴리즈",
              "content": "${{ github.event.release.body }}"
            }'
```

### 5.3 Postman/Insomnia 활용

`function_list` API로 전체 API 스펙을 조회한 후, Postman Collection을 자동 생성할 수 있습니다.

**Postman Collection 변환 스크립트:**

```javascript
// function_list 응답을 Postman Collection으로 변환
const functionList = await fetch('/api.php?func=function_list').then(r => r.json());

const collection = {
  info: {
    name: "Center API",
    schema: "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  item: functionList.functions.map(func => ({
    name: func.name,
    request: {
      method: "POST",
      header: [{ key: "Content-Type", value: "application/json" }],
      body: {
        mode: "raw",
        raw: JSON.stringify({ func: func.name, token: "{{token}}" })
      },
      url: "{{baseUrl}}/api.php"
    }
  }))
};

// collection.json으로 저장 후 Postman에서 Import
```

### 5.4 배치 스크립트 활용

**여러 게시글 일괄 생성:**

```bash
#!/bin/bash

# 토큰 생성
USER_ID=1
TOKEN=$(php -r "
require 'vendor/autoload.php';
echo (new Center\Service\UserService())->generateUserTokenById($USER_ID);
")

# JSON 파일에서 게시글 데이터 읽어서 순차 생성
cat posts.json | jq -c '.[]' | while read post; do
    title=$(echo $post | jq -r '.title')
    content=$(echo $post | jq -r '.content')

    curl -s -X POST 'https://sonub.com/api.php' \
      -H 'Content-Type: application/json' \
      -d "{
        \"func\": \"create_post\",
        \"token\": \"$TOKEN\",
        \"category_slug\": \"news\",
        \"title\": \"$title\",
        \"content\": \"$content\"
      }"

    echo "Created: $title"
    sleep 1  # Rate limiting
done
```

**posts.json 형식:**

```json
[
  {"title": "첫 번째 뉴스", "content": "뉴스 내용 1"},
  {"title": "두 번째 뉴스", "content": "뉴스 내용 2"},
  {"title": "세 번째 뉴스", "content": "뉴스 내용 3"}
]
```

### 5.5 자동화 도구 통합 시나리오

| 시나리오 | 활용 API | 설명 |
|---------|---------|------|
| **콘텐츠 자동 게시** | `create_post` | RSS 피드 파싱 후 자동 게시 |
| **사용자 통계 수집** | `root_count_all` | 일일 사용자 통계 리포트 생성 |
| **배너 관리 자동화** | `create_banner`, `start_banner` | 예약된 캠페인 자동 실행 |
| **검색 트렌드 분석** | `get_popular_keywords` | 인기 검색어 대시보드 구축 |
| **알림 시스템 연동** | `list_notification` | 외부 알림 서비스(Slack, Discord) 연동 |

---

## 6. 주요 API 함수 요약 (⚠️ 참고용)

> **🚨 주의: 아래 함수 목록은 참고용입니다!**
>
> API 함수는 자주 추가/변경/삭제됩니다. **반드시 `function_list` API를 호출**하여 최신 함수 목록을 확인하세요.

### 인증 불필요 API

| 함수명 | 설명 |
|--------|------|
| `function_list` | **API 함수 목록 조회 (필수 사용!)** |
| `version` | API 버전 정보 |
| `build_date` | 빌드 날짜 조회 (Docker 빌드 시점 UTC) |
| `get_user` | 사용자 공개 정보 조회 (`id`: PostgreSQL users.id, `uid`: Firebase UID 또는 숫자일 경우 users.id) |
| `get_user_hovercard` | 사용자 Hovercard 정보 조회 |
| `check_subdomain` | 서브도메인 사용 가능 여부 확인 |
| `get_branch_admin_email` | 가맹사 운영자 이메일 조회 |
| `list_category` | 카테고리 목록 조회 |
| `get_category` | 카테고리 단일 조회 |
| `get_category_by_slug` | 카테고리 슬러그로 조회 |
| `get_shared_categories_by_country` | 특정 국가의 공유 카테고리 목록 |
| `list_post` | 게시글 목록 조회 |
| `get_post` | 게시글 상세 조회 |
| `search_posts` | TypeSense 게시글 검색 |
| `search_posts_by_country` | 국가 코드 기반 게시글 검색 |
| `search_posts_by_share_category` | 공유 카테고리별 게시글 검색 |
| `list_comment` | 댓글 목록 조회 |
| `get_countries` | 국가 목록 조회 |
| `get_popular_keywords` | 인기 검색어 조회 |
| `get_search_statistics_years` | 검색 통계 가능한 년도 목록 |
| `get_banner_point_cost` | 배너 포인트 비용 조회 |
| `get_all_banner_point_costs` | 모든 배너 유형 포인트 비용 |
| `get_active_banners` | 활성 배너 조회 |
| `get_post_list_banners` | 게시글 목록 배너 조회 |
| `get_bank_accounts` | 입금 계좌 목록 조회 |

### 인증 필요 API (token 필수)

| 함수명 | 설명 |
|--------|------|
| `my` | 내 정보 조회 |
| `logout` | 로그아웃 처리 |
| `update_user` | 사용자 정보 수정 |
| `update_profile_photo_url` | 프로필 사진 URL 업데이트 |
| `register_branch` | Branch 등록 |
| `get_my_branch` | 내 Branch 정보 조회 |
| `list_my_branches` | 내가 운영하는 모든 Branch 목록 |
| `update_branch` | Branch 업데이트 |
| `update_branch_settings` | Branch 설정 업데이트 |
| `update_branch_layout` | Branch 레이아웃 업데이트 |
| `get_branch_meta` | Branch 메타 정보 조회 |
| `update_branch_meta` | Branch 메타 정보 업데이트 |
| `delete_branch_meta` | Branch 메타 정보 삭제 |
| `get_categories` | 현재 도메인 카테고리 목록 |
| `get_branch_categories` | 가맹사 카테고리 목록 (배열 반환) |
| `create_category` | 카테고리 생성 |
| `update_category` | 카테고리 수정 |
| `delete_category` | 카테고리 삭제 |
| `reorder_category` | 카테고리 순서 변경 (단일) |
| `bulk_reorder_category` | 카테고리 순서 일괄 변경 |
| `link_to_share_category` | 가맹사 카테고리를 공유 카테고리에 연결 |
| `unlink_from_share_category` | 공유 카테고리 연결 해제 |
| `create_post` | 게시글 생성 |
| `update_post` | 게시글 수정 |
| `delete_post` | 게시글 삭제 |
| `list_my_posts` | 내 게시글 목록 조회 (카테고리 필터 지원) |
| `create_comment` | 댓글 생성 |
| `update_comment` | 댓글 수정 |
| `delete_comment` | 댓글 삭제 |
| `file_upload` | 파일 업로드 |
| `file_delete` | 파일 삭제 |
| `get_my_point` | 내 포인트 조회 |
| `get_point_history` | 포인트 내역 조회 |
| `create_banner` | 배너 등록 |
| `start_banner` | 배너 시작 (포인트 차감) |
| `stop_banner` | 배너 중단 (잔여 기간 환불) |
| `update_banner` | 배너 수정 |
| `delete_banner` | 배너 삭제 |
| `list_banners` | 배너 목록 조회 |
| `get_my_banners` | 내 배너 목록 조회 |
| `create_report` | 신고 생성 |
| `list_notification` | 알림 목록 조회 |
| `count_unread_notification` | 읽지 않은 알림 수 |
| `mark_notification_read` | 단일 알림 읽음 처리 |
| `mark_all_notifications_read` | 전체 알림 읽음 처리 |
| `list_reaction` | 반응 목록 조회 |
| `get_reaction_stats` | 반응 통계 조회 |
| `count_unread_reaction` | 읽지 않은 반응 수 |
| `mark_reaction_read` | 단일 반응 읽음 처리 |
| `mark_all_reactions_read` | 모든 반응 읽음 처리 |

### 관리자 전용 API

| 함수명 | 권한 | 설명 |
|--------|------|------|
| `list_branch` | 루트 | 가맹사 목록 조회 |
| `register_branch_domain` | 루트 | 새 도메인(Branch) 등록 |
| `change_branch_operator` | 루트 | Branch 운영자 변경 |
| `root_list_post` | 루트 | 전체 게시글 조회 |
| `root_list_user` | 루트 | 전체 회원 조회 |
| `root_count_all` | 루트 | 전체 통계 조회 |
| `admin_charge_point` | 루트 | 포인트 충전 |
| `admin_deduct_point` | 루트 | 포인트 차감 |
| `search_users` | 루트 | 사용자 검색 |
| `list_shared_categories` | 루트 | 공유 카테고리 목록 |
| `create_shared_category` | 루트 | 공유 카테고리 생성 |
| `update_shared_category` | 루트 | 공유 카테고리 수정 |
| `delete_shared_category` | 루트 | 공유 카테고리 삭제 |
| `list_linked_categories` | 루트 | 공유 카테고리에 연결된 목록 |
| `create_all_shared_categories` | 루트 | 모든 공유 카테고리 일괄 생성 |
| `delete_all_shared_categories` | 루트 | 모든 공유 카테고리 삭제 |
| `list_all_reports` | 루트 | 전체 신고 목록 조회 |
| `get_search_statistics` | 루트/일반 | 검색 통계 조회 |
| `register_branch_admin` | 가맹사 운영자 | 운영자 등록 |
| `user_list_by_branch` | 가맹사 운영자 | 소속 회원 조회 |
| `admin_update_user` | 가맹사 운영자 | 회원 정보 수정 |
| `admin_get_user` | 가맹사 운영자/루트 | 회원 전체 정보 조회 (id 또는 uid) |
| `branch_admin_charge_point` | 가맹사 운영자 | 회원 포인트 충전 |
| `branch_admin_deduct_point` | 가맹사 운영자 | 회원 포인트 차감 |
| `list_report_by_branch` | 가맹사 운영자 | Branch별 신고 목록 |
| `get_recent_reports` | 가맹사 운영자 | 최근 신고 목록 |
| `handle_report` | 가맹사 운영자 | 신고 처리 |
| `start_report_review` | 가맹사 운영자 | 신고 검토 시작 |
| `count_pending_reports` | 가맹사 운영자 | 대기 중인 신고 수 |
| `save_banner_settings` | 가맹사 운영자 | 배너 설정 저장 |

---

## 7. 에러 코드 목록

| HTTP 상태 | 에러 코드 | 설명 |
|-----------|-----------|------|
| 400 | `invalid-json` | 잘못된 JSON 형식 |
| 400 | `func-required` | func 파라미터 누락 |
| 400 | `invalid-func-format` | 함수명 형식 오류 |
| 400 | `assert-token/token-required` | 토큰 누락 |
| 400 | `assert-token/invalid-api-token-hash` | 토큰 해시 불일치 |
| 400 | `failed-to-verify-token` | 토큰 검증 실패 |
| 403 | `func-not-accessible` | 접근 불가한 함수 |
| 403 | `permission-denied` | 권한 없음 |
| 404 | `func-not-found` | 존재하지 않는 함수 |
| 404 | `user/not-found` | 사용자 없음 |
| 404 | `category/not-found` | 카테고리 없음 |
| 404 | `post/not-found` | 게시글 없음 |
| 405 | `method-not-allowed` | POST 외 HTTP 메서드 |

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [api.md](api.md) | API 개발자를 위한 문서 |
| [user-token.md](user-token.md) | 토큰 생성 상세 설명 |
| [test-process.md](test-process.md) | API 테스트 가이드 |
| [post.md](post.md) | 게시글 시스템 |
| [batch-post-creation.md](batch-post-creation.md) | 배치 게시글 생성 |

---

## 소스코드 파일 위치

| 파일 | 설명 |
|------|------|
| `api.php` | API 진입점 |
| `lib/api/api.allowed_functions.php` | AllowedFunctions 클래스 (모든 API 함수 정의) |
| `lib/api/api.functions.php` | API 공통 함수 (error() 등) |
| `lib/Service/UserService.php` | 토큰 생성/검증 메서드 |
