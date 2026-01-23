센터 구인 공고 기능 개발
=======================


개요
-----------------------
센터 구인 공고 기능은 사용자들이 센터에서 제공하는 구인 정보를 쉽게 찾고 지원할 수 있도록 돕는 기능입니다. 이 기능은 구인 공고의 작성, 조회, 수정, 삭제 및 지원 프로세스를 포함합니다.



주요 기능
-----------------------

게시판과 통합
- 구인 공고는 기존 게시판 시스템과 통합되어 사용자들이 쉽게 접근할 수 있습니다. 따라서 제목과 내용 필드를 활용하여 구인 공고를 작성할 수 있습니다.
- `share-xx-job-hiring` 카테고리인 경우만, 글 쓰기 양식과 글 보여주기 디자인을 별도로 제공합니다.


데이터베이스 구조
-----------------------

가능한 posts 테이블을 활용하며, 구인 공고 관련 추가 필드는 post_meta 테이블을 사용합니다.

- id: 고유 식별자
- branch_id: 가맹사(센터) 식별자
- user_id: 작성자 식별자
- category_id: share-xx-job-hiring
- country_code: 가맹사의 country_code
- title: 구인 공고 제목
- content: 구인 공고 소개
- view_count: 조회수
- urls: 구인 공고 관련 링크 (예: 지원서 링크). 이미지 업로드한 링크도 포함 가능


구인 공고(게시판)에 추가할 필드 목록

### 기본 정보
| 필드 이름      | 타입        | 설명                         |
|----------------|-------------|------------------------------|
| position       | VARCHAR(255)| 모집 직책                    |
| location       | VARCHAR(255)| 근무지 위치                  |
| employment_type| VARCHAR(100)| 고용 형태 (예: 정규직, 계약직, 파트타임, 인턴)|
| deadline       | TIMESTAMP   | 지원 마감일                  |
| hiring_count   | VARCHAR(50) | 채용 인원 (예: 1명, 0명, 00명)|
| start_date     | TIMESTAMP   | 입사 예정일                  |

### 급여 정보
| 필드 이름        | 타입        | 설명                         |
|------------------|-------------|------------------------------|
| salary           | VARCHAR(100)| 급여 정보 (텍스트 형식)      |
| salary_type      | VARCHAR(50) | 급여 유형 (annual/monthly/hourly/daily)|
| min_salary       | VARCHAR(50) | 최소 급여                    |
| max_salary       | VARCHAR(50) | 최대 급여                    |
| salary_currency  | VARCHAR(10) | 급여 통화 (예: KRW, USD, JPY)|
| salary_negotiable| VARCHAR(10) | 급여 협상 가능 여부 (yes/no) |

### 경력 요건
| 필드 이름           | 타입        | 설명                         |
|---------------------|-------------|------------------------------|
| experience_level    | VARCHAR(50) | 경력 수준 (신입/경력/무관)   |
| min_experience_years| VARCHAR(10) | 최소 경력 연수               |
| max_experience_years| VARCHAR(10) | 최대 경력 연수               |
| education_level     | VARCHAR(100)| 학력 요건 (예: 고졸, 대졸, 석사)|

### 근무 조건
| 필드 이름       | 타입        | 설명                         |
|-----------------|-------------|------------------------------|
| work_hours      | VARCHAR(100)| 근무 시간 (예: 09:00-18:00)  |
| work_days       | VARCHAR(100)| 근무 요일 (예: 월-금)        |
| remote_work     | VARCHAR(50) | 원격 근무 가능 여부 (yes/no/hybrid)|
| probation_period| VARCHAR(50) | 수습 기간 (예: 3개월)        |

### 자격 요건 및 업무
| 필드 이름      | 타입        | 설명                         |
|----------------|-------------|------------------------------|
| requirements   | TEXT        | 자격 요건                    |
| responsibilities| TEXT       | 주요 업무 내용               |
| skills         | TEXT        | 필요 기술/스킬 (쉼표 구분)   |
| languages      | TEXT        | 언어 요건 (예: 영어 비즈니스급)|
| certifications | TEXT        | 자격증 요건                  |

### 복리후생
| 필드 이름      | 타입        | 설명                         |
|----------------|-------------|------------------------------|
| benefits       | TEXT        | 복리후생                     |
| visa_sponsorship| VARCHAR(10)| 비자 지원 여부 (yes/no)      |

### 회사/조직 정보
| 필드 이름      | 타입        | 설명                         |
|----------------|-------------|------------------------------|
| company_name   | VARCHAR(255)| 회사명                       |
| department     | VARCHAR(255)| 부서명                       |
| team_size      | VARCHAR(50) | 팀 규모 (예: 5-10명)         |
| industry       | VARCHAR(100)| 업종 (예: IT, 제조, 서비스)  |

### 지원 관련
| 필드 이름       | 타입        | 설명                         |
|-----------------|-------------|------------------------------|
| application_url | VARCHAR(500)| 지원서 링크                  |
| application_email| VARCHAR(255)| 지원 이메일                 |
| contact_person  | VARCHAR(100)| 담당자명                     |
| contact_phone   | VARCHAR(50) | 담당자 연락처                |
| application_docs| TEXT        | 제출 서류 (예: 이력서, 포트폴리오)|

### 원본 출처 (선택)
| 필드 이름         | 타입        | 설명                         |
|-------------------|-------------|------------------------------|
| origin_source_url | VARCHAR(500)| 원본 구인 공고 URL (선택사항). 해당 구인 정보의 원문이 있는 외부 사이트 링크 |
- posts 테이블의 title, content 필드를 활용하여 구인 공고의 기본 정보를 저장합니다.
- 추가적인 구인 공고 세부 정보는 post_meta 테이블에 key-value 쌍으로 저장됩니다.
- post_meta 테이블 참고: [post_meta.md](./post-meta.md)


API를 통한 구인 공고 등록
-----------------------

### CURL 예제 - 모든 필드 포함

```bash
curl -s -X POST 'https://sonub.com/api.php' \
  -H 'Content-Type: application/json' \
  -d '{
    "func": "create_post",
    "token": "apikey-{user_id}-{md5_hash}",
    "category_slug": "share-sg-jobs-hiring",
    "title": "[테크노바] 시니어 풀스택 개발자 채용",
    "content": "회사 소개 및 추가 상세 내용을 입력하세요.\n\n- 빠르게 성장하는 핀테크 스타트업\n- 글로벌 진출 준비 중",
    "urls": [
      "https://example.com/company-logo.jpg"
    ],
    "meta": {
      "company_name": "테크노바 주식회사",
      "position": "시니어 풀스택 개발자",
      "department": "플랫폼 개발팀",
      "industry": "IT",
      "location": "서울시 강남구 테헤란로 123",
      "employment_type": "정규직",
      "remote_work": "hybrid",
      "work_hours": "10:00-19:00 (유연근무제)",
      "work_days": "월-금",
      "hiring_count": "2명",
      "probation_period": "3개월",
      "team_size": "8-12명",
      "salary": "7,000만원 ~ 1억원",
      "salary_type": "annual",
      "salary_currency": "KRW",
      "salary_negotiable": "yes",
      "min_salary": "70000000",
      "max_salary": "100000000",
      "experience_level": "경력",
      "min_experience_years": "5",
      "max_experience_years": "10",
      "education_level": "대졸",
      "languages": "영어 비즈니스 회화 가능자 우대",
      "deadline": "2026-02-28",
      "start_date": "2026-03-15",
      "responsibilities": "- 핀테크 플랫폼 백엔드 API 설계 및 개발\n- 프론트엔드 React 개발\n- 시스템 아키텍처 설계",
      "requirements": "- 풀스택 개발 경력 5년 이상\n- Node.js, React 실무 경험\n- PostgreSQL/MongoDB 경험",
      "skills": "Node.js, React, TypeScript, PostgreSQL, AWS",
      "certifications": "AWS Solutions Architect 우대",
      "benefits": "- 4대보험\n- 연차 15일\n- 점심 제공\n- 자기개발비 지원",
      "visa_sponsorship": "yes",
      "application_url": "https://technova.com/careers/apply",
      "application_email": "hr@technova.com",
      "contact_person": "김인사",
      "contact_phone": "010-1234-5678",
      "application_docs": "이력서, 포트폴리오, 자기소개서",
      "origin_source_url": "https://jobkorea.com/jobs/12345"
    }
  }' | jq .
```

### CURL 예제 - 필수 항목만

```bash
curl -s -X POST 'https://sonub.com/api.php' \
  -H 'Content-Type: application/json' \
  -d '{
    "func": "create_post",
    "token": "apikey-{user_id}-{md5_hash}",
    "category_slug": "share-sg-jobs-hiring",
    "title": "[회사명] 개발자 채용",
    "meta": {
      "company_name": "회사명",
      "position": "개발자",
      "industry": "IT",
      "location": "서울시 강남구",
      "employment_type": "정규직",
      "deadline": "2026-02-28",
      "responsibilities": "주요 업무 내용",
      "application_email": "hr@company.com",
      "contact_person": "담당자",
      "contact_phone": "010-1234-5678"
    }
  }' | jq .
```

### 토큰 생성

API 토큰은 다음 공식으로 생성됩니다:

```
토큰 형식: apikey-{user_id}-{md5_hash}
MD5 해시: MD5(user_id + FLOOR(created_at_timestamp) + email + branch_id)
```

서버에서 동적으로 생성하는 방법:

```bash
# Docker 컨테이너에서 토큰 생성
docker exec center-php php -r "
  require '/var/www/html/lib/bootstrap.php';
  echo (new UserService())->generateUserTokenById(168);
"
```

자세한 내용은 [user-token.md](./user-token.md) 및 [api-spec.md](./api-spec.md) 참조.


필드 설명표
-----------------------

### posts 테이블 필드 (기본)

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `title` | ✅ | 공고 제목 | `[회사명] 직책 채용` |
| `content` | | 상세 내용 | 추가 설명 |
| `category_slug` | ✅ | 카테고리 슬러그 | `share-sg-jobs-hiring` |
| `urls` | | 첨부 이미지/파일 URL 배열 | `["https://..."]` |

### meta 필드 - 회사/조직 정보

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `company_name` | ✅ | 회사명 | `테크노바 주식회사` |
| `position` | ✅ | 모집 직책 | `시니어 개발자` |
| `department` | | 부서/팀 | `개발팀` |
| `industry` | ✅ | 업종 | `IT`, `제조업`, `서비스업`, `금융/은행`, `유통/물류`, `건설/부동산`, `교육`, `의료/제약`, `무역/수출입`, `외식/숙박`, `미디어/엔터테인먼트`, `기타` |

### meta 필드 - 근무 조건

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `location` | ✅ | 근무지 | `서울시 강남구` |
| `employment_type` | ✅ | 고용 형태 | `정규직`, `계약직`, `인턴`, `아르바이트`, `프리랜서` |
| `remote_work` | | 원격 근무 | `no`(불가), `yes`(가능), `hybrid`(하이브리드), `full`(완전 원격) |
| `work_hours` | | 근무 시간 | `09:00-18:00` |
| `work_days` | | 근무 요일 | `월-금` |
| `hiring_count` | | 채용 인원 | `2명` |
| `probation_period` | | 수습 기간 | `3개월` |
| `team_size` | | 팀 규모 | `10-15명` |

### meta 필드 - 급여 정보

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `salary` | | 급여 (텍스트) | `5,000만원 ~ 7,000만원` |
| `salary_type` | | 급여 유형 | `annual`(연봉), `monthly`(월급), `hourly`(시급), `negotiable`(협의) |
| `salary_currency` | | 통화 | `KRW`, `USD`, `PHP`, `JPY`, `CNY`, `EUR` |
| `salary_negotiable` | | 협상 가능 | `yes`, `no` |
| `min_salary` | | 최소 급여 (숫자) | `50000000` |
| `max_salary` | | 최대 급여 (숫자) | `70000000` |

### meta 필드 - 경력 요건

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `experience_level` | | 경력 수준 | `신입`, `경력`, `무관` |
| `min_experience_years` | | 최소 경력(년) | `3` |
| `max_experience_years` | | 최대 경력(년) | `10` |
| `education_level` | | 학력 | `학력무관`, `고졸`, `초대졸`, `대졸`, `석사`, `박사` |
| `languages` | | 언어 요건 | `영어 비즈니스급` |

### meta 필드 - 채용 일정

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `deadline` | ✅ | 지원 마감일 | `2026-02-28` |
| `start_date` | | 입사 예정일 | `2026-03-15` |

### meta 필드 - 상세 정보

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `responsibilities` | ✅ | 주요 업무 | 담당 업무 내용 (여러 줄 가능) |
| `requirements` | | 자격 요건 | 필수/우대 조건 |
| `skills` | | 필요 기술 | `React, Node.js, TypeScript` |
| `certifications` | | 자격증 | `정보처리기사, AWS Solutions Architect` |

### meta 필드 - 복리후생

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `benefits` | | 복리후생 | `4대보험, 연차, 점심 제공` |
| `visa_sponsorship` | | 비자 지원 | `yes`, `no` |

### meta 필드 - 지원 방법

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `application_url` | | 지원서 링크 | `https://company.com/apply` |
| `application_email` | ✅ | 지원 이메일 | `hr@company.com` |
| `contact_person` | ✅ | 담당자명 | `홍길동` |
| `contact_phone` | ✅ | 담당자 연락처 | `010-1234-5678` |
| `application_docs` | | 제출 서류 | `이력서, 포트폴리오` |

### meta 필드 - 원본 출처

| 필드명 | 필수 | 설명 | 예시 값 |
|--------|:----:|------|---------|
| `origin_source_url` | | 원본 출처 URL | `https://jobkorea.com/jobs/12345` |

> **참고:** 원본 출처 URL은 해당 구인 정보의 원문이 있는 외부 사이트 링크입니다. 뉴스나 다른 사이트에서 가져온 구인 정보를 등록할 때 출처를 명시하기 위해 사용합니다.