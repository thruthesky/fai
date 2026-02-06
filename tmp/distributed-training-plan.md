# FAI 분산 학습 시스템 — 상세 구현 계획

## 1. 비전 및 목표

### 1.1 프로젝트 비전

전 세계 자발적 참여자들이 자신의 컴퓨터(GPU/CPU)를 제공하여,
협력적으로 대규모 LLM을 학습시키는 **오픈 분산 학습 플랫폼**을 구축합니다.

```
BOINC (과학 분산 컴퓨팅) + Federated Learning (연합 학습)
= FAI 분산 학습 시스템
```

### 1.2 핵심 요구사항

| 요구사항 | 설명 |
|----------|------|
| **자유 참여/이탈** | 스크립트 실행으로 참여, Ctrl+C로 이탈. 다른 워커에 영향 없음 |
| **하드웨어 무관** | NVIDIA GPU, Apple Silicon, CPU 모두 참여 가능 |
| **대규모 확장** | 수십 ~ 수만 대의 컴퓨터가 동시 참여 가능 |
| **장기 학습** | 며칠 ~ 몇 개월에 걸친 지속적 학습 |
| **진행 보존** | 어떤 워커가 빠져도 학습 진행 상태 유지 |
| **기여도 추적** | 누가 얼마나 기여했는지 투명하게 기록 |

### 1.3 시스템 개요

```
┌──────────────────────────────────────────────────────────────┐
│                    중앙 서버 (Coordinator)                      │
│                    (24시간 상시 가동)                            │
│                                                              │
│   ┌──────────────────────────────────────────────────┐    │
│   │              Supabase (자체 호스팅)                 │    │
│   │                                                    │    │
│   │  ┌─────────────────────┐  ┌──────────────────┐   │    │
│   │  │ Supabase PostgreSQL  │  │  Supabase Storage │   │    │
│   │  │                     │  │  (S3 호환)         │   │    │
│   │  │ - 워커 관리/실시간 상태│  │                   │   │    │
│   │  │ - 작업 큐 (SKIP LOCKED)│ │ - 체크포인트 저장  │   │    │
│   │  │ - 메트릭 / 기여도     │  │ - 데이터셋 배포    │   │    │
│   │  │ - 이벤트 알림 (NOTIFY)│  │                   │   │    │
│   │  │ - 분산 락 (advisory)  │  │                   │   │    │
│   │  └─────────────────────┘  └──────────────────┘   │    │
│   └──────────────────────────────────────────────────┘    │
│                                                              │
│   ┌──────────────────────────────────────┐                  │
│   │         Coordinator API Server        │                  │
│   │         (FastAPI / REST + WebSocket)   │                  │
│   └──────────────────────┬───────────────┘                  │
└──────────────────────────┼───────────────────────────────────┘
                           │ HTTPS
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ 워커 A    │     │ 워커 B    │     │ 워커 C    │
    │ (GPU)    │     │ (CPU)    │     │ (GPU)    │
    │ 오전 참여  │     │ 밤에 참여  │     │ 주말 참여  │
    └──────────┘     └──────────┘     └──────────┘
```

---

## 2. 기술 스택

### 2.1 확정 기술

| 구성요소 | 기술 | 이유 |
|----------|------|------|
| **학습 프레임워크** | PyTorch | 기존 FAI 프로젝트가 PyTorch 기반 |
| **BaaS 플랫폼** | Supabase (자체 호스팅) | PostgreSQL + Storage + Auth 통합 플랫폼 |
| **DB** | Supabase PostgreSQL | 작업 관리, 기여도 추적, 트랜잭션, 작업 큐, 이벤트 알림 |
| **파일 스토리지** | Supabase Storage | 체크포인트 파일 관리, S3 호환 API |
| **API 서버** | FastAPI (Python) | PyTorch와 같은 언어, 비동기 지원 |
| **파일 전송** | HTTP (청크 업/다운로드) | 체크포인트 파일 전송 |

### 2.2 인프라 구성 — Supabase 활용

```
기존 계획: Docker로 PostgreSQL 직접 운영 + MinIO 별도 설치
변경 계획: 이미 운영 중인 Supabase 인스턴스를 활용

Supabase 제공 기능 중 활용할 것:
  ✅ PostgreSQL         — 모든 테이블, 작업 큐, 이벤트 알림
  ✅ Storage (S3 호환)   — 체크포인트 파일, 데이터셋 파일 저장/배포
  ⬜ Auth (선택적)       — 워커 인증 (현재는 API 키 방식으로 충분)
  ⬜ Edge Functions     — 향후 서버리스 API 확장 시 활용 가능

설정 파일: .environments (Supabase 접속 정보)
  → 파이썬 코드에서 이 파일을 로드하여 DB 접속
```

### 2.3 권장 추가 기술

| 구성요소 | 기술 | 이유 |
|----------|------|------|
| **오브젝트 스토리지** | Supabase Storage (기본) | 이미 Supabase에 포함, 별도 설치 불필요 |
| **모니터링** | Prometheus + Grafana | 워커 상태, 학습 메트릭 실시간 대시보드 |
| **메시지 큐** (대규모 시) | RabbitMQ 또는 Kafka | 수만 대 워커 시 Supabase PostgreSQL LISTEN/NOTIFY의 한계 보완 |
| **리버스 프록시** | Nginx | API 서버 앞단, SSL 종료, 로드밸런싱 |

### 2.4 Supabase PostgreSQL 단독 구성 — 역할 상세

```
Supabase PostgreSQL 하나로 모든 역할을 수행:
──────────────────────────────────

영속적 데이터                            실시간 기능 (Supabase PostgreSQL 내장)
─────────────                           ──────────────────────
✅ 워커 등록/프로필                       ✅ 워커 heartbeat → last_seen 컬럼 + 주기적 쿼리
✅ 체크포인트 이력                        ✅ 활성 워커 목록 → WHERE status = 'online'
✅ 기여(contribution) 기록               ✅ 작업 큐 → SELECT FOR UPDATE SKIP LOCKED
✅ 학습 메트릭 (loss, step 히스토리)       ✅ 분산 락 → pg_advisory_lock()
✅ 실험 설정/하이퍼파라미터               ✅ 이벤트 알림 → LISTEN / NOTIFY
✅ 감사 로그 (audit trail)               ✅ API 크레딧 잔액 → 직접 쿼리
✅ API 키 / 토큰 크레딧                  ✅ Rate limiting → 앱 메모리 카운터
```

---

## 3. 프로젝트 구조 (신규 파일)

```
fai/
├── (기존 파일들 유지)
├── scripts/
│   ├── (기존 스크립트 유지)
│   ├── train_gpt.py              # 기존 (수정 필요)
│   └── ...
│
├── distributed/                   # ★ 신규: 분산 학습 패키지
│   ├── __init__.py
│   │
│   ├── server/                    # 중앙 서버 (Coordinator)
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI 앱 진입점
│   │   ├── config.py              # 서버 설정 (.environments에서 Supabase 접속 정보 로드)
│   │   ├── models.py              # SQLAlchemy ORM 모델 (Supabase PostgreSQL)
│   │   ├── schemas.py             # Pydantic 스키마 (API 요청/응답)
│   │   ├── database.py            # Supabase PostgreSQL 연결, 세션, 유틸리티
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── workers.py         # 워커 등록/상태 API
│   │   │   ├── checkpoints.py     # 체크포인트 업로드/다운로드 API
│   │   │   ├── tasks.py           # 작업 할당/완료 API
│   │   │   └── metrics.py         # 메트릭 조회 API
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── merger.py          # ★ 핵심: FedAvg 병합 엔진
│   │   │   ├── scheduler.py       # 작업 스케줄러 (워커에게 작업 할당)
│   │   │   ├── validator.py       # 기여 검증 (악의적 gradient 탐지)
│   │   │   └── heartbeat.py       # 워커 생존 확인 서비스
│   │   └── migrations/            # Alembic DB 마이그레이션
│   │       └── ...
│   │
│   ├── worker/                    # 워커 클라이언트 (팀원 컴퓨터에서 실행)
│   │   ├── __init__.py
│   │   ├── cli.py                 # CLI 진입점 (python -m distributed.worker)
│   │   ├── config.py              # 워커 설정
│   │   ├── client.py              # 서버 API 통신 클라이언트
│   │   ├── trainer.py             # 로컬 학습 루프 (train_gpt.py 기반)
│   │   ├── device_manager.py      # GPU/CPU 자동 감지 및 최적화
│   │   └── checkpoint_io.py       # 체크포인트 다운로드/업로드
│   │
│   └── common/                    # 서버/워커 공통 모듈
│       ├── __init__.py
│       ├── constants.py           # 공통 상수
│       ├── serialization.py       # 모델 가중치 직렬화/역직렬화
│       └── protocol.py            # 통신 프로토콜 정의
│
├── .environments                  # ★ Supabase 접속 정보 (기존 파일, git 제외)
├── docker/                        # Docker 배포 (Coordinator만, DB는 Supabase 사용)
│   ├── docker-compose.yml         # Coordinator API 서버
│   ├── Dockerfile.server          # 서버 이미지
│   └── Dockerfile.worker          # 워커 이미지 (선택적)
│
└── distributed-training-plan.md   # 이 문서
```

---

## 4. 데이터베이스 스키마

### 4.1 Supabase PostgreSQL 테이블

```sql
-- ============================================================
-- 1. 워커 (참여자의 컴퓨터)
-- ============================================================
CREATE TABLE workers (
    id              SERIAL PRIMARY KEY,
    worker_uid      UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,              -- "철수의 맥북", "영희 RTX 서버"
    hostname        TEXT,
    ip_address      INET,

    -- 하드웨어 정보
    device_type     TEXT NOT NULL,              -- 'cuda', 'mps', 'cpu'
    device_name     TEXT,                       -- 'RTX 4090', 'M4 Pro', 'i7-12700'
    gpu_memory_mb   INTEGER,                    -- GPU 메모리 (MB), NULL이면 CPU
    ram_mb          INTEGER,                    -- 시스템 RAM (MB)
    cpu_cores       INTEGER,                    -- CPU 코어 수

    -- 성능 벤치마크 (첫 참여 시 자동 측정)
    benchmark_score FLOAT,                      -- 상대적 성능 점수 (GPU=100 기준)
    recommended_batch_size INTEGER,             -- 이 하드웨어에 권장되는 배치 크기
    recommended_local_steps INTEGER,            -- 권장 로컬 학습 step 수

    -- 상태
    status          TEXT DEFAULT 'offline',      -- 'online', 'training', 'uploading', 'offline'
    total_contributions INTEGER DEFAULT 0,
    total_steps_trained BIGINT DEFAULT 0,

    -- 타임스탬프
    first_seen      TIMESTAMP DEFAULT NOW(),
    last_seen       TIMESTAMP DEFAULT NOW(),

    -- 신뢰도 (악의적 참여자 방지)
    trust_score     FLOAT DEFAULT 1.0,          -- 0.0 ~ 1.0
    is_banned       BOOLEAN DEFAULT FALSE,

    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_workers_status ON workers(status);
CREATE INDEX idx_workers_uid ON workers(worker_uid);

-- ============================================================
-- 2. 학습 실험 (Experiment / Run)
-- ============================================================
CREATE TABLE experiments (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,              -- "fai-v1-small", "fai-v2-large"
    description     TEXT,
    status          TEXT DEFAULT 'active',      -- 'active', 'paused', 'completed'

    -- 모델 하이퍼파라미터 (모든 워커가 동일한 모델 사용)
    config          JSONB NOT NULL,             -- {vocab_size, block_size, n_layer, ...}

    -- 학습 설정
    target_steps    BIGINT,                     -- 목표 총 학습 step
    local_steps_per_round INTEGER DEFAULT 50,   -- 워커당 1라운드에 수행할 step 수
    max_stale_gap   INTEGER DEFAULT 200,        -- 허용하는 최대 stale step 갭
    merge_strategy  TEXT DEFAULT 'fedavg',      -- 'fedavg', 'weighted_fedavg'

    -- 현재 진행 상태
    current_global_step BIGINT DEFAULT 0,
    current_train_loss  FLOAT,
    current_val_loss    FLOAT,
    best_val_loss       FLOAT,

    -- 데이터셋 정보
    dataset_checksum    TEXT,                   -- train.bin의 SHA256 해시
    tokenizer_checksum  TEXT,                   -- tokenizer.json의 SHA256 해시

    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 3. 체크포인트 (모델의 진화 이력)
-- ============================================================
CREATE TABLE checkpoints (
    id              SERIAL PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    global_step     BIGINT NOT NULL,
    round_number    INTEGER NOT NULL,           -- 병합 라운드 번호

    -- 파일 정보
    file_path       TEXT NOT NULL,              -- 스토리지 내 경로
    file_size_bytes BIGINT,
    file_checksum   TEXT,                       -- SHA256

    -- 성능 지표
    train_loss      FLOAT,
    val_loss        FLOAT,

    -- 병합 정보
    merged_from     JSONB,                      -- [{worker_id, steps, loss, weight}, ...]
    num_contributors INTEGER DEFAULT 0,         -- 이 체크포인트에 기여한 워커 수

    -- 상태
    is_latest       BOOLEAN DEFAULT FALSE,      -- 최신 체크포인트 여부
    is_best         BOOLEAN DEFAULT FALSE,      -- 최고 성능 체크포인트 여부

    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_checkpoints_experiment ON checkpoints(experiment_id);
CREATE INDEX idx_checkpoints_latest ON checkpoints(is_latest) WHERE is_latest = TRUE;
CREATE INDEX idx_checkpoints_step ON checkpoints(global_step);

-- ============================================================
-- 4. 기여 (Contribution) — 워커의 학습 결과
-- ============================================================
CREATE TABLE contributions (
    id                  SERIAL PRIMARY KEY,
    experiment_id       INTEGER NOT NULL REFERENCES experiments(id),
    worker_id           INTEGER NOT NULL REFERENCES workers(id),

    -- 기반 체크포인트
    base_checkpoint_id  INTEGER NOT NULL REFERENCES checkpoints(id),
    base_global_step    BIGINT NOT NULL,

    -- 학습 결과
    steps_trained       INTEGER NOT NULL,       -- 로컬에서 학습한 step 수
    local_train_loss    FLOAT,                  -- 학습 후 train loss
    local_val_loss      FLOAT,                  -- 학습 후 val loss

    -- 업로드된 가중치
    upload_path         TEXT,                   -- 업로드된 가중치 파일 경로
    upload_size_bytes   BIGINT,
    upload_checksum     TEXT,

    -- 메타데이터
    device_type         TEXT,                   -- 학습에 사용된 디바이스
    batch_size_used     INTEGER,
    learning_rate_used  FLOAT,
    training_duration_s FLOAT,                  -- 학습 소요 시간 (초)

    -- 검증 및 상태
    status              TEXT DEFAULT 'pending', -- 'pending', 'validating', 'merged', 'rejected', 'expired'
    rejection_reason    TEXT,                   -- 거부 사유
    merge_weight        FLOAT,                  -- 병합 시 적용된 가중치
    stale_gap           INTEGER,                -- 제출 시점과 글로벌 step의 갭

    submitted_at        TIMESTAMP DEFAULT NOW(),
    validated_at        TIMESTAMP,
    merged_at           TIMESTAMP
);

CREATE INDEX idx_contributions_experiment ON contributions(experiment_id);
CREATE INDEX idx_contributions_worker ON contributions(worker_id);
CREATE INDEX idx_contributions_status ON contributions(status);
CREATE INDEX idx_contributions_pending ON contributions(status) WHERE status = 'pending';

-- ============================================================
-- 5. 학습 메트릭 히스토리
-- ============================================================
CREATE TABLE training_metrics (
    id              SERIAL PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    global_step     BIGINT NOT NULL,
    round_number    INTEGER,

    train_loss      FLOAT,
    val_loss        FLOAT,
    num_active_workers INTEGER,
    num_contributions  INTEGER,                 -- 이 라운드에 기여한 수

    recorded_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_metrics_experiment_step ON training_metrics(experiment_id, global_step);

-- ============================================================
-- 6. 감사 로그 (Audit Log)
-- ============================================================
CREATE TABLE audit_log (
    id              SERIAL PRIMARY KEY,
    event_type      TEXT NOT NULL,              -- 'worker_joined', 'worker_left',
                                               -- 'contribution_submitted', 'merge_completed',
                                               -- 'worker_banned', 'checkpoint_created'
    actor_id        INTEGER,                    -- worker_id 또는 NULL (시스템)
    details         JSONB,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_audit_event ON audit_log(event_type);
CREATE INDEX idx_audit_time ON audit_log(created_at);

-- ============================================================
-- 7. API 키 및 사용량 (기여 보상 시스템)
-- ============================================================
CREATE TABLE api_keys (
    id              SERIAL PRIMARY KEY,
    worker_id       INTEGER NOT NULL REFERENCES workers(id),
    api_key         UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    name            TEXT,                       -- "내 앱용 키", "테스트용"
    is_active       BOOLEAN DEFAULT TRUE,

    -- 토큰 크레딧 (학습 기여 → API 사용량)
    earned_tokens   BIGINT DEFAULT 0,           -- 학습 기여로 적립된 총 토큰 수
    used_tokens     BIGINT DEFAULT 0,           -- API 호출로 사용한 토큰 수
    -- 잔여 크레딧 = earned_tokens - used_tokens

    -- 사용 제한
    rate_limit_rpm  INTEGER DEFAULT 60,         -- 분당 최대 요청 수
    max_tokens_per_request INTEGER DEFAULT 256, -- 요청당 최대 토큰 수

    created_at      TIMESTAMP DEFAULT NOW(),
    last_used_at    TIMESTAMP,
    expires_at      TIMESTAMP                   -- NULL이면 무기한
);

CREATE INDEX idx_api_keys_key ON api_keys(api_key);
CREATE INDEX idx_api_keys_worker ON api_keys(worker_id);

-- API 사용 로그 (토큰 소비 추적)
CREATE TABLE api_usage_log (
    id              SERIAL PRIMARY KEY,
    api_key_id      INTEGER NOT NULL REFERENCES api_keys(id),
    worker_id       INTEGER NOT NULL REFERENCES workers(id),

    -- 요청 정보
    endpoint        TEXT NOT NULL,              -- '/v1/completions', '/v1/chat'
    prompt_tokens   INTEGER NOT NULL,           -- 입력 토큰 수
    completion_tokens INTEGER NOT NULL,         -- 출력 토큰 수
    total_tokens    INTEGER NOT NULL,           -- prompt + completion

    -- 메타데이터
    model_version   TEXT,                       -- 사용한 모델 버전 (체크포인트 ID)
    response_time_ms INTEGER,                   -- 응답 시간 (ms)
    status_code     INTEGER DEFAULT 200,        -- HTTP 상태 코드

    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_api_usage_key ON api_usage_log(api_key_id);
CREATE INDEX idx_api_usage_worker ON api_usage_log(worker_id);
CREATE INDEX idx_api_usage_time ON api_usage_log(created_at);

-- 토큰 크레딧 트랜잭션 이력 (적립/차감 내역)
CREATE TABLE token_transactions (
    id              SERIAL PRIMARY KEY,
    worker_id       INTEGER NOT NULL REFERENCES workers(id),
    api_key_id      INTEGER REFERENCES api_keys(id),  -- 차감 시에만

    -- 트랜잭션 정보
    type            TEXT NOT NULL,              -- 'earn' (적립), 'spend' (사용), 'bonus' (보너스), 'expire' (만료)
    amount          BIGINT NOT NULL,            -- 토큰 수 (earn: 양수, spend: 음수)
    balance_after   BIGINT NOT NULL,            -- 트랜잭션 후 잔액

    -- 적립 근거 (type='earn' 시)
    contribution_id INTEGER REFERENCES contributions(id),  -- 어떤 기여로 적립되었는지
    steps_trained   INTEGER,                    -- 해당 기여의 학습 step 수

    -- 설명
    description     TEXT,                       -- "50 step 학습 기여 → 800 토큰 적립"

    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_token_tx_worker ON token_transactions(worker_id);
CREATE INDEX idx_token_tx_type ON token_transactions(type);
```

### 4.2 Supabase PostgreSQL 대체 패턴

기존에 인메모리 DB가 담당하던 역할을 Supabase PostgreSQL 내장 기능으로 대체합니다.

#### 워커 Heartbeat (실시간 상태 감지)

```sql
-- 워커가 30초마다 heartbeat 전송 시:
UPDATE workers SET last_seen = NOW(), status = 'training' WHERE worker_uid = $1;

-- 오프라인 워커 감지 (서버가 주기적으로 실행, 60초 기준):
UPDATE workers SET status = 'offline'
WHERE status != 'offline' AND last_seen < NOW() - INTERVAL '60 seconds';

-- 활성 워커 목록 조회:
SELECT * FROM workers WHERE status IN ('online', 'training') AND last_seen > NOW() - INTERVAL '60 seconds';
```

#### 작업 큐 (SELECT FOR UPDATE SKIP LOCKED)

```sql
-- 대기 중인 기여를 큐처럼 가져오기 (동시에 여러 프로세스가 접근해도 안전):
WITH next_contributions AS (
    SELECT id FROM contributions
    WHERE status = 'pending' AND experiment_id = $1
    ORDER BY submitted_at
    LIMIT $2                          -- merge_threshold 개수만큼
    FOR UPDATE SKIP LOCKED            -- 다른 프로세스가 잠근 행은 건너뜀
)
UPDATE contributions SET status = 'validating'
WHERE id IN (SELECT id FROM next_contributions)
RETURNING *;
```

#### 분산 락 (pg_advisory_lock)

```sql
-- 병합 시작 전 락 획득 (experiment_id를 락 키로 사용):
SELECT pg_try_advisory_lock($experiment_id);   -- 성공 시 TRUE, 이미 잠김이면 FALSE

-- 병합 완료 후 락 해제:
SELECT pg_advisory_unlock($experiment_id);

-- 또는 트랜잭션 범위 락 (트랜잭션 종료 시 자동 해제):
SELECT pg_advisory_xact_lock($experiment_id);
```

#### 이벤트 알림 (LISTEN / NOTIFY)

```sql
-- 서버: 새 체크포인트 생성 시 알림 발행
NOTIFY checkpoint_updated, '{"experiment_id": 1, "global_step": 1100}';

-- 워커: 알림 수신 대기 (비동기, asyncpg 지원)
LISTEN checkpoint_updated;

-- 실험 제어 명령
NOTIFY experiment_control, '{"experiment_id": 1, "action": "pause"}';

-- 워커 이벤트
NOTIFY worker_event, '{"worker_uid": "abc-123", "event": "joined"}';
```

#### API 크레딧 잔액 조회 (직접 쿼리)

```sql
-- API 키 검증 + 잔액 확인 (한 번의 쿼리로):
SELECT ak.id, ak.worker_id, ak.is_active,
       ak.earned_tokens, ak.used_tokens,
       (ak.earned_tokens - ak.used_tokens) AS remaining
FROM api_keys ak
JOIN workers w ON ak.worker_id = w.id
WHERE ak.api_key = $1
  AND ak.is_active = TRUE
  AND w.is_banned = FALSE;

-- 원자적 크레딧 차감 (잔액 부족 시 실패):
UPDATE api_keys
SET used_tokens = used_tokens + $1, last_used_at = NOW()
WHERE api_key = $2
  AND is_active = TRUE
  AND (earned_tokens - used_tokens) >= $1  -- 잔액 확인과 차감을 원자적으로
RETURNING earned_tokens - used_tokens AS remaining;

-- Rate limiting (앱 메모리 카운터 + 주기적 DB 체크):
-- 앱 레벨에서 collections.defaultdict 또는 sliding window 카운터 사용
-- 서버 재시작 시에만 DB에서 최근 1분 요청 수 복구
SELECT COUNT(*) FROM api_usage_log
WHERE api_key_id = $1 AND created_at > NOW() - INTERVAL '1 minute';
```

---

## 5. API 설계

### 5.1 Coordinator REST API

```
기본 URL: https://<server>/api/v1

인증: API Key (헤더: X-Worker-Key: <uuid>)
     첫 등록 시 발급, 이후 모든 요청에 포함
```

#### 워커 관리

```
POST   /workers/register           # 워커 등록 (첫 참여 시)
  요청: { name, device_type, device_name, gpu_memory_mb, ram_mb, cpu_cores }
  응답: { worker_uid, api_key, recommended_batch_size, recommended_local_steps }

POST   /workers/heartbeat          # 생존 신호 (30초마다)
  요청: { worker_uid, status, current_local_step }
  응답: { ok, server_time, experiment_status }

POST   /workers/benchmark          # 벤치마크 결과 보고 (첫 참여 시)
  요청: { worker_uid, benchmark_score, steps_per_second }
  응답: { recommended_batch_size, recommended_local_steps }

GET    /workers/me                 # 내 정보 조회
  응답: { worker info, total_contributions, total_steps, trust_score }

POST   /workers/leave              # 명시적 이탈 (Ctrl+C 시 호출)
  요청: { worker_uid }
  응답: { ok }
```

#### 학습 작업

```
GET    /experiments/{id}/status    # 실험 상태 조회
  응답: { global_step, train_loss, val_loss, active_workers, latest_checkpoint_url }

POST   /tasks/request              # 학습 작업 요청 (워커 → 서버)
  요청: { worker_uid, experiment_id }
  응답: {
    task_id,
    checkpoint_url,               # 다운로드할 체크포인트 URL
    base_global_step,             # 기반 글로벌 step
    local_steps,                  # 수행할 로컬 step 수
    batch_size,                   # 사용할 배치 크기
    learning_rate,                # 사용할 학습률
    dataset_url                   # 데이터셋 다운로드 URL (첫 참여 시)
  }

POST   /tasks/{task_id}/complete   # 학습 완료 보고 (워커 → 서버)
  요청: {
    worker_uid,
    steps_trained,
    local_train_loss,
    local_val_loss,
    training_duration_s,
    upload_checksum
  }
  + 멀티파트 파일 업로드: 학습된 가중치 파일

GET    /tasks/{task_id}/status     # 작업 상태 조회
  응답: { status, merged_at, merge_weight }
```

#### 체크포인트

```
GET    /checkpoints/latest         # 최신 체크포인트 다운로드
  응답: 바이너리 파일 스트림 (.pt)

GET    /checkpoints/{id}/download  # 특정 체크포인트 다운로드
  응답: 바이너리 파일 스트림 (.pt)

GET    /checkpoints/history        # 체크포인트 히스토리
  응답: [{ id, global_step, train_loss, val_loss, created_at }, ...]
```

#### 메트릭 및 대시보드

```
GET    /metrics/summary            # 전체 요약
  응답: {
    global_step, train_loss, val_loss,
    total_workers, active_workers,
    total_contributions, total_steps_trained,
    steps_per_hour, estimated_completion
  }

GET    /metrics/loss-history       # Loss 추이
  응답: [{ step, train_loss, val_loss, timestamp }, ...]

GET    /metrics/leaderboard        # 기여도 리더보드
  응답: [{ name, device, contributions, total_steps, avg_loss }, ...]

GET    /metrics/workers            # 활성 워커 목록
  응답: [{ name, device, status, current_step, last_seen }, ...]
```

#### API 키 및 크레딧 관리

```
POST   /api-keys/create             # API 키 발급 (워커 프로필 기반)
  요청: { worker_uid, name }
  응답: { api_key, earned_tokens, used_tokens, remaining_tokens }

GET    /api-keys/list               # 내 API 키 목록 조회
  응답: [{ api_key, name, earned_tokens, used_tokens, remaining, is_active, created_at }, ...]

GET    /api-keys/{key}/balance      # 잔여 크레딧 조회
  응답: { earned_tokens, used_tokens, remaining_tokens, recent_transactions }

DELETE /api-keys/{key}              # API 키 비활성화
  응답: { ok }

GET    /api-keys/{key}/usage        # 사용 내역 조회
  쿼리: ?from=2026-01-01&to=2026-02-06
  응답: {
    total_requests, total_tokens_used,
    daily_breakdown: [{ date, requests, tokens }, ...]
  }

GET    /api-keys/{key}/transactions # 토큰 적립/차감 이력
  응답: [{ type, amount, balance_after, description, created_at }, ...]
```

#### LLM 추론 API (학습된 모델 사용 — API 키 필수)

```
POST   /v1/completions              # 텍스트 생성 (API 키 인증)
  헤더: Authorization: Bearer <api_key>
  요청: { prompt, max_tokens, temperature }
  응답: {
    text, prompt_tokens, completion_tokens, total_tokens,
    remaining_credits               # 잔여 크레딧 안내
  }

  에러 응답 (크레딧 부족):
  { error: "insufficient_credits", earned: 50000, used: 50000, remaining: 0 }
```

### 5.2 WebSocket (실시간 이벤트)

```
WS /ws/events?worker_uid={uid}

서버 → 워커 이벤트:
  { type: "checkpoint_updated", global_step: 1100, checkpoint_url: "..." }
  { type: "experiment_paused", reason: "관리자에 의해 일시 중지" }
  { type: "experiment_resumed" }
  { type: "config_updated", new_local_steps: 100 }
  { type: "worker_stats", active: 42, steps_per_hour: 1200 }
```

---

## 6. 핵심 알고리즘

### 6.1 Federated Averaging (FedAvg) 병합

```
입력:
  W_global     = 현재 글로벌 모델 가중치
  {W_1, W_2, ..., W_k} = k개 워커의 로컬 학습 결과 가중치
  {n_1, n_2, ..., n_k} = 각 워커가 학습한 데이터 샘플 수
  {s_1, s_2, ..., s_k} = 각 워커의 stale gap

알고리즘:
  1. 각 워커의 가중치 계산:
     - 기본 가중치: α_i = n_i / Σn_j  (데이터 비례)
     - Stale 감쇠:  α_i *= 1.0 / (1.0 + s_i / max_stale_gap)
     - 신뢰도 반영: α_i *= trust_score_i
     - 정규화:      α_i /= Σα_j

  2. 가중 평균 병합:
     W_new = Σ(α_i × W_i)

  3. 글로벌 모델 업데이트:
     W_global ← W_new

  4. 새 체크포인트 저장
```

### 6.2 병합 트리거 전략

단일 기여가 도착할 때마다 즉시 병합하지 않고, 효율적으로 묶어서 병합합니다.

```
트리거 조건 (OR):
  1. 대기 중인 기여 수 >= merge_threshold (기본값: 3)
  2. 마지막 병합 이후 경과 시간 >= merge_timeout (기본값: 300초)
  3. 대기 중인 기여의 총 step 수 >= step_threshold (기본값: 100)

예시:
  - 활성 워커 3대: 3개 기여 도착 시 즉시 병합
  - 활성 워커 1대: 300초 대기 후 1개 기여라도 병합
  - 대규모 (100대+): merge_threshold를 동적으로 조절
```

### 6.3 Stale Contribution 처리

```
기여 제출 시:
  stale_gap = current_global_step - contribution.base_global_step

판정:
  ┌──────────────────┬───────────────────────────────────┐
  │ stale_gap 범위    │ 처리                               │
  ├──────────────────┼───────────────────────────────────┤
  │ 0 ~ 50           │ ✅ 정상 수용 (가중치 1.0)            │
  │ 51 ~ 200         │ ⚠️ 감쇠 수용 (가중치 점차 감소)       │
  │ 201 이상          │ ❌ 거부, 워커에게 최신 모델 재다운로드  │
  └──────────────────┴───────────────────────────────────┘

워커에게 거부 응답:
  { status: "rejected", reason: "stale", latest_checkpoint_url: "..." }
  → 워커는 자동으로 최신 체크포인트를 다운로드하고 재시작
```

### 6.4 악의적 참여자 탐지

```
검증 단계 (기여 수용 전):

  1. Loss 이상 탐지
     - 기여의 local_loss가 글로벌 loss의 3배 이상 → 의심
     - 연속 3회 이상 이상 loss 제출 → 경고

  2. 가중치 이상 탐지
     - 업로드된 가중치에 NaN/Inf 포함 → 즉시 거부
     - 가중치 변화량(delta)이 비정상적으로 큰 경우 → 의심
       delta = ||W_uploaded - W_base|| / ||W_base||
       delta > threshold (예: 0.5) → 거부

  3. 신뢰도 시스템
     - 정상 기여 시: trust_score += 0.01 (최대 1.0)
     - 이상 기여 시: trust_score -= 0.1
     - trust_score < 0.3 → 자동 차단 (is_banned = TRUE)

  4. 검증 학습 (Validation Run)
     - 수상한 기여는 서버에서 소량 val 데이터로 검증
     - val_loss가 기존보다 현저히 나쁘면 거부
```

---

## 7. 워커 클라이언트 상세 설계

### 7.1 워커 생명주기

```
$ python -m distributed.worker --name "내 컴퓨터" --server https://fai.example.com

┌─────────────────────────────────────────────────────────┐
│                    워커 시작                              │
│                                                         │
│  [1] 서버 연결 확인                                       │
│  [2] 하드웨어 감지 (GPU/CPU, 메모리)                       │
│  [3] 워커 등록 (첫 실행 시) 또는 재접속                     │
│  [4] 벤치마크 (첫 실행 시: 더미 모델로 속도 측정)            │
│      → 서버가 최적 batch_size, local_steps 결정            │
│  [5] 데이터셋 다운로드 (첫 실행 시: train.bin, val.bin)     │
│      → 로컬 캐시에 저장, checksum으로 최신 여부 확인         │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
          ┌─────── 메인 루프 시작 ──────┐
          │                            │
          ▼                            │
  ┌───────────────┐                    │
  │ 서버에 작업 요청 │ ◄─────────────────┤
  │ GET /tasks     │                    │
  └───────┬───────┘                    │
          │                            │
          ▼                            │
  ┌───────────────────┐                │
  │ 최신 체크포인트     │                │
  │ 다운로드           │                │
  │ (변경 시에만)       │                │
  └───────┬───────────┘                │
          │                            │
          ▼                            │
  ┌───────────────────┐                │
  │ 로컬 학습 수행      │                │
  │                   │                │
  │ for step in range(local_steps):    │
  │   x, y = get_batch()              │
  │   loss = model(x, y)              │
  │   loss.backward()                 │
  │   optimizer.step()                │
  │                   │                │
  │ + 30초마다 heartbeat 전송           │
  │ + Ctrl+C 시 graceful shutdown      │
  └───────┬───────────┘                │
          │                            │
          ▼                            │
  ┌───────────────────┐                │
  │ 결과 업로드         │                │
  │                   │                │
  │ - model.state_dict() 직렬화         │
  │ - 서버에 업로드     │                │
  │ - 검증 대기        │                │
  └───────┬───────────┘                │
          │                            │
          ▼                            │
  ┌───────────────────┐                │
  │ 서버 응답 확인      │                │
  │                   │                │
  │ merged?  → 다음 라운드 ─────────────┘
  │ rejected? → 최신 모델 재다운로드 ────┘
  │ error?   → 재시도 (최대 3회) ────────┘
  └───────────────────┘

  Ctrl+C 감지 시:
  ┌───────────────────┐
  │ Graceful Shutdown  │
  │                   │
  │ 1. 현재 학습 중단   │
  │ 2. 서버에 이탈 통보 │
  │ 3. 로컬 임시파일 정리│
  │ 4. 종료            │
  └───────────────────┘
```

### 7.2 하드웨어 자동 감지 및 최적화

```
디바이스 감지 순서:
  1. NVIDIA GPU (torch.cuda) → backend: 'cuda'
  2. Apple Silicon (torch.backends.mps) → backend: 'mps'
  3. CPU fallback → backend: 'cpu'

배치 크기 자동 결정 (벤치마크 기반):
  ┌─────────────────┬──────────────┬──────────────┐
  │ GPU 메모리       │ 배치 크기     │ local_steps  │
  ├─────────────────┼──────────────┼──────────────┤
  │ 24GB+ (RTX4090) │ 64           │ 100          │
  │ 12GB  (RTX3060) │ 32           │ 100          │
  │ 8GB   (RTX3050) │ 16           │ 50           │
  │ MPS (M4 Pro)    │ 16           │ 50           │
  │ MPS (M4)        │ 8            │ 50           │
  │ CPU only        │ 4            │ 25           │
  └─────────────────┴──────────────┴──────────────┘

  * 실제 값은 벤치마크 결과에 따라 서버가 결정
  * OOM(Out of Memory) 발생 시 자동으로 batch_size 절반으로 재시도
```

### 7.3 체크포인트 전송 최적화

```
문제: 모델 가중치 파일이 클 수 있음 (수십MB ~ 수GB)

최적화 전략:

  1. Delta 전송 (차분 전송)
     - 전체 가중치 대신, 변화분(delta)만 전송
     - delta = W_local - W_base
     - 서버에서: W_result = W_base + delta
     - 효과: 전송량 50~80% 감소 (변화가 작은 레이어가 많으므로)

  2. 압축
     - delta를 gzip 또는 lz4로 압축
     - float32 → float16 변환 후 전송 (정밀도 약간 손실, 크기 50% 감소)
     - 효과: 추가 50% 감소

  3. 청크 업로드 (대용량 시)
     - 5MB 청크 단위로 분할 업로드
     - 중간에 끊겨도 이어서 업로드 가능 (resumable upload)

  전체 효과 예시:
    원본 가중치: 40MB
    → delta 추출: 20MB
    → float16 변환: 10MB
    → gzip 압축: 4MB
    실제 전송량: 4MB (원본의 10%)
```

### 7.4 워커 CLI 인터페이스

```
# 기본 실행
$ python -m distributed.worker \
    --name "철수의 맥북" \
    --server https://fai.example.com \
    --experiment 1

# 고급 옵션
$ python -m distributed.worker \
    --name "영희 GPU 서버" \
    --server https://fai.example.com \
    --experiment 1 \
    --device cuda:0 \               # 특정 GPU 지정
    --batch-size 32 \               # 배치 크기 수동 지정 (자동 감지 무시)
    --local-steps 100 \             # 라운드당 step 수 수동 지정
    --max-rounds 10 \               # 최대 라운드 수 (없으면 무한)
    --data-dir ./fai-data \         # 데이터셋 로컬 캐시 경로
    --verbose                       # 상세 로그 출력

# 실행 화면 예시
╔══════════════════════════════════════════════════════╗
║  FAI 분산 학습 워커 v1.0                              ║
║  서버: https://fai.example.com                       ║
║  실험: fai-v1-small (#1)                             ║
╠══════════════════════════════════════════════════════╣
║  디바이스: mps (Apple M4 Pro, 18GB)                   ║
║  배치 크기: 16 | 로컬 step: 50                        ║
║  데이터셋: ✅ 캐시됨 (train.bin: 99KB)                 ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  [라운드 1/∞] 글로벌 step: 1050                       ║
║  ├─ 체크포인트 다운로드: ████████████████ 100%         ║
║  ├─ 로컬 학습: ██████████░░░░░░░░░░  25/50 step      ║
║  │  └─ loss: 2.15 → 2.08 (↓0.07)                    ║
║  ├─ 업로드: 대기 중                                    ║
║  └─ 병합: 대기 중                                     ║
║                                                      ║
║  📊 내 기여: 총 150 step, 3 라운드 완료                 ║
║  🌍 글로벌: 42명 참여 중, 1,200 step/시간              ║
║                                                      ║
║  Ctrl+C: 현재 라운드 완료 후 안전 종료                   ║
║  Ctrl+C×2: 즉시 종료 (학습 결과 폐기)                   ║
╚══════════════════════════════════════════════════════╝
```

---

## 8. 서버 (Coordinator) 상세 설계

### 8.1 Coordinator 핵심 서비스

```
┌──────────────────────────────────────────────────────┐
│                  Coordinator 내부 구조                 │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │  API Server   │  │  Merge Worker │                 │
│  │  (FastAPI)    │  │  (백그라운드)   │                 │
│  │              │  │              │                 │
│  │ - REST API   │  │ - 기여 수집   │                 │
│  │ - WebSocket  │  │ - FedAvg 병합 │                 │
│  │ - 파일 업/다운│  │ - 체크포인트   │                 │
│  │              │  │   저장        │                 │
│  └──────┬───────┘  └──────┬───────┘                 │
│         │                 │                          │
│  ┌──────┴─────────────────┴──────┐                  │
│  │       공통 서비스 레이어         │                  │
│  │                               │                  │
│  │ - Heartbeat Monitor           │                  │
│  │   (오프라인 워커 감지, 60초 주기) │                  │
│  │                               │                  │
│  │ - Scheduler                   │                  │
│  │   (워커별 최적 작업 할당)        │                  │
│  │                               │                  │
│  │ - Validator                   │                  │
│  │   (악의적 기여 탐지)            │                  │
│  └───────────────────────────────┘                  │
└──────────────────────────────────────────────────────┘
```

### 8.2 병합 워커 (Merge Worker) 프로세스

```
별도 프로세스 또는 백그라운드 태스크로 실행:

while True:
  1. Supabase PostgreSQL에서 pending 기여 확인
     SELECT COUNT(*) FROM contributions
     WHERE status = 'pending' AND experiment_id = $1;

  2. 트리거 조건 확인
     if count >= merge_threshold OR time_since_last_merge >= merge_timeout:

  3. 분산 락 획득 (pg_advisory_lock)
     SELECT pg_try_advisory_lock($experiment_id);
     if not acquired: continue  # 다른 프로세스가 병합 중

  4. pending 기여 가져오기 (행 잠금으로 안전하게)
     SELECT * FROM contributions
     WHERE status = 'pending' AND experiment_id = $1
     ORDER BY submitted_at LIMIT $merge_threshold
     FOR UPDATE SKIP LOCKED;

  5. 각 기여의 가중치 파일 로드
     for c in contributions:
       W_i = torch.load(c.upload_path)

  6. FedAvg 실행
     W_new = federated_average(W_global, [W_i, ...], [weight_i, ...])

  7. 검증
     val_loss_new = evaluate(W_new, val_data)
     if val_loss_new > val_loss_old * 1.5:
       # 병합 결과가 크게 나빠짐 → 롤백
       reject_contributions(contributions)
       continue

  8. 새 체크포인트 저장 (Supabase Storage)
     save_checkpoint(W_new, new_step)

  9. DB 업데이트 (하나의 트랜잭션으로)
     INSERT INTO checkpoints (...)
     UPDATE contributions SET status = 'merged' WHERE id IN (...)
     UPDATE experiments SET current_global_step = $new_step,
                            current_train_loss = $loss

  10. 이벤트 알림 (Supabase PostgreSQL NOTIFY)
      NOTIFY checkpoint_updated, '{"experiment_id": 1, "step": 1100}'

  11. 락 해제
      SELECT pg_advisory_unlock($experiment_id);

  12. sleep(check_interval)
```

---

## 9. 스케일링 전략

### 9.1 규모별 아키텍처 변화

```
┌─────────────┬──────────────────────────────────────────────────┐
│ 규모         │ 아키텍처                                          │
├─────────────┼──────────────────────────────────────────────────┤
│             │                                                  │
│ 10대 이하    │  단일 서버 (API + Supabase)                        │
│ (팀 프로젝트) │  FastAPI 1 프로세스                                │
│             │  Supabase PostgreSQL: 단일 인스턴스                │
│             │  체크포인트: Supabase Storage                      │
│             │                                                  │
├─────────────┼──────────────────────────────────────────────────┤
│             │                                                  │
│ 10~100대     │  API 서버 2~4 프로세스 (uvicorn workers)           │
│ (소규모 커뮤  │  Supabase PostgreSQL: Supavisor 커넥션 풀링       │
│  니티)       │  체크포인트: Supabase Storage                      │
│             │  Nginx 리버스 프록시                                │
│             │                                                  │
├─────────────┼──────────────────────────────────────────────────┤
│             │                                                  │
│ 100~1,000대  │  API 서버: 로드밸런서 + 다수 인스턴스               │
│ (대규모 커뮤  │  Supabase PostgreSQL: Primary-Replica 구성        │
│  니티)       │  체크포인트: S3 / GCS                               │
│             │  Merge Worker: 전용 GPU 서버에서 실행               │
│             │  모니터링: Prometheus + Grafana                    │
│             │                                                  │
├─────────────┼──────────────────────────────────────────────────┤
│             │                                                  │
│ 1,000~      │  API 서버: Kubernetes 오토스케일링                  │
│ 10,000대     │  Supabase PostgreSQL: Citus 또는 파티셔닝          │
│ (글로벌 프로  │  체크포인트: CDN + S3 (지역별 캐시)                 │
│  젝트)       │  Merge Worker: 큐 기반 다중 인스턴스                │
│             │  작업 큐: RabbitMQ 또는 Kafka (NOTIFY 한계 시)      │
│             │  지역별 릴레이 서버 (지연시간 최적화)                 │
│             │                                                  │
└─────────────┴──────────────────────────────────────────────────┘
```

### 9.2 대규모 시 병합 전략 변화

```
소규모 (10대): 모든 기여를 한번에 병합
  W_new = avg(W_1, W_2, ..., W_10)

중규모 (100대): 계층적 병합 (Hierarchical FedAvg)
  그룹1: avg(W_1 ~ W_10)  → W_group1
  그룹2: avg(W_11 ~ W_20) → W_group2
  ...
  최종:  avg(W_group1, W_group2, ...) → W_new

대규모 (1000대+): 비동기 계층적 병합
  - 기여 도착 즉시 부분 병합 (streaming aggregation)
  - 서버 메모리에 누적 합산 유지
  - running_sum += α_i × W_i
  - running_count += α_i
  - 주기적으로: W_new = running_sum / running_count
```

### 9.3 네트워크 최적화

```
문제: 수만 명이 동시에 체크포인트를 다운로드하면 대역폭 폭발

해결:

  1. CDN 활용
     - 체크포인트를 CDN(CloudFlare, AWS CloudFront)에 캐시
     - 워커는 가장 가까운 CDN 엣지에서 다운로드

  2. P2P 체크포인트 공유 (선택적, 대규모 시)
     - 이미 최신 체크포인트를 가진 워커가 다른 워커에게 전달
     - BitTorrent 방식의 피어 공유

  3. 증분 체크포인트 (Incremental Checkpoint)
     - 전체 모델 대신 이전 체크포인트와의 차분만 배포
     - 변경된 레이어만 다운로드

  4. 체크포인트 버전 관리
     - 모든 워커가 항상 최신일 필요는 없음
     - stale_gap 허용 범위 내라면 이전 버전으로도 학습 가능
     - → 다운로드 빈도 감소
```

---

## 10. 구현 단계 (Phase)

### Phase 1: 기반 구축 (1~2주)

```
목표: 단일 서버에서 2~3대 워커로 기본 흐름이 작동하는 것

작업:
  □ 프로젝트 구조 생성 (distributed/ 패키지)
  □ config.py (.environments 파일에서 Supabase 접속 정보 로드)
  □ database.py (Supabase PostgreSQL 연결, 세션, LISTEN/NOTIFY 유틸리티)
  □ Supabase PostgreSQL에 테이블 생성 (Alembic 마이그레이션)
  □ Supabase Storage 버킷 생성 (checkpoints, datasets)
  □ docker-compose.yml (Coordinator API 서버만, DB는 Supabase 사용)

산출물:
  - distributed/server/config.py
  - distributed/server/database.py
  - distributed/server/models.py
  - distributed/common/constants.py
  - docker/docker-compose.yml
  - DB 마이그레이션 파일
```

### Phase 2: 서버 API 구현 (2~3주)

```
목표: Coordinator API가 완전히 작동하는 것

작업:
  □ FastAPI 앱 기본 구조 (app.py)
  □ 워커 등록/heartbeat API
  □ 작업 요청/완료 API
  □ 체크포인트 업로드/다운로드 API
  □ 메트릭 조회 API
  □ Heartbeat 모니터링 (백그라운드 태스크, last_seen 기반)
  □ LISTEN/NOTIFY 이벤트 시스템
  □ API 키 인증 미들웨어

산출물:
  - distributed/server/app.py
  - distributed/server/routes/*.py
  - distributed/server/database.py
  - distributed/server/schemas.py
```

### Phase 3: 워커 클라이언트 구현 (2~3주)

```
목표: 워커가 서버에서 작업을 받아 로컬 학습 후 결과를 업로드하는 것

작업:
  □ CLI 진입점 (cli.py)
  □ 서버 통신 클라이언트 (client.py)
  □ 하드웨어 자동 감지 (device_manager.py)
  □ 로컬 학습 루프 (trainer.py) — 기존 train_gpt.py 기반
  □ 체크포인트 다운로드/업로드 (checkpoint_io.py)
  □ Graceful shutdown (SIGINT 처리)
  □ Heartbeat 백그라운드 스레드
  □ 자동 벤치마크

산출물:
  - distributed/worker/*.py
  - distributed/common/serialization.py
```

### Phase 4: 병합 엔진 구현 (1~2주)

```
목표: FedAvg 병합이 자동으로 작동하는 것

작업:
  □ FedAvg 병합 알고리즘 (merger.py)
  □ 병합 트리거 로직 (시간/개수 기반)
  □ Stale contribution 감지 및 처리
  □ 기본 검증 (NaN/Inf 체크, loss 이상 탐지)
  □ 체크포인트 저장 및 DB 업데이트
  □ Pub/Sub 알림

산출물:
  - distributed/server/services/merger.py
  - distributed/server/services/validator.py
  - distributed/server/services/scheduler.py
```

### Phase 5: 통합 테스트 및 안정화 (1~2주)

```
목표: 3~5대 워커로 안정적으로 학습이 진행되는 것

작업:
  □ 통합 테스트 시나리오 작성
  □ 워커 참여/이탈 시나리오 테스트
  □ 네트워크 단절 시 복구 테스트
  □ 장시간 학습 안정성 테스트 (24시간+)
  □ 에러 처리 및 재시도 로직 보강
  □ 로깅 시스템 정비

산출물:
  - tests/ 폴더
  - 운영 가이드 문서
```

### Phase 6: 모니터링 및 대시보드 (1주)

```
목표: 학습 현황을 실시간으로 확인할 수 있는 것

작업:
  □ 메트릭 수집 (Prometheus exporter)
  □ Grafana 대시보드 구성
    - 글로벌 학습 진행 (step, loss 곡선)
    - 활성 워커 수 추이
    - 기여도 리더보드
    - 워커별 성능 비교
  □ 또는 간단한 웹 대시보드 (HTML + Chart.js)

산출물:
  - 대시보드 설정 파일
  - 또는 distributed/server/routes/dashboard.py
```

### Phase 7: 스케일링 및 보안 (필요 시)

```
목표: 100대 이상 워커 지원, 보안 강화

작업:
  □ Delta 전송 구현
  □ 체크포인트 압축
  □ 계층적 병합
  □ 악의적 참여자 탐지 강화
  □ Rate limiting
  □ SSL/TLS 적용
  □ API 키 관리 시스템

산출물:
  - 최적화된 serialization.py
  - 보안 미들웨어
```

---

## 11. 기존 코드 수정 사항

### 11.1 train_gpt.py 수정 계획

기존 `train_gpt.py`의 핵심 컴포넌트를 재사용하되, 분산 워커 전용 학습 루프를 별도로 작성합니다.

```
기존 train_gpt.py에서 재사용할 것:
  ✅ GPT 모델 클래스 (CausalSelfAttention, MLP, Block, GPT)
  ✅ CFG 설정 구조
  ✅ get_batch() 함수
  ✅ estimate_loss() 함수
  ✅ get_device() 함수

분산 워커에서 새로 작성할 것:
  🆕 학습 루프 (N step만 수행 후 종료)
  🆕 체크포인트 로드 (서버에서 다운로드한 파일)
  🆕 결과 저장 (state_dict만 저장, 옵티마이저 상태 제외 가능)
  🆕 서버 통신 (heartbeat, 진행 보고)

방법: train_gpt.py의 모델 정의 부분을 공통 모듈로 추출
  scripts/train_gpt.py          → 기존 단독 학습 (유지)
  distributed/common/model.py   → GPT 모델 클래스 (train_gpt.py에서 추출)
  distributed/worker/trainer.py → 분산 학습 루프 (model.py 임포트)
```

### 11.2 의존성 추가 (pyproject.toml)

```toml
# 기존 의존성 유지 + 추가
[project]
dependencies = [
    # 기존
    "numpy>=2.4.1",
    "tokenizers>=0.22.2",
    "torch>=2.9.1",
    "tqdm>=4.67.1",

    # 서버 (distributed.server)
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.14.0",
    "asyncpg>=0.30.0",         # Supabase PostgreSQL 비동기 드라이버
    "supabase>=2.0.0",        # Supabase Python 클라이언트 (Storage 접근)
    "python-multipart>=0.0.9", # 파일 업로드
    "pydantic>=2.0.0",         # 데이터 검증

    # 워커 (distributed.worker)
    "httpx>=0.27.0",           # HTTP 클라이언트 (비동기)
    "websockets>=13.0",        # WebSocket 클라이언트
    "click>=8.0.0",            # CLI 프레임워크

    # 공통
    "psutil>=6.0.0",           # 시스템 정보 (CPU, RAM, GPU)
]
```

---

## 12. 데이터 흐름 전체도

```
┌────────────────────────────────────────────────────────────────────┐
│                        전체 데이터 흐름                              │
│                                                                    │
│  [한번만 실행 - 사전 준비]                                           │
│                                                                    │
│  raw.txt                                                           │
│    → prepare_samples.py → samples.txt                              │
│    → train_tokenizer.py → tokenizer.json                           │
│    → build_bin_dataset.py → train.bin + val.bin                    │
│                                                                    │
│  위 파일들을 서버 스토리지에 업로드                                    │
│  + 초기 모델 체크포인트 생성 (랜덤 초기화)                             │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [반복 - 분산 학습 루프]                                             │
│                                                                    │
│  워커 참여                                                          │
│    │                                                               │
│    ├─ 1. 서버에서 train.bin, val.bin 다운로드 (첫 참여 시)            │
│    │     → 로컬 캐시에 저장 (checksum으로 최신 여부 확인)              │
│    │                                                               │
│    ├─ 2. 서버에서 최신 체크포인트 (ckpt.pt) 다운로드                   │
│    │     → model.load_state_dict(checkpoint)                       │
│    │                                                               │
│    ├─ 3. 로컬에서 N step 학습                                       │
│    │     → train.bin에서 랜덤 배치 추출                              │
│    │     → forward → backward → optimizer.step()                   │
│    │     → val.bin으로 loss 측정                                    │
│    │                                                               │
│    ├─ 4. 학습된 가중치를 서버에 업로드                                 │
│    │     → model.state_dict() → 직렬화 → HTTP 업로드                │
│    │                                                               │
│    └─ 5. 서버가 FedAvg 병합                                        │
│          → 여러 워커의 가중치를 가중 평균                              │
│          → 새 체크포인트 저장                                        │
│          → 2번으로 돌아감                                            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  글로벌 모델 진화:                                         │      │
│  │                                                          │      │
│  │  ckpt_r0 (초기) → ckpt_r1 → ckpt_r2 → ... → ckpt_rN    │      │
│  │  loss: 8.5       loss: 6.2   loss: 4.1       loss: 1.5  │      │
│  │                                                          │      │
│  │  각 라운드마다 여러 워커의 기여가 병합됨                     │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 13. 실행 가이드 (최종 목표)

### 13.1 서버 실행

```bash
# 0. 전제 조건: Supabase가 이미 동작 중이어야 함
#    .environments 파일에 Supabase 접속 정보가 있어야 함

# 1. DB 마이그레이션 (Supabase PostgreSQL에 테이블 생성)
#    .environments에서 POSTGRES_PASSWORD, SUPABASE_HOST 등을 읽어 접속
uv run alembic upgrade head

# 2. 초기 데이터 준비 (한번만)
uv run python scripts/prepare_samples.py
uv run python scripts/train_tokenizer.py
uv run python scripts/build_bin_dataset.py

# 3. 실험 생성 및 초기 체크포인트 등록
#    체크포인트 파일은 Supabase Storage에 업로드됨
uv run python -m distributed.server.init_experiment \
    --name "fai-v1" \
    --train-bin data/train.bin \
    --val-bin data/val.bin \
    --tokenizer data/tokenizer.json

# 4. Coordinator API 서버 실행
uv run uvicorn distributed.server.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

### 13.2 워커 실행 (팀원/참여자)

```bash
# 1. FAI 프로젝트 클론
git clone https://github.com/example/fai.git
cd fai

# 2. 의존성 설치
uv sync

# 3. 학습 참여! (이것 하나만 실행하면 됨)
uv run python -m distributed.worker \
    --name "내 컴퓨터" \
    --server https://fai.example.com

# 4. 종료하고 싶으면 Ctrl+C
# 5. 다시 참여하고 싶으면 3번 다시 실행
```

---

## 14. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 악의적 워커가 잘못된 가중치 업로드 | 모델 성능 저하 | 검증 시스템 (loss 체크, 가중치 이상 탐지) |
| 서버 단일 장애점 (SPOF) | 전체 시스템 중단 | Supabase 자동 재시작, DB 백업 (pg_dump), 향후 HA 구성 |
| 네트워크 대역폭 부족 | 체크포인트 전송 느림 | Delta 전송, 압축, CDN |
| 수렴 불안정 (너무 많은 비동기 기여) | 학습 품질 저하 | merge_threshold 조절, stale gap 제한 |
| 데이터 프라이버시 | 학습 데이터 유출 | 데이터는 서버에서만 배포, 워커는 학습만 |
| 모델 가중치 유출 | 모델 도용 | API 키 인증, 필요 시 가중치 암호화 |

---

## 15. 성공 지표

| 지표 | 목표값 |
|------|--------|
| 워커 참여/이탈이 다른 워커에 영향 없이 작동 | 100% |
| GPU/CPU 혼합 환경에서 학습 진행 | 정상 작동 |
| 10대 동시 참여 시 안정적 학습 | loss가 단조 감소 |
| 장시간 학습 (24시간+) 안정성 | 크래시 없음 |
| 체크포인트 병합 후 val_loss 개선 | 단독 학습 대비 동등 이상 |
| 워커 CLI 실행에서 학습 시작까지 | 2분 이내 |

---

## 부록 A: 기여도 쿼리 모음

```sql
-- 팀원별 총 기여 통계
SELECT
    w.name                                AS 팀원,
    w.device_name                         AS 장비,
    COUNT(c.id)                           AS 참여횟수,
    SUM(c.steps_trained)                  AS 총_학습_step,
    ROUND(AVG(c.local_train_loss)::numeric, 4) AS 평균_loss,
    ROUND(SUM(c.training_duration_s)/3600.0, 1) AS 총_학습시간_h,
    MAX(c.submitted_at)                   AS 마지막_참여
FROM contributions c
JOIN workers w ON c.worker_id = w.id
WHERE c.status = 'merged'
GROUP BY w.id, w.name, w.device_name
ORDER BY 총_학습_step DESC;

-- 일별 학습 진행 요약
SELECT
    DATE(created_at)                      AS 날짜,
    MAX(global_step)                      AS 최종_step,
    MIN(train_loss)                       AS 최저_train_loss,
    MIN(val_loss)                         AS 최저_val_loss,
    SUM(num_contributors)                 AS 총_기여자수
FROM checkpoints
GROUP BY DATE(created_at)
ORDER BY 날짜;

-- 시간대별 활성 워커 수 (최근 24시간)
SELECT
    DATE_TRUNC('hour', submitted_at)      AS 시간대,
    COUNT(DISTINCT worker_id)             AS 활성_워커수,
    SUM(steps_trained)                    AS 시간당_step
FROM contributions
WHERE submitted_at > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', submitted_at)
ORDER BY 시간대;
```

---

## 부록 B: 인프라 구성

### B.1 Supabase 접속 설정 (.environments 활용)

```
기존에 운영 중인 Supabase 인스턴스를 사용합니다.
PostgreSQL을 별도로 설치/운영할 필요 없음.

.environments 파일에서 사용하는 핵심 변수:
  SUPABASE_HOST       → Supabase 도메인 (API 서버 접속 URL 구성)
  POSTGRES_PASSWORD    → PostgreSQL 접속 비밀번호
  POSTGRES_HOST        → PostgreSQL 호스트 (Docker 내부: db, 외부: SUPABASE_HOST)
  POSTGRES_DB          → 데이터베이스 이름 (postgres)
  POSTGRES_PORT        → PostgreSQL 포트 (5432)
  ANON_KEY             → Supabase Anonymous 키 (Storage API 접근 시)
  SERVICE_ROLE_KEY     → Supabase Service Role 키 (관리 작업 시)
```

### B.2 config.py에서 .environments 로드 방식

```python
# distributed/server/config.py

import os
from pathlib import Path

def load_environments(env_path: str = ".environments") -> dict:
    """
    .environments 파일에서 KEY=VALUE 형태의 설정을 로드합니다.
    Supabase 접속 정보, PostgreSQL 비밀번호 등이 포함됩니다.
    """
    env_vars = {}
    env_file = Path(env_path)
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()
    return env_vars

# .environments에서 설정 로드
_env = load_environments()

# PostgreSQL 접속 URL 구성
# Supabase 외부 접속 시: SUPABASE_HOST 사용
# Supabase Docker 내부 접속 시: POSTGRES_HOST (db) 사용
SUPABASE_HOST = _env.get("SUPABASE_HOST", "localhost")
POSTGRES_PASSWORD = _env.get("POSTGRES_PASSWORD", "")
POSTGRES_DB = _env.get("POSTGRES_DB", "postgres")
POSTGRES_PORT = int(_env.get("POSTGRES_PORT", "5432"))
POOLER_PORT = int(_env.get("POOLER_PROXY_PORT_TRANSACTION", "6543"))

# 외부 접속 시 연결 풀러(Supavisor) 포트 사용 권장
DATABASE_URL = (
    f"postgresql+asyncpg://postgres.{SUPABASE_HOST}:{POSTGRES_PASSWORD}"
    f"@{SUPABASE_HOST}:{POOLER_PORT}/{POSTGRES_DB}"
)

# Supabase Storage 접근 키
SUPABASE_URL = f"https://{SUPABASE_HOST}"
SUPABASE_ANON_KEY = _env.get("ANON_KEY", "")
SUPABASE_SERVICE_KEY = _env.get("SERVICE_ROLE_KEY", "")
```

### B.3 docker-compose.yml (Coordinator만, DB는 Supabase 사용)

```yaml
version: "3.9"

services:
  coordinator:
    build:
      context: ..
      dockerfile: docker/Dockerfile.server
    ports:
      - "8000:8000"
    env_file:
      - ../.environments
    environment:
      # .environments의 값을 자동으로 사용
      STORAGE_PATH: /app/storage
    volumes:
      - storage:/app/storage
    restart: unless-stopped

volumes:
  storage:

# 참고: PostgreSQL은 Supabase에서 제공하므로 별도 서비스 불필요
```

### B.4 Supabase Storage 활용 (체크포인트 파일)

```
Supabase Storage 구성:
  버킷: "checkpoints"    → 모델 체크포인트 파일 (.pt)
  버킷: "datasets"       → train.bin, val.bin, tokenizer.json

접근 방식:
  서버 (Coordinator): SERVICE_ROLE_KEY로 직접 업로드/다운로드
  워커 (클라이언트):   Coordinator API를 경유하여 다운로드
                      (워커가 직접 Supabase에 접근하지 않음)

파일 경로 예시:
  checkpoints/exp_1/ckpt_r0.pt       → 초기 체크포인트
  checkpoints/exp_1/ckpt_r21.pt      → 21번째 병합 체크포인트
  datasets/exp_1/train.bin           → 학습 데이터
  datasets/exp_1/val.bin             → 검증 데이터
  datasets/exp_1/tokenizer.json      → 토크나이저
```

---

## 부록 C: 용어 정리

| 용어 | 설명 |
|------|------|
| **Coordinator** | 중앙 서버. 워커 관리, 체크포인트 저장, 병합 수행 |
| **Worker** | 팀원의 컴퓨터. 로컬에서 학습 수행 후 결과 업로드 |
| **Round** | 하나의 병합 주기. 여러 워커의 기여를 모아 병합하는 단위 |
| **Contribution** | 워커가 제출한 학습 결과 (가중치 + 메타데이터) |
| **FedAvg** | Federated Averaging. 여러 모델의 가중 평균으로 병합 |
| **Stale Gap** | 워커의 기반 step과 현재 글로벌 step의 차이 |
| **Global Step** | 병합을 통해 누적된 전체 학습 step 수 |
| **Local Step** | 워커가 한 라운드에서 수행하는 학습 step 수 |
| **Trust Score** | 워커의 신뢰도 점수 (0.0~1.0). 이상 기여 시 감소 |
| **Delta 전송** | 전체 가중치 대신 변화분만 전송하는 최적화 기법 |
| **Checkpoint** | 특정 시점의 모델 가중치 + 메타데이터를 저장한 파일 |
| **Experiment** | 하나의 학습 실험 단위. 모델 설정 + 학습 이력 포함 |
| **API Credit** | 학습 기여로 적립되는 API 사용 토큰. 1 학습 토큰 = 1 API 토큰 |
| **Earned Tokens** | 워커가 학습 기여로 적립한 총 토큰 수 |
| **Remaining Credits** | earned_tokens - used_tokens. API 호출 가능한 잔여 토큰 |

---

## 16. 기여 보상 시스템 — API 크레딧

### 16.1 핵심 원칙

```
학습에 기여한 만큼, 학습된 모델을 무료로 사용할 수 있다.

  기여 (학습)  ────→  적립 (토큰 크레딧)  ────→  사용 (API 호출)

  1 학습 토큰 적립 = 1 API 토큰 사용 가능
```

| 원칙 | 설명 |
|------|------|
| **1:1 등가** | 학습에 기여한 토큰 수만큼 API 토큰을 사용 가능 |
| **누적 적립** | 크레딧은 사라지지 않음. 계속 쌓임 |
| **투명한 추적** | 모든 적립/차감이 트랜잭션으로 기록됨 |
| **워커 프로필 기반** | API 키는 워커(참여자) 프로필에 연결됨 |

### 16.2 크레딧 적립 흐름

```
워커가 50 step 로컬 학습 완료
        │
        ▼
서버에 기여(contribution) 제출
        │
        ▼
서버가 검증 후 병합 (status = 'merged')
        │
        ▼
적립량 계산:
  학습된 토큰 수 = steps_trained × batch_size × block_size
                 = 50 × 16 × 256
                 = 204,800 토큰
        │
        ▼
워커의 earned_tokens에 204,800 적립
        │
        ▼
token_transactions에 기록:
  { type: 'earn', amount: 204800, description: "50 step 학습 기여" }
```

### 16.3 크레딧 적립 계산식

```
적립 토큰 수 = steps_trained × batch_size × block_size

예시 (현재 FAI 설정 기준):
  ┌────────────────┬──────────┬──────────┬──────────┬──────────────┐
  │ 시나리오        │ steps    │ batch    │ block    │ 적립 토큰     │
  ├────────────────┼──────────┼──────────┼──────────┼──────────────┤
  │ 1라운드 (CPU)   │ 25       │ 4        │ 256      │ 25,600       │
  │ 1라운드 (MPS)   │ 50       │ 16       │ 256      │ 204,800      │
  │ 1라운드 (GPU)   │ 100      │ 64       │ 256      │ 1,638,400    │
  │ 하루 10라운드    │ 500      │ 16       │ 256      │ 2,048,000    │
  │ 1주일 매일 참여  │ 3,500    │ 16       │ 256      │ 14,336,000   │
  └────────────────┴──────────┴──────────┴──────────┴──────────────┘

보너스 적립 (선택적):
  - 첫 참여 보너스:           +10,000 토큰
  - 연속 7일 참여 보너스:      +적립량의 10%
  - 상위 기여자 (월간 Top 10): +적립량의 20%
```

### 16.4 크레딧 차감 흐름

```
사용자가 API 호출: POST /v1/completions
        │
        ▼
┌──────────────────────────────────┐
│ 1. API 키 검증 + 잔액 확인        │
│ (하나의 Supabase PostgreSQL 쿼리)  │
│                                  │
│    SELECT ak.*, w.is_banned,     │
│      (earned_tokens - used_tokens)│
│      AS remaining                │
│    FROM api_keys ak              │
│    JOIN workers w ON ...          │
│    WHERE ak.api_key = $1         │
│      AND ak.is_active = TRUE     │
│                                  │
│    if remaining < estimated_tokens:
│      → 403 Insufficient Credits  │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 2. LLM 추론 실행                  │
│    모델에 프롬프트 전달 → 텍스트 생성
│    실제 사용 토큰 계산:            │
│      prompt_tokens + completion_tokens
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 3. 크레딧 차감 (원자적 Supabase PG) │
│                                  │
│    UPDATE api_keys               │
│    SET used_tokens = used_tokens  │
│        + $total_tokens           │
│    WHERE api_key = $1            │
│      AND (earned_tokens          │
│          - used_tokens)          │
│          >= $total_tokens;       │
│                                  │
│    + INSERT INTO api_usage_log   │
│    + INSERT INTO token_transactions│
│    (하나의 트랜잭션으로 묶음)       │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 5. 응답 반환                      │
│    { text, total_tokens,         │
│      remaining_credits }         │
└──────────────────────────────────┘
```

### 16.5 API 키 발급 규칙

```
API 키 발급 조건:
  - 워커가 최소 1회 이상 기여(contribution)가 병합(merged)된 상태
  - is_banned = FALSE
  - trust_score >= 0.5

API 키 특성:
  - 워커 1명당 최대 5개 키 발급 가능
  - 키별로 독립적인 rate limit 설정 가능
  - 크레딧은 워커 단위로 공유 (키가 여러 개여도 잔액은 하나)

  예시:
    철수 (earned: 500,000 / used: 120,000 / remaining: 380,000)
      ├── API Key A: "내 앱용"     → 같은 잔액 380,000 공유
      ├── API Key B: "테스트용"    → 같은 잔액 380,000 공유
      └── API Key C: "친구에게 공유" → 같은 잔액 380,000 공유
```

### 16.6 Rate Limiting

```
계층별 제한:

  ┌─────────────────┬──────────────┬───────────────┬──────────────┐
  │ 등급             │ 조건          │ 분당 요청 수   │ 요청당 토큰   │
  ├─────────────────┼──────────────┼───────────────┼──────────────┤
  │ 일반 참여자       │ 1회+ 기여     │ 30            │ 256          │
  │ 활성 참여자       │ 100+ step    │ 60            │ 512          │
  │ 핵심 기여자       │ 10,000+ step │ 120           │ 1024         │
  │ 최고 기여자       │ 100,000+ step│ 300           │ 2048         │
  └─────────────────┴──────────────┴───────────────┴──────────────┘

  Rate limit 구현 (앱 메모리 + Supabase PostgreSQL):
    앱 메모리: collections.defaultdict로 sliding window 카운터
      → 요청마다 카운트 증가, 1분 경과 시 리셋
      → 초과 시 429 Too Many Requests 반환
    Supabase PostgreSQL: 서버 재시작 시 api_usage_log에서 최근 1분 카운트 복구
      → SELECT COUNT(*) FROM api_usage_log
         WHERE api_key_id = $1 AND created_at > NOW() - INTERVAL '1 minute'
```

### 16.7 워커 CLI에서 크레딧 확인

```
# 워커 실행 화면에 크레딧 정보 표시
╔══════════════════════════════════════════════════════╗
║  FAI 분산 학습 워커 v1.0                              ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  [라운드 5/∞] 글로벌 step: 1250                       ║
║  ├─ 로컬 학습: ██████████████████░░  45/50 step      ║
║  │  └─ loss: 2.05 → 1.98 (↓0.07)                    ║
║                                                      ║
║  💰 내 크레딧:                                        ║
║  ├─ 총 적립:    1,024,000 토큰                        ║
║  ├─ 사용:        52,300 토큰                          ║
║  ├─ 잔여:       971,700 토큰                          ║
║  └─ 이번 라운드 예상 적립: +204,800 토큰               ║
║                                                      ║
╚══════════════════════════════════════════════════════╝

# 크레딧만 확인하는 명령어
$ python -m distributed.worker --credits --server https://fai.example.com
╔══════════════════════════════════════════╗
║  💰 철수의 맥북 — 크레딧 현황            ║
╠══════════════════════════════════════════╣
║  총 적립:     1,024,000 토큰             ║
║  사용:           52,300 토큰             ║
║  잔여:          971,700 토큰             ║
║                                          ║
║  API 키: 2개 활성                        ║
║  ├─ sk-abc...xyz (내 앱용)               ║
║  └─ sk-def...uvw (테스트용)              ║
║                                          ║
║  등급: 핵심 기여자 (120 req/min)          ║
║  총 기여: 250 라운드, 12,500 step        ║
╚══════════════════════════════════════════╝
```

### 16.8 크레딧 보호 및 어뷰징 방지

```
문제: 악의적 사용자가 쓸모없는 학습을 해서 크레딧만 쌓으려 할 수 있음

방지 전략:

  1. 병합 성공 시에만 적립
     - status = 'merged'인 기여만 크레딧 적립
     - 거부된(rejected) 기여는 적립 없음

  2. Loss 기반 품질 가중치
     - 학습 후 loss가 개선되지 않은 기여: 적립량 50% 감소
     - loss가 악화된 기여: 적립 없음

  3. Trust Score 연동
     - trust_score < 0.5인 워커: API 키 발급 불가
     - trust_score < 0.7인 워커: 적립량 50% 감소

  4. 일일 적립 상한
     - 워커당 하루 최대 적립: 10,000,000 토큰
     - 비정상적으로 빠른 적립 패턴 감지 시 검토
```

### 16.9 기여도 대시보드 쿼리

```sql
-- 워커별 크레딧 현황
SELECT
    w.name                              AS 팀원,
    ak.earned_tokens                    AS 총_적립,
    ak.used_tokens                      AS 총_사용,
    (ak.earned_tokens - ak.used_tokens) AS 잔여_크레딧,
    w.total_steps_trained               AS 총_학습_step,
    COUNT(DISTINCT ak2.id)              AS API키_수
FROM workers w
LEFT JOIN api_keys ak ON w.id = ak.worker_id AND ak.is_active = TRUE
LEFT JOIN api_keys ak2 ON w.id = ak2.worker_id AND ak2.is_active = TRUE
GROUP BY w.id, w.name, ak.earned_tokens, ak.used_tokens, w.total_steps_trained
ORDER BY 잔여_크레딧 DESC;

-- 일별 토큰 적립/사용 추이
SELECT
    DATE(created_at)                    AS 날짜,
    SUM(CASE WHEN type = 'earn' THEN amount ELSE 0 END)  AS 적립_토큰,
    SUM(CASE WHEN type = 'spend' THEN ABS(amount) ELSE 0 END) AS 사용_토큰
FROM token_transactions
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY 날짜;
```

---

## 17. CoT/ToT 아키텍처 분석

본 섹션은 Chain of Thought(CoT)와 Tree of Thought(ToT) 분석 기법을 사용하여
전체 분산 학습 시스템의 아키텍처 결정을 체계적으로 검증합니다.

### 17.1 CoT 분석 — 순차적 의사결정 추적

#### 결정 1: 데이터베이스 아키텍처 (Redis + PostgreSQL → PostgreSQL 단독 → Supabase)

```
사고 흐름:

1단계: 초기 설계 — Redis + PostgreSQL 이중 구조
  → Redis: heartbeat, 작업 큐, Pub/Sub, 분산 락
  → PostgreSQL: 영속 데이터 (워커, 기여도, 체크포인트)
  → 문제: 두 시스템 간 데이터 동기화 복잡성, 운영 부담 증가

2단계: Redis 필요성 재평가
  → PostgreSQL 자체 기능으로 Redis 역할 100% 대체 가능:
     - SKIP LOCKED → 작업 큐 (Redis RPOP 대체)
     - pg_advisory_lock() → 분산 락 (Redis SETNX 대체)
     - LISTEN/NOTIFY → Pub/Sub (Redis PUBLISH/SUBSCRIBE 대체)
     - last_seen 컬럼 → heartbeat (Redis SETEX 대체)
  → 결론: 소~중규모(1,000대 이하)에서 Redis 불필요

3단계: Supabase 통합
  → 이미 운영 중인 Supabase 인스턴스 활용 (.environments)
  → 추가 이점:
     - Supabase Storage → 체크포인트/데이터셋 저장 (MinIO 불필요)
     - Supabase Auth → 향후 워커 인증 확장 가능
     - Supavisor → 커넥션 풀링 내장 (PgBouncer 불필요)
     - 단일 인프라로 DB + 스토리지 + 인증 통합
  → 결론: 인프라 복잡성 최소화, 운영 비용 절감

최종 판단: ✅ Supabase PostgreSQL 단독 구성이 최적
  - 장점: 운영 단순화, 기존 인프라 재활용, 기능 충분
  - 한계: 10,000대+ 시 LISTEN/NOTIFY 성능 → RabbitMQ/Kafka 보완
```

#### 결정 2: 학습 병합 전략 (동기 → 비동기 FedAvg)

```
사고 흐름:

1단계: 동기식 분산 학습 (PyTorch DDP) 고려
  → 장점: 학습 품질 최고, 구현 단순
  → 문제: 모든 워커가 동시 온라인 필수 → "자유 참여/이탈" 요구사항 위배
  → 결론: ❌ 부적합

2단계: 비동기 FedAvg 선택
  → 워커가 독립적으로 N 스텝 학습 → 가중치 업로드 → 서버에서 병합
  → 장점: 자유 참여/이탈 완벽 지원, 하드웨어 이질성 허용
  → 도전: stale contribution 처리 필요
  → 해결: gap 기반 수용/거부 + 가중치 가중 평균
  → 결론: ✅ 프로젝트 요구사항에 완벽 부합

3단계: 병합 트리거 전략
  → 시간 기반: 일정 시간마다 병합 → 워커 적을 때 비효율
  → 카운트 기반: N개 기여 도착 시 병합 → 예측 가능
  → 하이브리드: MIN(카운트, 타임아웃) 중 먼저 도달 시
  → 결론: ✅ 하이브리드 (merge_threshold OR merge_timeout)

최종 판단: ✅ 비동기 FedAvg + 하이브리드 트리거
```

#### 결정 3: 기여 보상 시스템 (API 크레딧)

```
사고 흐름:

1단계: 보상 필요성
  → 자발적 참여 동기 부여 → 학습 기여 = API 사용권
  → 비금전적 인센티브로 지속 참여 유도
  → 공정성: 기여량에 비례한 보상 (1 학습 토큰 = 1 API 토큰)

2단계: 구현 방식
  → 옵션 A: 별도 마이크로서비스 → 과도한 복잡성
  → 옵션 B: PostgreSQL 트랜잭션 기반 → 원자적 처리, 단순
  → 결론: ✅ 옵션 B (earned_tokens/used_tokens 컬럼 + 원자적 UPDATE)

3단계: 부정 방지
  → 문제: 가짜 기여로 크레딧 획득 시도
  → 대응: loss 검증 + 가중치 이상 탐지 + trust_score 시스템
  → loss가 전역 평균보다 50% 이상 높으면 기여 거부 = 크레딧 미지급

최종 판단: ✅ PostgreSQL 원자적 트랜잭션 기반 크레딧 시스템
```

### 17.2 ToT 분석 — 대안 분기 평가

#### 분기 1: 인프라 아키텍처

```
                    인프라 선택
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     [A] 독립 구성   [B] Supabase  [C] 클라우드
      PostgreSQL      통합 구성     매니지드 서비스
      + MinIO         (현재 선택)    (AWS RDS +S3)
      + PgBouncer
          │            │            │
     운영 부담 높음    운영 부담 낮음   비용 높음
     컴포넌트 3개+     단일 플랫폼     벤더 종속
     직접 백업 필요    스토리지 포함    스케일링 용이
          │            │            │
     점수: 5/10      점수: 9/10     점수: 6/10
```

**선택: [B] Supabase 통합 구성**

| 평가 기준 | [A] 독립 구성 | [B] Supabase (선택) | [C] 클라우드 매니지드 |
|-----------|:---:|:---:|:---:|
| 운영 복잡성 | 높음 | **낮음** | 중간 |
| 비용 | 중간 | **낮음 (기존 인프라)** | 높음 |
| 스케일링 유연성 | 높음 | 중간 | **높음** |
| 초기 설정 시간 | 높음 | **최소** | 중간 |
| 기존 인프라 활용 | 불가 | **100%** | 불가 |
| Storage 통합 | 별도 설정 | **내장** | 별도 설정 |

#### 분기 2: 학습 병합 전략

```
                    병합 전략
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   [A] 동기식       [B] 비동기       [C] Gossip
    All-Reduce      FedAvg           Protocol
   (PyTorch DDP)   (현재 선택)       (P2P 분산)
        │              │              │
   모두 동시 온라인   자유 참여/이탈    서버 불필요
   동질 하드웨어     이질 하드웨어     수렴 불확실
   최고 품질        양호 품질         낮은 품질
        │              │              │
   점수: 3/10       점수: 9/10       점수: 4/10
```

**선택: [B] 비동기 FedAvg**

| 평가 기준 | [A] 동기식 | [B] FedAvg (선택) | [C] Gossip |
|-----------|:---:|:---:|:---:|
| 자유 참여/이탈 | ❌ | **✅** | ✅ |
| GPU/CPU 혼합 | ❌ | **✅** | ✅ |
| 학습 품질 | 최고 | **양호** | 불확실 |
| 구현 복잡성 | 낮음 | **중간** | 높음 |
| 중앙 서버 필요 | 아니오 | **예** | 아니오 |
| 수만 대 확장 | 어려움 | **가능** | 이론적 가능 |

#### 분기 3: 워커 인증 및 보안

```
                    워커 인증
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     [A] API 키만   [B] API 키 +   [C] Supabase
       (현재 선택)    Trust Score    Auth 통합
          │            │            │
     단순/빠른 구현   중간 복잡성     높은 보안
     기본 보안       악의적 워커 탐지  OAuth/SSO
          │            │            │
     Phase 1 적합   Phase 2 확장    Phase 3 확장
```

**현재 선택: [A] Phase 1에서 API 키 → Phase 2에서 [B] Trust Score 추가 → Phase 3에서 [C] Supabase Auth 확장**

### 17.3 종합 평가 — 아키텍처 리스크 매트릭스

| 결정 사항 | 선택 | 리스크 | 완화 방안 | 신뢰도 |
|-----------|------|--------|-----------|--------|
| Supabase 단독 인프라 | ✅ 채택 | 대규모 시 NOTIFY 병목 | RabbitMQ 보완 경로 확보 | 높음 |
| 비동기 FedAvg | ✅ 채택 | stale 기여로 수렴 불안정 | gap 제한 + 검증 시스템 | 높음 |
| PostgreSQL 작업 큐 | ✅ 채택 | 고빈도 폴링 시 DB 부하 | LISTEN/NOTIFY + 적절한 interval | 중간 |
| 1:1 크레딧 비율 | ✅ 채택 | 부정 기여 시 크레딧 남발 | trust_score + loss 검증 | 높음 |
| Delta 전송 | ✅ 채택 | 정밀도 손실 가능 | float16 + 검증 후 적용 | 중간 |
| Supabase Storage | ✅ 채택 | 대용량 체크포인트 전송 속도 | CDN 도입 경로 (Phase 3+) | 중간 |

### 17.4 결론 및 권장 사항

**현재 아키텍처의 종합 적합도: 8.5/10**

```
강점:
  ✅ Supabase 통합으로 인프라 단순화 (DB + Storage + Auth = 단일 플랫폼)
  ✅ 비동기 FedAvg로 "자유 참여/이탈" 완벽 지원
  ✅ PostgreSQL 내장 기능만으로 모든 실시간 요구사항 충족
  ✅ 크레딧 시스템으로 자발적 참여 동기 부여
  ✅ 단계별 구현 (Phase 1~4)으로 점진적 복잡성 증가

주의점:
  ⚠️ 1,000대 이상 스케일 시 LISTEN/NOTIFY → 메시지 큐 전환 계획 필요
  ⚠️ Supabase Storage 대용량 파일 전송 성능 모니터링 필요
  ⚠️ 비동기 FedAvg의 수렴 품질은 실제 실험으로 검증 필요

권장 우선순위:
  1순위: Phase 1 구현 (기본 인프라 + 단일 워커 학습)
  2순위: Phase 2 구현 (다중 워커 + 병합)
  3순위: 실제 10대 테스트로 수렴 품질 검증
  4순위: 크레딧 시스템 및 API 서비스 구현
```
