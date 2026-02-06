# FAI 분산 학습 시스템 아키텍처

## 개요

자발적 참여자들이 자신의 GPU/CPU를 제공하여 FAI GPT 모델을 협업으로 학습하는 시스템.
Federated Averaging(FedAvg) 기반 비동기 분산 학습 아키텍처.

### 핵심 개념

| 개념 | 설명 |
|------|------|
| **Coordinator** | 중앙 서버. 작업 배분, 체크포인트 관리, FedAvg 병합 수행 |
| **Worker** | 참여자 클라이언트. 서버에서 모델을 받아 로컬 N 스텝 학습 후 가중치 업로드 |
| **FedAvg** | 가중 평균 병합. `global = Σ(w_i × local_i) / Σ(w_i)` |
| **Stale Gap** | 기여의 신선도. gap 0-50: 수용, 51-200: 감쇠, 201+: 거절 |
| **크레딧 시스템** | 1 학습 스텝 = 1 API 토큰 크레딧 |

### 학습 흐름

```
Worker                         Coordinator
  │                                │
  ├─── register ─────────────────►│ 워커 등록 + 하드웨어 정보
  │◄── worker_uid + 추천설정 ─────┤
  │                                │
  ├─── request_task ─────────────►│ 작업 할당 + 체크포인트 URL
  │◄── task_id + checkpoint_url ──┤
  │                                │
  ├─── download_checkpoint ──────►│ 글로벌 모델 다운로드
  │◄── .pt 파일 ──────────────────┤
  │                                │
  │  [로컬 N 스텝 학습]            │
  │  (heartbeat 30초마다)          │
  │                                │
  ├─── complete_task ────────────►│ 가중치 업로드 + 결과 보고
  │◄── ok ────────────────────────┤
  │                                │
  │                         [FedAvg 병합 루프]
  │                         (pending ≥ 3개 시)
```

---

## 파일 구조 및 핵심 코드

### 전체 구조 (33개 파일)

```
distributed/
├── __init__.py
├── common/                          # 서버+워커 공통 모듈
│   ├── __init__.py
│   ├── model.py                     # ★ GPT 모델 (train_gpt.py에서 추출)
│   ├── constants.py                 # 상태 상수, 기본값
│   ├── serialization.py             # 가중치 직렬화 + gzip + delta 전송
│   └── protocol.py                  # API 엔드포인트, 헤더
├── server/                          # Coordinator 서버
│   ├── __init__.py
│   ├── config.py                    # .environments → ServerConfig
│   ├── database.py                  # SQLAlchemy async 엔진
│   ├── models.py                    # ORM 9개 테이블
│   ├── schemas.py                   # Pydantic v2 스키마
│   ├── app.py                       # ★ FastAPI + lifespan
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── workers.py               # 5개 엔드포인트
│   │   ├── tasks.py                 # 3개 엔드포인트
│   │   ├── checkpoints.py           # 3개 엔드포인트
│   │   └── metrics.py               # 4개 엔드포인트
│   └── services/
│       ├── __init__.py
│       ├── heartbeat.py             # 워커 생존 감시
│       ├── validator.py             # 기여 검증
│       ├── merger.py                # ★ FedAvg 병합 엔진
│       └── scheduler.py             # GPU 기반 작업 스케줄링
└── worker/                          # 학습 워커
    ├── __init__.py
    ├── __main__.py                  # python -m distributed.worker
    ├── config.py                    # WorkerConfig
    ├── device_manager.py            # CUDA > MPS > CPU 자동 감지
    ├── client.py                    # httpx 서버 통신
    ├── checkpoint_io.py             # 체크포인트 다운/업/캐시
    ├── trainer.py                   # ★ 로컬 N 스텝 학습 루프
    └── cli.py                       # Click CLI 진입점

docker/
├── Dockerfile
└── docker-compose.yml
```

---

## 핵심 모듈 상세

### 1. common/model.py — GPT 모델

`scripts/train_gpt.py`에서 추출한 GPT 모델. 기존 체크포인트와 state_dict 키 100% 호환.

```python
@dataclass
class GPTConfig:
    vocab_size: int = 24000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

    def to_dict(self) -> dict: ...     # JSONB 저장용
    @classmethod
    def from_dict(cls, d: dict): ...   # JSONB 복원용

class GPT(nn.Module):
    # 구조: tok_emb → pos_emb → blocks(N) → ln_f → head
    # state_dict 키: tok_emb, pos_emb, blocks.0.ln1, blocks.0.attn.qkv, ...
    @classmethod
    def from_checkpoint(cls, path): ...  # .pt 파일에서 모델 로드
```

**주의**: 레이어 이름(tok_emb, pos_emb, blocks, ln_f, head)을 변경하면 기존 체크포인트 호환이 깨짐.

### 2. common/constants.py — 상태 상수

```python
class WorkerStatus:
    OFFLINE = "offline"
    ONLINE = "online"
    TRAINING = "training"
    UPLOADING = "uploading"

class ContributionStatus:
    PENDING = "pending"
    VALIDATING = "validating"
    MERGED = "merged"
    REJECTED = "rejected"

class Defaults:
    MERGE_THRESHOLD = 3          # 병합 최소 기여 수
    MAX_STALE_GAP = 200          # stale gap 최대값
    HEARTBEAT_INTERVAL_SEC = 30
    OFFLINE_TIMEOUT_SEC = 60
```

### 3. common/protocol.py — API 경로

```python
API_PREFIX = "/api/v1"
WORKER_KEY_HEADER = "X-Worker-Key"

class Endpoints:
    WORKERS_REGISTER = f"{API_PREFIX}/workers/register"
    WORKERS_HEARTBEAT = f"{API_PREFIX}/workers/heartbeat"
    TASKS_REQUEST = f"{API_PREFIX}/tasks/request"
    CHECKPOINTS_LATEST = f"{API_PREFIX}/checkpoints/latest"

    @staticmethod
    def tasks_complete(task_id: int) -> str:
        return f"{API_PREFIX}/tasks/{task_id}/complete"
```

### 4. server/models.py — DB 테이블 (9개)

| 테이블 | 역할 | 핵심 컬럼 |
|--------|------|----------|
| `workers` | 워커 관리 | worker_uid(UUID), device_type, status, trust_score, last_seen |
| `experiments` | 실험 관리 | config(JSONB/GPTConfig), current_global_step, best_val_loss |
| `checkpoints` | 체크포인트 | file_path, merged_from(JSONB), is_latest, is_best |
| `contributions` | 학습 결과 | base_global_step, steps_trained, stale_gap, merge_weight, status |
| `training_metrics` | 학습 곡선 | global_step, train_loss, val_loss |
| `audit_log` | 감사 로그 | event_type, details(JSONB) |
| `api_keys` | API 크레딧 | earned_tokens, used_tokens, rate_limit_rpm |
| `api_usage_log` | API 사용 | prompt_tokens, completion_tokens |
| `token_transactions` | 거래 기록 | type(earn/spend), amount, balance_after |

### 5. server/services/merger.py — FedAvg 병합 엔진

```python
def fedavg_merge(base_state_dict, contributions):
    """
    global_weights = Σ (w_i * local_weights_i) / Σ w_i
    w_i = merge_weight × steps_trained
    """
    total_weight = sum(w for _, w in contributions)
    merged = OrderedDict()
    for key in base_state_dict:
        weighted_sum = torch.zeros_like(base_state_dict[key], dtype=torch.float32)
        for state_dict, weight in contributions:
            weighted_sum += state_dict[key].float() * (weight / total_weight)
        merged[key] = weighted_sum.to(base_state_dict[key].dtype)
    return merged

class MergeLoop:
    # 30초마다 실행, pending ≥ 3개면 병합
    # pg_advisory_lock(42)로 동시 병합 방지
    # 검증 → FedAvg → 새 체크포인트 저장 → 크레딧 적립
```

### 6. server/services/validator.py — 기여 검증

```python
class ContributionValidator:
    # 4단계 검증:
    # 1. NaN/Inf 체크 — torch.isnan/isinf
    # 2. Loss 이상 탐지 — local > global × 3.0 이면 거절
    # 3. 가중치 크기 체크 — L2 노름 이상
    # 4. Stale gap 감쇠:
    #    - gap 0~50: weight=1.0
    #    - gap 51~200: weight = 1.0 - (gap-50)/150 (선형 감쇠)
    #    - gap 201+: 거절
    # 최종: merge_weight = stale_weight × trust_score
```

### 7. worker/trainer.py — 로컬 학습

```python
class LocalTrainer:
    def _get_batch(self, split, batch_size):
        # numpy memmap에서 랜덤 배치 생성
        # train_gpt.py의 get_batch()와 동일 로직
        data = self._train_data if split == "train" else self._val_data
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    async def train(self, model, local_steps, batch_size, ...):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
        for step in range(local_steps):
            x, y = self._get_batch("train", batch_size)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # heartbeat 콜백 (10스텝마다)
        val_loss = self._estimate_val_loss(model, batch_size)
        return TrainResult(steps_trained, train_loss, val_loss, duration_s, ...)
```

### 8. worker/cli.py — CLI 진입점

```bash
# 워커 실행
uv run python -m distributed.worker \
    --name "철수의 맥북" \
    --server http://coordinator:8000 \
    --experiment 1 \
    --max-rounds 10

# 서버 실행
uv run uvicorn distributed.server.app:app --host 0.0.0.0 --port 8000
```

---

## 서버 설정 및 DB 접속

### .environments 파일 (Supabase PostgreSQL)

```python
# server/config.py
class ServerConfig:
    supabase_host: str          # fai-supabase-...traefik.me
    postgres_password: str
    postgres_port: int = 5432   # 직접 접속 (Supavisor X)

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://postgres:{self.postgres_password}@{self.supabase_host}:{self.postgres_port}/postgres"
```

### PostgreSQL 패턴 (Redis 대체)

| Redis 기능 | PostgreSQL 대체 |
|-----------|----------------|
| 분산 락 | `pg_advisory_lock(42)` |
| 작업 큐 | `FOR UPDATE SKIP LOCKED` |
| Pub/Sub | `LISTEN/NOTIFY` |
| TTL 키 | `last_seen` 컬럼 + 주기적 확인 |

---

## API 엔드포인트 (15개)

### 워커 관리 (5개)
| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | /api/v1/workers/register | 워커 등록 |
| POST | /api/v1/workers/heartbeat | 생존 신호 |
| POST | /api/v1/workers/benchmark | 벤치마크 결과 |
| GET | /api/v1/workers/me | 내 정보 조회 |
| POST | /api/v1/workers/leave | 이탈 통보 |

### 작업 관리 (3개)
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | /api/v1/experiments/{id}/status | 실험 상태 |
| POST | /api/v1/tasks/request | 작업 요청 |
| POST | /api/v1/tasks/{id}/complete | 완료 보고 (multipart) |

### 체크포인트 (3개)
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | /api/v1/checkpoints/latest | 최신 다운로드 |
| GET | /api/v1/checkpoints/{id}/download | 특정 다운로드 |
| GET | /api/v1/checkpoints/history | 히스토리 |

### 메트릭 (4개)
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | /api/v1/metrics/summary | 전체 요약 |
| GET | /api/v1/metrics/loss-history | Loss 추이 |
| GET | /api/v1/metrics/leaderboard | 리더보드 |
| GET | /api/v1/metrics/workers | 활성 워커 |

---

## Docker 구성

```yaml
# docker/docker-compose.yml
services:
  coordinator:
    command: uvicorn distributed.server.app:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    volumes:
      - ../.environments:/app/.environments:ro
      - fai-storage:/app/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]

  worker:
    command: python -m distributed.worker --name "docker-worker" --server http://coordinator:8000
    volumes:
      - ../data:/app/data:ro
    depends_on:
      coordinator: { condition: service_healthy }
    deploy:
      resources: { limits: { memory: 4G } }
```

```bash
# 서버만 실행
docker compose up coordinator

# 서버 + 워커 3대
docker compose up --scale worker=3
```

---

## 의존성

pyproject.toml에 추가된 분산 학습 전용 패키지:

| 패키지 | 용도 |
|--------|------|
| fastapi>=0.115.0 | Coordinator REST API |
| uvicorn[standard]>=0.32.0 | ASGI 서버 |
| sqlalchemy[asyncio]>=2.0.0 | ORM + async DB |
| asyncpg>=0.30.0 | PostgreSQL async 드라이버 |
| python-multipart>=0.0.9 | 파일 업로드 |
| python-dotenv>=1.0.0 | .environments 파싱 |
| pydantic>=2.0.0 | 스키마 검증 |
| httpx>=0.27.0 | 워커 HTTP 클라이언트 |
| click>=8.0.0 | 워커 CLI |
| psutil>=6.0.0 | 하드웨어 정보 수집 |

---

## 설계 문서

상세 설계는 `distributed-training-plan.md` (프로젝트 루트) 참조.
17개 섹션, ~2100줄의 전체 시스템 설계 문서.
