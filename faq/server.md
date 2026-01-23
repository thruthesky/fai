# JAI LLM 서버

JAI LLM을 데몬(서비스)으로 실행하여 여러 클라이언트가 동시에 접속하고 프롬프트를 전송하면 응답을 받을 수 있는 서버를 구축합니다.

---

## 한 줄 요약

> **"FastAPI + Semaphore로 동시 요청을 안전하게 처리"**

---

## 1. 아키텍처

```
┌─────────────┐     HTTP/REST      ┌─────────────────┐
│  Client 1   │ ──────────────────▶│                 │
└─────────────┘                    │                 │
                                   │   JAI Server    │
┌─────────────┐     HTTP/REST      │   (FastAPI)     │──▶ JAI LLM Model
│  Client 2   │ ──────────────────▶│                 │      (ckpt.pt)
└─────────────┘                    │                 │
                                   └─────────────────┘
┌─────────────┐     HTTP/REST             │
│  Client N   │ ──────────────────────────┘
└─────────────┘
```

### 주요 특징

- **FastAPI 기반**: 비동기 처리로 동시 요청 처리
- **모델 싱글톤**: 서버 시작 시 모델 1회 로드, 메모리 효율화
- **요청 큐**: 동시 요청을 순차 처리하여 GPU 메모리 보호
- **데몬 실행**: systemd 또는 nohup으로 백그라운드 실행

---

## 2. 의존성 설치

```bash
uv add fastapi uvicorn
```

---

## 3. 서버 코드 개요

### scripts/server.py

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 동시 요청 제어를 위한 세마포어 (GPU 메모리 보호)
inference_semaphore = asyncio.Semaphore(1)

# 전역 변수 (싱글톤 패턴)
model = None
tokenizer = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 lifecycle 관리"""
    print("JAI LLM 서버 시작 중...")
    load_model()  # 모델 로드
    print("JAI LLM 서버 준비 완료!")
    yield
    print("JAI LLM 서버 종료 중...")

app = FastAPI(
    title="JAI LLM Server",
    description="구인 정보 특화 LLM API 서버",
    lifespan=lifespan,
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50

class GenerateResponse(BaseModel):
    prompt: str
    generated: str
    full_text: str

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """텍스트 생성 API"""
    async with inference_semaphore:  # 동시 추론 제어
        # 토큰화 → 생성 → 디코딩
        ...
        return GenerateResponse(...)

@app.post("/chat")
async def chat(request: GenerateRequest):
    """대화형 API (QUESTION/ANSWER 형식으로 자동 감싸기)"""
    formatted_prompt = f"[QUESTION]\n{request.prompt}\n[/QUESTION]\n\n[ANSWER]\n"
    request.prompt = formatted_prompt
    return await generate(request)
```

---

## 4. 서버 실행 방법

### 4.1 직접 실행

```bash
# 개발 모드 (자동 리로드)
uv run uvicorn scripts.server:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
uv run uvicorn scripts.server:app --host 0.0.0.0 --port 8000 --workers 1
```

### 4.2 백그라운드 실행 (nohup)

```bash
# 백그라운드 실행
nohup uv run uvicorn scripts.server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# 프로세스 확인
ps aux | grep uvicorn

# 종료
pkill -f "uvicorn scripts.server:app"
```

### 4.3 systemd 데몬 등록 (Linux)

`/etc/systemd/system/jai-server.service` 파일 생성:

```ini
[Unit]
Description=JAI LLM Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/jai
ExecStart=/home/ubuntu/.local/bin/uv run uvicorn scripts.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 데몬 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable jai-server
sudo systemctl start jai-server

# 상태 확인
sudo systemctl status jai-server

# 로그 확인
sudo journalctl -u jai-server -f
```

### 4.4 launchd 데몬 등록 (macOS)

`~/Library/LaunchAgents/com.jai.server.plist` 파일 생성:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jai.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/사용자명/.local/bin/uv</string>
        <string>run</string>
        <string>uvicorn</string>
        <string>scripts.server:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/사용자명/apps/jai</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

```bash
# 데몬 등록 및 시작
launchctl load ~/Library/LaunchAgents/com.jai.server.plist

# 상태 확인
launchctl list | grep jai

# 종료
launchctl unload ~/Library/LaunchAgents/com.jai.server.plist
```

---

## 5. API 사용법

### 5.1 curl 예시

```bash
# 헬스 체크
curl http://localhost:8000/health

# 텍스트 생성
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "[QUESTION]\n서울에서 React 개발자 채용 있어?\n[/QUESTION]\n\n[ANSWER]\n",
    "max_tokens": 256,
    "temperature": 0.8
  }'

# 대화형 API (자동 포맷팅)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "판교에서 백엔드 개발자 채용하는 곳 알려줘",
    "max_tokens": 256
  }'
```

### 5.2 Python 클라이언트

```python
import requests

BASE_URL = "http://localhost:8000"

def chat(prompt: str, max_tokens: int = 256) -> str:
    """대화형 API 호출"""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.8,
        },
    )
    response.raise_for_status()
    return response.json()["generated"]

# 사용 예시
answer = chat("서울에서 프론트엔드 개발자 채용 있어?")
print(answer)
```

---

## 6. 동시 접속 처리

### 동작 방식

```
Client A ─────┐
              │
Client B ─────┼──▶ FastAPI (비동기) ──▶ Semaphore(1) ──▶ GPU 추론
              │         │
Client C ─────┘         │
                        ▼
                   요청 큐잉 (대기)
```

1. **FastAPI 비동기 처리**: 여러 클라이언트의 HTTP 요청을 동시에 받음
2. **Semaphore(1)**: GPU 추론은 한 번에 하나씩만 실행 (메모리 보호)
3. **요청 큐잉**: 추론 중인 요청이 있으면 다른 요청은 대기

### 동시성 조절

```python
# 동시 추론 수 조절 (GPU 메모리에 따라)
inference_semaphore = asyncio.Semaphore(2)  # 동시 2개 추론
```

---

## 7. 모니터링

### Swagger UI

서버 실행 후 브라우저에서 접속:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 로깅 추가

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("jai-server")

@app.post("/generate")
async def generate(request: GenerateRequest):
    logger.info(f"요청 수신: {request.prompt[:50]}...")
    # ...
    logger.info(f"응답 완료: {len(generated)} 토큰 생성")
```

---

## 요약

| 질문 | 답변 |
|------|------|
| 프레임워크? | FastAPI + Uvicorn |
| 동시 요청 처리? | Semaphore로 GPU 보호 |
| 데몬 실행? | systemd (Linux) / launchd (Mac) |
| API 문서? | http://localhost:8000/docs |

---

## 관련 문서

- [generation.md](generation.md) - 텍스트 생성
- [training.md](training.md) - 모델 학습
- [environment-setup.md](environment-setup.md) - 환경 설정

---

## 참고 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Uvicorn 공식 문서](https://www.uvicorn.org/)
