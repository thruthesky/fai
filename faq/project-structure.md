# 프로젝트 폴더 구조

JAI 프로젝트의 폴더 구조와 각 파일의 역할을 설명합니다.

---

## 한 줄 요약

> **data/ + scripts/ + checkpoints/ = JAI LLM 완성**

---

## 1. 전체 구조

```
jai/
├── data/
│   ├── raw.txt                 # 원본 구인 데이터
│   ├── samples.txt             # 전처리 후 학습용 샘플
│   ├── tokenizer.json          # 내 데이터로 학습한 토크나이저
│   ├── train.bin               # 학습용 토큰 시퀀스 (바이너리)
│   └── val.bin                 # 검증용 토큰 시퀀스 (바이너리)
├── scripts/
│   ├── prepare_samples.py      # 데이터 전처리 및 샘플 생성
│   ├── train_tokenizer.py      # BPE 토크나이저 학습
│   ├── build_bin_dataset.py    # 토큰화 → 바이너리 변환
│   ├── train_gpt.py            # GPT 모델 학습
│   ├── generate.py             # 텍스트 생성
│   └── server.py               # FastAPI 서버 (선택)
├── checkpoints/
│   └── ckpt.pt                 # 모델 체크포인트
├── docs/                       # 상세 기술 문서
├── faq/                        # FAQ 문서 (개념별)
├── pyproject.toml              # uv 프로젝트 설정
├── uv.lock                     # 의존성 락 파일
├── CLAUDE.md                   # AI 어시스턴트 가이드
└── README.md                   # 프로젝트 설명
```

---

## 2. 각 폴더/파일 설명

### data/ 폴더

| 파일 | 설명 | 생성 시점 |
|------|------|-----------|
| `raw.txt` | 원본 구인 데이터 | 사용자가 준비 |
| `samples.txt` | 전처리된 학습용 텍스트 | `prepare_samples.py` 실행 후 |
| `tokenizer.json` | BPE 토크나이저 | `train_tokenizer.py` 실행 후 |
| `train.bin` | 학습용 토큰 배열 | `build_bin_dataset.py` 실행 후 |
| `val.bin` | 검증용 토큰 배열 | `build_bin_dataset.py` 실행 후 |

### scripts/ 폴더

| 파일 | 역할 | 실행 순서 |
|------|------|-----------|
| `prepare_samples.py` | raw.txt를 학습 가능한 형식으로 변환 | 1번째 |
| `train_tokenizer.py` | BPE 토크나이저 학습 | 2번째 |
| `build_bin_dataset.py` | 텍스트를 토큰 ID 배열로 변환 | 3번째 |
| `train_gpt.py` | GPT 모델 학습 | 4번째 |
| `generate.py` | 학습된 모델로 텍스트 생성 | 5번째 |
| `server.py` | FastAPI 서버 (선택사항) | 학습 완료 후 |

### checkpoints/ 폴더

| 파일 | 설명 |
|------|------|
| `ckpt.pt` | 모델 가중치, 옵티마이저 상태, 학습 단계 저장 |

---

## 3. 실행 순서

```bash
# 1) 학습 샘플 생성 (raw.txt → samples.txt)
uv run python scripts/prepare_samples.py

# 2) 토크나이저 학습 (samples.txt → tokenizer.json)
uv run python scripts/train_tokenizer.py

# 3) 바이너리 데이터셋 생성 (samples.txt → train.bin, val.bin)
uv run python scripts/build_bin_dataset.py

# 4) LLM 학습 (train.bin, val.bin → checkpoints/ckpt.pt)
uv run python scripts/train_gpt.py

# 5) 생성 테스트
uv run python scripts/generate.py
```

---

## 4. 데이터 흐름도

```
┌─────────────┐
│   raw.txt   │  ← 원본 구인 데이터
└──────┬──────┘
       │
       ▼ scripts/prepare_samples.py
┌─────────────┐
│ samples.txt │  ← [QUESTION]...[ANSWER] 형식으로 변환
└──────┬──────┘
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
scripts/train_tokenizer.py   scripts/build_bin_dataset.py
       │                      │
       ▼                      ▼
┌──────────────┐      ┌─────────────┐
│tokenizer.json│      │ train.bin   │
└──────────────┘      │ val.bin     │
                      └──────┬──────┘
                             │
                             ▼ scripts/train_gpt.py
                      ┌─────────────┐
                      │ checkpoints │
                      │   ckpt.pt   │
                      └──────┬──────┘
                             │
                             ▼ scripts/generate.py
                      ┌─────────────┐
                      │  생성 결과   │
                      └─────────────┘
```

---

## 5. pyproject.toml

uv 프로젝트 설정 파일입니다:

```toml
[project]
name = "jai"
version = "0.1.0"
description = "From-scratch GPT for job information summarization"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "tokenizers",
  "torch",
  "tqdm",
]
```

---

## 6. 프로젝트 초기 설정

새 프로젝트를 시작할 때:

```bash
# 새 프로젝트 생성
uv init jai
cd jai

# 필요한 폴더 생성
mkdir -p data scripts checkpoints docs faq

# 의존성 추가
uv add torch tokenizers tqdm numpy

# 서버 구축 시
uv add fastapi uvicorn
```

---

## 요약

| 폴더 | 역할 |
|------|------|
| `data/` | 데이터 저장 (원본, 전처리, 토크나이저, 바이너리) |
| `scripts/` | 실행 스크립트 (전처리, 학습, 생성, 서버) |
| `checkpoints/` | 모델 체크포인트 저장 |
| `docs/` | 상세 기술 문서 |
| `faq/` | FAQ 문서 |

---

## 관련 문서

- [environment-setup.md](environment-setup.md) - 환경 설정
- [data-flow.md](data-flow.md) - 데이터 흐름 상세 설명
- [data-preparation.md](data-preparation.md) - 데이터 전처리
