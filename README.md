# FAI (Family AI)

**FAI(Family AI)**는 수백~수만 명이 자발적으로 참여하여, 각자가 소유한 컴퓨터의 컴퓨팅 파워를 공유하는 **품앗이 형태의 오픈 LLM**을 만드는 프로젝트입니다. 초기 학습 데이터는 Dart와 Flutter 개발 학습 자료를 기반으로, 개념 설명·코드 예시·학습 가이드형 응답을 생성합니다.

- **공식 명칭**: FAI (Family AI)
- **기술적 분류 명칭**: Flutter Study GPT, Flutter LM, Family AI, Dart/Flutter Learning Model
- **전체적 의미**: 같은 가족·팀이 함께 컴퓨팅 파워를 모아 만드는 Dart/Flutter 특화 오픈 LLM

## 왜 FAI인가?

기존 대형 AI 회사들은 막대한 자본으로 데이터 센터를 건설하고, 그 비용을 회수하기 위해 수익화에 집중합니다. FAI는 이와 다른 접근을 합니다:

- **익명의 자발적 참여**: 인터넷상의 자발적인 참여자들이 자신의 컴퓨터(GPU/CPU)를 제공하여 슈퍼 컴퓨팅 환경을 구성
- **수익화 불가 모델**: 익명의 참여로 만들어진 AI는 특정 회사가 마음대로 운영하거나 수익화할 수 없음
- **완전 오픈소스**: 100% 오픈소스로, 최소한의 운영비를 위한 수익화만 가능한 구조
- **100% 직접 구현**: 파인튜닝이 아닌, 토크나이저부터 GPT 모델까지 처음부터 학습

## 분산 학습 시스템

BOINC(과학 분산 컴퓨팅) + Federated Learning(연합 학습)의 개념을 결합한 분산 학습 플랫폼입니다.

```
┌─────────────────────────────────────┐
│        중앙 서버 (Coordinator)        │
│     Supabase PostgreSQL + Storage    │
│     FastAPI (REST + WebSocket)       │
└──────────────────┬──────────────────┘
                   │ HTTPS
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ 워커 A    │ │ 워커 B    │ │ 워커 C    │
  │ (GPU)    │ │ (CPU)    │ │ (GPU)    │
  │ 오전 참여  │ │ 밤에 참여  │ │ 주말 참여  │
  └──────────┘ └──────────┘ └──────────┘
```

### 핵심 특징

| 특징 | 설명 |
|------|------|
| **자유 참여/이탈** | 스크립트 실행으로 참여, Ctrl+C로 이탈. 다른 워커에 영향 없음 |
| **하드웨어 무관** | NVIDIA GPU, Apple Silicon, CPU 모두 참여 가능 |
| **대규모 확장** | 수십 ~ 수만 대의 컴퓨터가 동시 참여 가능 |
| **진행 보존** | 어떤 워커가 빠져도 학습 진행 상태 유지 |
| **기여도 추적** | 누가 얼마나 기여했는지 투명하게 기록 |
| **기여 보상** | 학습 기여량에 비례하여 API 토큰 크레딧 적립 |

### 워커 참여 방법

```bash
# 간단한 참여 방법
$ python -m distributed.worker \
    --name "내 컴퓨터" \
    --server https://fai.example.com \
    --experiment 1

# 하드웨어 자동 감지 → 최적 배치 크기 결정 → 학습 시작
# Ctrl+C로 언제든 안전하게 종료 가능
```

### 학습 흐름

```
워커 참여 → 최신 체크포인트 다운로드 → 로컬 학습 수행 → 결과 업로드
→ 서버에서 FedAvg 병합 → 글로벌 모델 업데이트 → 반복
```

상세 구현 계획은 [distributed-training-plan.md](distributed-training-plan.md)를 참조하세요.

## 프로젝트 목표

- **학습 가이드형 출력**: 개념 설명, 코드 예시, 학습 체크리스트를 구조화된 형식으로 생성
- **Mac M4 최적화**: MPS(Metal Performance Shaders) GPU 가속 지원
- **Dart/Flutter 특화**: 초기 학습 데이터는 Dart/Flutter 공식 문서 기반

## 단계별 가이드

| 문서 | 설명 |
|------|------|
| [study.md](study.md) | 단계별 학습 가이드 (8단계) |
| [step-by-step.md](step-by-step.md) | 프로젝트 생성부터 텍스트 생성까지 9단계 가이드 |
| [faq.md](faq.md) | 자주 묻는 질문 (토큰화, 임베딩, 학습 데이터 등) |

## 빠른 시작

```bash
# 1. 의존성 설치 (uv 사용)
uv add torch tokenizers tqdm numpy

# 2. 순차 실행
uv run python scripts/prepare_samples.py      # 데이터 전처리
uv run python scripts/train_tokenizer.py      # 토크나이저 학습
uv run python scripts/build_bin_dataset.py    # 바이너리 데이터셋 생성
uv run python scripts/train_gpt.py            # GPT 모델 학습
uv run python scripts/generate.py             # 텍스트 생성
```

## 프로젝트 구조

```
fai/
├── data/                    # 데이터 디렉토리
│   ├── raw/                 # Stage 1: 원본 Markdown (사이트별)
│   ├── samples/             # Stage 2: 전처리된 학습 데이터
│   ├── samples.txt          # Stage 3: 통합 학습 데이터
│   ├── tokenizer.json       # BPE 토크나이저
│   └── train.bin, val.bin   # 바이너리 데이터셋
├── scripts/                 # 실행 스크립트
│   ├── prepare_samples.py   # 데이터 전처리
│   ├── train_tokenizer.py   # 토크나이저 학습
│   ├── build_bin_dataset.py # 바이너리 변환
│   ├── train_gpt.py         # GPT 학습
│   └── generate.py          # 텍스트 생성
├── distributed/             # 분산 학습 패키지
│   ├── server/              # 중앙 서버 (Coordinator)
│   ├── worker/              # 워커 클라이언트
│   └── common/              # 서버/워커 공통 모듈
├── checkpoints/             # 모델 체크포인트
├── docs/                    # 상세 기술 문서 (00~08)
├── faq/                     # FAQ 문서 (개념별 분리)
├── faq.md                   # FAQ 목차 및 빠른 참조
├── CLAUDE.md                # AI 어시스턴트 가이드
└── pyproject.toml           # uv 프로젝트 설정
```

## 문서 목록

### 기술 문서 (docs/)

상세한 학습 자료는 `docs/` 폴더에서 확인하세요.

| 문서 | 내용 |
|------|------|
| [00-overview.md](docs/00-overview.md) | 프로젝트 개요, 목표, 출력 형식 예시 |
| [01-environment-setup.md](docs/01-environment-setup.md) | Python 환경, PyTorch, MPS 설정 |
| [02-project-structure.md](docs/02-project-structure.md) | 폴더 구조, 파일 설명, 실행 순서 |
| [03-data-preparation.md](docs/03-data-preparation.md) | 데이터 수집, 전처리, 학습 형식 |
| [04-tokenizer.md](docs/04-tokenizer.md) | BPE 토크나이저 학습 |
| [05-model-architecture.md](docs/05-model-architecture.md) | GPT 모델 아키텍처, 코드 |
| [06-training.md](docs/06-training.md) | 학습 루프, 체크포인트 |
| [07-generation.md](docs/07-generation.md) | 텍스트 생성, 샘플링 파라미터 |
| [08-server.md](docs/08-server.md) | FAI LLM 서버, API, 데몬 실행 |

### 분산 학습 문서

| 문서 | 내용 |
|------|------|
| [distributed-training-plan.md](distributed-training-plan.md) | 분산 학습 시스템 상세 구현 계획 |

### FAQ 문서 (faq/)

핵심 개념을 쉽게 설명합니다. 전체 목차는 [faq.md](faq.md)를 참조하세요.

| 카테고리 | 문서 |
|----------|------|
| **프로젝트 개요** | [data-flow.md](faq/data-flow.md), [why-flutter-llm.md](faq/why-flutter-llm.md), [sample-data.md](faq/sample-data.md), [sample-data-strategy.md](faq/sample-data-strategy.md) |
| **환경 설정** | [environment-setup.md](faq/environment-setup.md), [project-structure.md](faq/project-structure.md) |
| **데이터 준비** | [data-preparation.md](faq/data-preparation.md) |
| **토큰화** | [why-tokenize.md](faq/why-tokenize.md), [how-to-tokenize.md](faq/how-to-tokenize.md), [bpe-algorithm.md](faq/bpe-algorithm.md), [vocab-size.md](faq/vocab-size.md), [after-tokenize.md](faq/after-tokenize.md) |
| **모델 구조** | [embedding-and-prediction.md](faq/embedding-and-prediction.md), [model-architecture.md](faq/model-architecture.md) |
| **학습 및 배포** | [training.md](faq/training.md), [generation.md](faq/generation.md), [server.md](faq/server.md) |
| **심화 학습** | [core-concepts.md](faq/core-concepts.md), [troubleshooting.md](faq/troubleshooting.md) |

## 핵심 개념

1. **Tokenizer**: 텍스트 → 정수 토큰 변환
2. **Embedding**: 토큰 ID → 고차원 벡터
3. **Self-Attention**: 문맥 내 단어 간 관계 학습
4. **Next-token Prediction**: GPT의 유일한 학습 목표
5. **데이터 포맷 = 모델 능력**: Dart/Flutter 학습 데이터로 학습하면 개발 학습 정보 출력
6. **Federated Averaging**: 다수 워커의 로컬 학습 결과를 가중 평균으로 병합

## 권장 하이퍼파라미터 (M4 기준)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| vocab_size | 24,000 | 토크나이저 어휘 크기 |
| block_size | 256 | 컨텍스트 길이 |
| n_layer | 6 | Transformer 블록 수 |
| n_head | 6 | Attention Head 수 |
| n_embd | 384 | 임베딩 차원 |
| batch_size | 16 | 배치 크기 |

## 출력 예시

```
[ANSWER]
요약:
- Flutter의 StatefulWidget 개념과 생명주기 설명
- 상태 관리의 기본 원리 이해

학습 체크리스트:
- 선수 지식:
  - (1) Dart 기본 문법 이해
  - (2) 객체지향 프로그래밍 개념
- 학습 목표:
  - (1) StatefulWidget과 StatelessWidget의 차이 이해
  - (2) setState() 사용법 숙지

코드 예시:
- StatefulWidget 기본 구조
  - 클래스: MyStatefulWidget extends StatefulWidget
  - 상태 클래스: _MyStatefulWidgetState extends State
  - 참고: https://docs.flutter.dev/

상세 설명:
StatefulWidget은 변경 가능한 상태를 가진 위젯입니다...
[/ANSWER]
```

## 참고 자료

- [build-nanogpt](https://github.com/karpathy/build-nanogpt) - Karpathy의 GPT 구현 튜토리얼
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) - BPE 토크나이저 문서
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html) - Mac GPU 가속 문서
- [Flutter 공식 문서](https://docs.flutter.dev/) - Flutter 개발 공식 가이드
- [Dart 공식 문서](https://dart.dev/guides) - Dart 언어 공식 가이드
