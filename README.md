# FAI (Flutter AI)

**FAI(Flutter AI)**는 Dart와 Flutter 개발 학습 자료를 기반으로, Flutter 스터디 특화 LLM을 처음부터 학습하여 개념 설명·코드 예시·학습 가이드형 응답을 생성하는 프로젝트입니다.

- **공식 명칭**: FAI
- **기술적 분류 명칭**: Flutter Study GPT, Flutter LM, Flutter AI, Dart/Flutter Learning Model
- **전체적 의미**: Dart/Flutter 개발 학습에 특화된 소규모 LLM 기반 AI

## 프로젝트 목표

- **100% 직접 구현**: 파인튜닝이 아닌, 토크나이저부터 GPT 모델까지 처음부터 학습
- **학습 가이드형 출력**: 개념 설명, 코드 예시, 학습 체크리스트를 구조화된 형식으로 생성
- **Mac M4 최적화**: MPS(Metal Performance Shaders) GPU 가속 지원
- **소규모 스터디 모델**: 대규모 LLM이 아닌, Dart/Flutter 학습 정보 제공에 특화된 경량 모델

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
│   ├── raw.txt              # 원본 데이터
│   ├── samples.txt          # 전처리된 학습 샘플
│   ├── tokenizer.json       # BPE 토크나이저
│   └── train.bin, val.bin   # 바이너리 데이터셋
├── scripts/                 # 실행 스크립트
│   ├── prepare_samples.py   # 데이터 전처리
│   ├── train_tokenizer.py   # 토크나이저 학습
│   ├── build_bin_dataset.py # 바이너리 변환
│   ├── train_gpt.py         # GPT 학습
│   └── generate.py          # 텍스트 생성
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
