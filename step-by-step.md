# JAI 인공지능 만들기: 단계별 가이드

본 문서를 따라하면 자신만의 Job LLM을 처음부터 직접 구현하고 학습시킬 수 있습니다.

---

## 개요

**JAI(Job AI)**는 전 세계 구인/구직 정보 생성에 특화된 LLM을 만드는 학습 프로젝트입니다.

### 활용 예시

> 해외 취업을 준비 중 → 미국 실리콘밸리 소프트웨어 엔지니어 채용 정보가 필요함 → JAI에게 질문

이런 상황에서 전 세계에 흩어져 있는 구인/구직 정보를 제공하는 LLM입니다.

---

## 1단계: 프로젝트 생성

```bash
mkdir jai
cd jai
uv init
```

`uv init` 명령어가 `pyproject.toml` 파일을 생성합니다.

> 📚 **참고**: [02-project-structure.md](docs/02-project-structure.md) - 프로젝트 구조

---

## 2단계: 패키지 설치

```bash
uv add torch tokenizers tqdm numpy
```

| 패키지 | 용도 |
|--------|------|
| `torch` | PyTorch 딥러닝 프레임워크 |
| `tokenizers` | BPE 토크나이저 |
| `tqdm` | 진행률 표시 |
| `numpy` | 수치 연산 |

> 📚 **참고**: [01-environment-setup.md](docs/01-environment-setup.md) - 환경 설정

---

## 3단계: MPS 확인

Apple Silicon Mac에서 GPU 가속 사용 가능 여부를 확인합니다.

```bash
uv run python -c "import torch; print('MPS 사용 가능:', torch.backends.mps.is_available())"
```

`True`가 출력되면 GPU 가속을 사용할 수 있습니다.

> 📚 **참고**: [01-environment-setup.md](docs/01-environment-setup.md) - 환경 설정

---

## 4단계: 폴더 구조 생성

LLM 학습에 필요한 폴더들을 생성합니다.

```bash
mkdir -p data scripts checkpoints
```

```
jai/
├── data/           # 데이터 파일
├── scripts/        # Python 스크립트
├── checkpoints/    # 모델 체크포인트
└── pyproject.toml  # 프로젝트 설정
```

### 각 폴더 역할

| 폴더 | 용도 | 생성되는 파일 |
|------|------|---------------|
| `data/` | 학습 데이터 저장 | `raw.txt`, `samples.txt`, `tokenizer.json`, `train.bin`, `val.bin` |
| `scripts/` | 실행 스크립트 | `prepare_samples.py`, `train_tokenizer.py`, `build_bin_dataset.py`, `train_gpt.py`, `generate.py` |
| `checkpoints/` | 학습된 모델 저장 | `ckpt.pt` (모델 가중치, 옵티마이저 상태) |

### 개념 설명

- [체크포인트 파일 (`ckpt.pt`)](study.md#체크포인트-파일-ckptpt) - 모델 가중치, 옵티마이저 상태, 학습 단계
- [모델 가중치 vs 벡터](study.md#모델-가중치-vs-벡터) - 역할과 형태의 차이
- [파라미터란?](study.md#파라미터-parameter) - 모델이 학습하는 조절 가능한 숫자
- [파라미터·벡터·텐서의 관계](study.md#파라미터벡터텐서의-관계) - 파라미터 하나가 스칼라 숫자라는 의미
- [파라미터 수와 메모리](study.md#파라미터-수와-메모리) - 8B·32B 모델, 비트 수, 양자화와 실행 메모리

> 📚 **참고**: [02-project-structure.md](docs/02-project-structure.md) - 프로젝트 구조

---

## 5단계: 데이터 준비

학습용 샘플 데이터를 준비합니다. 두 가지 방법 중 선택할 수 있습니다.

### 방법 A: 텍스트 파일에서 변환

`data/raw.txt`에 원본 구인/구직 데이터를 준비합니다.

```bash
uv run python scripts/prepare_samples.py
```

### 방법 B: JSON 파일에서 변환 (권장)

구인 정보가 담긴 JSON 파일을 샘플 데이터로 변환합니다.

```bash
uv run python scripts/convert_jobs_to_samples.py
```

**입력**: `data/jobs-sg.json` (구인 정보 JSON)
**출력**: `data/samples.txt` (학습용 샘플)

### 샘플 데이터 형식

```
[QUESTION]
이 연락처 정보를 이해하기 쉽게 요약해줘.
[/QUESTION]

[DOC]
=== [회사명] 포지션 채용 ===
회사명: ...
포지션: ...
위치: ...
[/DOC]

[ANSWER]
요약:
- 회사: ...
- 포지션: ...

체크리스트:
- 해야 할 일: ...

연락처(공공정보):
- WEB: ...
[/ANSWER]
```

> 📚 **참고**: [03-data-preparation.md](docs/03-data-preparation.md) - 데이터 전처리 | [job-hiring.md](docs/job-hiring.md) - 구인 정보 필드

---

## 6단계: 토크나이저 학습

텍스트를 토큰으로 변환하는 BPE 토크나이저를 학습합니다.

```bash
uv run python scripts/train_tokenizer.py
```

**결과**: `data/tokenizer.json` 생성

> 📚 **참고**: [04-tokenizer.md](docs/04-tokenizer.md) - 토크나이저

---

## 7단계: 바이너리 데이터셋 생성

토큰화된 데이터를 학습용 바이너리 파일로 변환합니다.

```bash
uv run python scripts/build_bin_dataset.py
```

**결과**: `data/train.bin`, `data/val.bin` 생성

> 📚 **참고**: [03-data-preparation.md](docs/03-data-preparation.md) - 데이터 전처리

---

## 8단계: GPT 모델 학습

Transformer 기반 GPT 모델을 학습합니다.

```bash
uv run python scripts/train_gpt.py
```

**결과**: `checkpoints/ckpt.pt` 생성

학습 중 다음과 같은 출력이 표시됩니다:
```
step 0: train loss 10.234, val loss 10.198
step 100: train loss 6.543, val loss 6.612
...
```

Loss가 점점 줄어들면 학습이 잘 되고 있는 것입니다.

> 📚 **참고**: [05-model-architecture.md](docs/05-model-architecture.md) - 모델 아키텍처 | [06-training.md](docs/06-training.md) - GPT 학습

---

## 9단계: 텍스트 생성

학습된 모델로 텍스트를 생성합니다.

```bash
uv run python scripts/generate.py
```

**예시 출력**:
```
[QUESTION]
미국 실리콘밸리 소프트웨어 엔지니어 채용
[/QUESTION]

[ANSWER]
요약:
- 실리콘밸리 주요 기업 채용 정보

구인 정보:
- Google Inc.
  - 포지션: Senior Software Engineer
  - 연봉: $150,000 - $200,000
  - 위치: Mountain View, CA
[/ANSWER]
```

> 📚 **참고**: [07-generation.md](docs/07-generation.md) - 텍스트 생성 | [AI 런타임](study.md#ai-런타임-학습된-모델을-실행하는-엔진) - ONNX Runtime, llama.cpp, FreeToken 비교

---

## 전체 실행 순서 요약

```bash
# 1. 프로젝트 설정
uv init
uv add torch tokenizers tqdm numpy
mkdir -p data scripts checkpoints

# 2. 순차 실행
uv run python scripts/convert_jobs_to_samples.py  # JSON → samples.txt
uv run python scripts/train_tokenizer.py           # BPE 토크나이저 학습
uv run python scripts/build_bin_dataset.py         # 바이너리 변환
uv run python scripts/train_gpt.py                 # GPT 모델 학습
uv run python scripts/generate.py                  # 텍스트 생성
```

---

## 전체 파이프라인 도식

```
[데이터 준비]
    jobs-sg.json (구인 정보 JSON)
         ↓
    convert_jobs_to_samples.py
         ↓
    samples.txt (학습용 텍스트)

[토크나이저 학습]
    samples.txt
         ↓
    train_tokenizer.py (BPE 알고리즘)
         ↓
    tokenizer.json (어휘 사전: 24,000개 토큰)

[바이너리 변환]
    samples.txt + tokenizer.json
         ↓
    build_bin_dataset.py
         ↓
    train.bin, val.bin (숫자 배열)

[GPT 학습]
    train.bin + val.bin
         ↓
    train_gpt.py (임베딩 + Transformer)
         ↓
    checkpoints/ckpt.pt (학습된 모델)

[텍스트 생성]
    ckpt.pt + tokenizer.json
         ↓
    generate.py
         ↓
    "서울에서 React 개발자를 채용합니다..."
```

