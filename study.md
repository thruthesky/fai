# JAI 학습 가이드

JAI (Job AI) 프로젝트를 통해 LLM을 처음부터(from scratch) 학습하는 단계별 가이드입니다.

---

## 학습 목표

이 가이드를 완료하면:
- ✅ BPE 토크나이저의 원리와 학습 방법을 이해합니다
- ✅ Decoder-only Transformer (GPT) 구조를 이해합니다
- ✅ 다음 토큰 예측(Next-token prediction) 학습 원리를 이해합니다
- ✅ 직접 학습한 LLM으로 텍스트를 생성할 수 있습니다

---

## 단계별 학습 가이드

### 1단계: 환경 설정

**학습 내용**: uv 패키지 관리자, PyTorch, MPS (Mac GPU) 설정

```bash
# uv 설치 및 프로젝트 생성
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init jai && cd jai
uv add torch tokenizers tqdm numpy
```

📚 참고: [environment-setup.md](faq/environment-setup.md), [project-structure.md](faq/project-structure.md)

### 2단계: 데이터 이해하기

**핵심 개념**: "데이터 포맷 = 모델 능력"

```
단순 텍스트 나열 → 텍스트 이어쓰기만 잘함
질문→답변 구조   → 질문에 답변하는 능력
```

📚 참고: [data-preparation.md](faq/data-preparation.md), [sample-data-strategy.md](faq/sample-data-strategy.md)

### 3단계: 토큰화 이해하기

**핵심 개념**: BPE = 자주 등장하는 문자 쌍을 병합하여 어휘 생성

```bash
uv run python scripts/train_tokenizer.py      # 토크나이저 학습
uv run python scripts/build_bin_dataset.py    # 바이너리 변환
```

📚 참고: [why-tokenize.md](faq/why-tokenize.md), [bpe-algorithm.md](faq/bpe-algorithm.md)

### 4단계: 임베딩 이해하기

**핵심 개념**: 임베딩 = 토큰 ID → 고차원 벡터 변환 (룩업 테이블)

```python
self.tok_emb = nn.Embedding(vocab_size, n_embd)  # 24000×384 행렬
x = self.tok_emb(token_ids)  # 인덱스로 행 가져오기
```

📚 참고: [embedding-and-prediction.md](faq/embedding-and-prediction.md)

### 5단계: GPT 아키텍처 이해하기

**핵심 개념**: Token Emb + Pos Emb → N × Block → Output Head

📚 참고: [model-architecture.md](faq/model-architecture.md), [core-concepts.md](faq/core-concepts.md)

### 6단계: 모델 학습하기

**핵심 개념**: Next-token prediction + CrossEntropy 손실

```bash
uv run python scripts/train_gpt.py
```

📚 참고: [training.md](faq/training.md)

### 7단계: 텍스트 생성하기

**핵심 개념**: Autoregressive 생성 (한 토큰씩 예측 → 반복)

```bash
uv run python scripts/generate.py
```

📚 참고: [generation.md](faq/generation.md)

### 8단계: 서버 배포하기 (선택)

**핵심 개념**: FastAPI + Semaphore로 동시 요청 처리

📚 참고: [server.md](faq/server.md)

---

## 전체 파이프라인

```
[데이터 준비]
jobs-sg.json → convert_jobs_to_samples.py → samples.txt
                                                 ↓
[토크나이저]                            train_tokenizer.py → tokenizer.json
                                                 ↓
[바이너리 변환]                         build_bin_dataset.py → train.bin / val.bin
                                                 ↓
[모델 학습]                             train_gpt.py → ckpt.pt
                                                 ↓
[텍스트 생성]                           generate.py → "서울에서 React 개발자를..."
```

### 각 단계의 역할

| 단계 | 입력 | 출력 | 핵심 작업 |
|------|------|------|----------|
| 데이터 준비 | JSON | 텍스트 | 구인 정보를 Q&A 형식으로 변환 |
| 토크나이저 | 텍스트 | 어휘 사전 | BPE로 24,000개 토큰 생성 |
| 바이너리 변환 | 텍스트 + 사전 | 숫자 배열 | 텍스트를 토큰 ID로 변환 |
| 모델 학습 | 숫자 배열 | 가중치 | 다음 토큰 예측 학습 |
| 텍스트 생성 | 프롬프트 | 텍스트 | 확률 분포에서 샘플링 |

---

## 권장 학습 순서

| 순서 | 주제 | 문서 |
|------|------|------|
| 1 | 프로젝트 개요 | [data-flow.md](faq/data-flow.md) |
| 2 | 환경 설정 | [environment-setup.md](faq/environment-setup.md) |
| 3 | 데이터 준비 | [data-preparation.md](faq/data-preparation.md) |
| 4 | 토큰화 | [bpe-algorithm.md](faq/bpe-algorithm.md) |
| 5 | 임베딩 | [embedding-and-prediction.md](faq/embedding-and-prediction.md) |
| 6 | 모델 구조 | [model-architecture.md](faq/model-architecture.md) |
| 7 | 학습 | [training.md](faq/training.md) |
| 8 | 생성 | [generation.md](faq/generation.md) |

---

# 인공지능 개발을 위한 학습 자료

## 파라미터 (Parameter)

### 한마디 정의

**파라미터 = 모델이 학습하는 "조절 가능한 숫자"**

### 비유로 이해하기

**요리사 비유**

```
레시피: 소금 [?]g, 설탕 [?]g, 간장 [?]ml

처음: 소금 10g, 설탕 5g, 간장 20ml  ← 랜덤하게 시작
     ↓ 맛을 보고 조절 (학습)
최종: 소금 3g, 설탕 8g, 간장 15ml   ← 최적의 값 찾음

이 [?] 자리에 들어가는 숫자들 = 파라미터
```

### 신경망에서의 파라미터

```
입력 → [파라미터] → 출력

"안녕" → [0.3, -0.5, 0.8, ...] → "하세요"
         ↑
      이 숫자들이 파라미터
```

신경망은 수백만~수십억 개의 숫자를 가지고 있고, 학습하면서 이 숫자들을 조금씩 바꿉니다.

### 구체적 예시

```python
# 간단한 신경망
y = W * x + b

# W와 b가 파라미터
W = 0.5   # 가중치 (weight)
b = 0.1   # 편향 (bias)

# 입력 x=2일 때
y = 0.5 * 2 + 0.1 = 1.1
```

학습 = "정답에 가까워지도록 W와 b 값을 조정하는 과정"

### GPT 모델의 파라미터

| 위치 | 파라미터 종류 | 역할 |
|------|-------------|------|
| 임베딩 | 단어 벡터 테이블 | 단어 → 숫자 변환 |
| 어텐션 | Q, K, V 행렬 | 단어 간 관계 파악 |
| MLP | 2개의 행렬 | 정보 변환 |

**GPT-2 Small**: 약 1억 2천만 개 파라미터
**GPT-3**: 약 1,750억 개 파라미터

### 요약

| 용어 | 의미 |
|------|------|
| **파라미터** | 모델 안의 조절 가능한 숫자 |
| **학습** | 파라미터를 좋은 값으로 바꾸는 과정 |
| **가중치** | 파라미터의 한 종류 (곱하는 숫자) |

**한 줄 정리**: 파라미터는 모델이 "배우는 내용"을 저장하는 숫자들

---

## 체크포인트 파일 (`ckpt.pt`)

학습 중간에 모델 상태를 저장하는 파일입니다.

| 항목 | 설명 |
|------|------|
| **모델 가중치** | 신경망이 학습한 숫자들의 집합. 임베딩, 어텐션, MLP 등 모든 파라미터 |
| **옵티마이저 상태** | 학습률, momentum 등. 학습 재개 시 이어서 학습하기 위해 필요 |
| **학습 단계** | 현재까지 진행된 step 번호 |

---

## 모델 가중치의 구성 요소

모델 가중치는 **상위 개념**이고, 임베딩/어텐션/MLP는 그 **하위 구성 요소**입니다.

### 관계도

```
모델 가중치 (Model Weights) ← 전체를 포함하는 상위 개념
    │
    ├── 임베딩 가중치 (Embedding Weights)
    │       └── 토큰 → 벡터 변환 테이블
    │
    ├── 어텐션 가중치 (Attention Weights)
    │       └── Q, K, V 행렬
    │
    └── MLP 가중치 (MLP Weights)
            └── Feed Forward 레이어
```

### 각각의 역할

| 개념 | 역할 | 비유 |
|------|------|------|
| **모델 가중치** | 모델의 **모든** 학습 가능한 숫자들 | 요리책 전체 |
| **임베딩** | 단어를 숫자 벡터로 변환 | 재료를 계량하는 규칙 |
| **어텐션** | "어떤 단어에 집중할지" 결정 | 어떤 재료를 강조할지 결정 |
| **MLP** | 정보를 비선형 변환 | 재료를 조리하는 과정 |

### 구체적 예시: GPT 모델의 가중치 구성

```python
# GPT 모델의 state_dict() 내용
model.state_dict() = {
    # ─────────────────────────────────────────────
    # 1. 임베딩 가중치 (단어 → 벡터 변환)
    # ─────────────────────────────────────────────
    "wte.weight": [24000, 384],   # 토큰 임베딩: 24000개 토큰, 각각 384차원 벡터
    "wpe.weight": [256, 384],     # 위치 임베딩: 256개 위치, 각각 384차원 벡터

    # ─────────────────────────────────────────────
    # 2. 어텐션 가중치 (단어 간 관계 파악)
    # ─────────────────────────────────────────────
    "blocks.0.attn.c_attn.weight": [384, 1152],  # Q, K, V 행렬 (한번에 계산)
    "blocks.0.attn.c_proj.weight": [384, 384],   # 출력 투영 행렬

    # ─────────────────────────────────────────────
    # 3. MLP 가중치 (비선형 변환)
    # ─────────────────────────────────────────────
    "blocks.0.mlp.c_fc.weight": [384, 1536],     # 확장 (384 → 1536)
    "blocks.0.mlp.c_proj.weight": [1536, 384],   # 축소 (1536 → 384)

    # ... (모든 블록에서 반복)
}
```

### 각 구성 요소 상세 설명

#### 1. 임베딩 (Embedding)

```
"채용" → [0.1, 0.3, -0.2, 0.5, ...]  (384개 숫자)
"연봉" → [0.4, -0.1, 0.8, 0.2, ...]  (384개 숫자)
```

- **역할**: 토큰(단어)을 고정 크기 벡터로 변환
- **학습 내용**: "비슷한 의미의 단어는 비슷한 벡터를 갖도록" 학습

#### 2. 어텐션 (Attention)

```
문장: "구글에서 시니어 엔지니어를 채용합니다"

"채용"이 다른 단어들에 주의를 기울이는 정도:
  구글에서  → 0.3 (높음, 어디서 채용하는지 중요)
  시니어    → 0.4 (높음, 어떤 레벨인지 중요)
  엔지니어를 → 0.2 (중간)
  채용합니다 → 0.1 (자기 자신)
```

- **역할**: "어떤 단어가 어떤 단어에 집중해야 하는지" 결정
- **Q, K, V**: Query(질문), Key(키), Value(값) 행렬로 관계 계산

#### 3. MLP (Multi-Layer Perceptron)

```
입력 벡터 (384차원)
    ↓ 확장
중간 벡터 (1536차원) + GELU 활성화
    ↓ 축소
출력 벡터 (384차원)
```

- **역할**: 어텐션이 찾은 관계를 바탕으로 정보를 변환
- **비선형 변환**: 복잡한 패턴을 학습할 수 있게 함

### 핵심 정리

| 질문 | 답변 |
|------|------|
| 모델 가중치 = 임베딩? | ❌ 임베딩은 가중치의 **일부** |
| 모델 가중치 = 어텐션? | ❌ 어텐션도 가중치의 **일부** |
| 모델 가중치 = ? | ✅ 임베딩 + 어텐션 + MLP + ... **전부 합친 것** |

**한 줄 정리**: "신경망이 학습한 숫자들의 집합"은 모델 가중치 **전체**를 말하며, 임베딩/어텐션/MLP는 각각 특정 역할을 담당하는 **부분**입니다.

---

## 모델 가중치 vs 벡터

| 구분 | 벡터 | 가중치 |
|------|------|--------|
| 정의 | 숫자 배열 (데이터 구조) | 학습되는 파라미터 (역할) |
| 형태 | 1차원 배열 | 벡터, 행렬, 텐서 등 다양 |
| 변화 | 고정 또는 계산됨 | 학습 중 계속 업데이트 |

```python
# 임베딩 테이블 예시
#         [----벡터----]
vocab = [[0.1, 0.2, 0.3],  # "채용"의 벡터
         [0.4, 0.5, 0.6],  # "연봉"의 벡터
         [0.7, 0.8, 0.9]]  # "회사"의 벡터

# 전체 테이블 = "가중치", 각 행 = "벡터"
```

**요약**: 가중치는 "무엇을 하는지(역할)", 벡터는 "어떤 모양인지(형태)"

---

## 토큰화 (Tokenization)

### 왜 토큰화가 필요한가?

컴퓨터는 텍스트를 직접 이해할 수 없습니다. 숫자로 변환해야 수학 연산이 가능합니다.

```
"서울에서 React 개발자 채용"
         ↓ 토큰화
["서울", "에서", "React", "개발자", "채용"]
         ↓ ID 변환
[1523, 892, 4521, 2847, 1956]
```

### BPE (Byte Pair Encoding) 알고리즘

자주 등장하는 문자 쌍을 반복적으로 병합하여 어휘를 생성합니다.

```
1단계: 문자 단위로 시작
  "low" → ['l', 'o', 'w']
  "lower" → ['l', 'o', 'w', 'e', 'r']

2단계: 가장 빈번한 쌍 병합
  ('l', 'o') → 'lo'  (빈도 높음)
  → ['lo', 'w'], ['lo', 'w', 'e', 'r']

3단계: 반복
  ('lo', 'w') → 'low'
  → ['low'], ['low', 'e', 'r']
```

### train_tokenizer.py 동작 원리

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# 1. BPE 토크나이저 생성
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. 트레이너 설정 (어휘 크기 24,000)
trainer = BpeTrainer(
    vocab_size=24000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)

# 3. samples.txt로 학습
tokenizer.train(["data/samples.txt"], trainer)

# 4. 저장
tokenizer.save("data/tokenizer.json")
```

---

## GPT의 다음 토큰 예측

### 흔한 오해

> "GPT가 임베딩 벡터들 중에서 가장 비슷한 벡터를 찾아서 다음 단어를 예측한다" ❌

### 실제 방식

GPT는 **벡터 검색이 아닌 수학적 연산**으로 다음 토큰을 예측합니다.

```
입력: "나는 밥을"
         ↓
[임베딩] → 벡터로 변환
         ↓
[Transformer × 6] → 문맥 이해
         ↓
[출력 헤드] → 24,000개 토큰의 확률 분포
         ↓
[샘플링] → 다음 토큰 선택

결과: "먹었다" (확률 35%)
```

### 비교 표

| | 벡터 검색 (RAG) | GPT 다음 토큰 예측 |
|---|---|---|
| **방식** | 코사인 유사도로 비슷한 벡터 찾기 | Transformer 연산 후 확률 분포 생성 |
| **용도** | 외부 문서/지식 검색 | 언어 모델 내부 예측 |
| **출력** | 유사한 문서/벡터 | vocab_size 크기의 확률 분포 |

### 핵심 차이

- **유사도 검색**: "비슷한 것 찾기" (검색)
- **확률 분포**: "다음에 올 것 예측" (생성)

📚 참고: [embedding-and-prediction.md](faq/embedding-and-prediction.md), [training.md](faq/training.md)
