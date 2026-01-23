# BPE 토크나이저 학습

LLM은 "문자"가 아니라 **"토큰"**으로 학습합니다.
토크나이저는 텍스트를 정수 토큰 ID로 변환하는 핵심 컴포넌트입니다.

---

## 한 줄 요약

> **"자주 등장하는 문자 쌍을 병합하여 어휘 생성"**

---

## 1. 왜 직접 학습하는가?

### 기존 토크나이저의 문제

- GPT-2, LLaMA 등의 토크나이저는 영어 중심
- 한국어, 특수 용어, 구인 정보 형식이 비효율적으로 토큰화됨
- 예: "대한민국대사관" → 여러 개의 작은 토큰으로 쪼개짐

### 직접 학습의 장점

- 내 데이터에 자주 나오는 단어를 효율적으로 토큰화
- 한국어 + 영어 + 구인 정보 형식에 최적화
- 더 짧은 토큰 시퀀스 = 더 빠른 학습

---

## 2. BPE (Byte-Pair Encoding) 기본 원리

1. 모든 문자를 개별 토큰으로 시작
2. 가장 자주 등장하는 토큰 쌍을 찾음
3. 그 쌍을 새로운 토큰으로 병합
4. vocab_size에 도달할 때까지 반복

### 예시

```
원본: "low lower lowest"

Step 1: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', 'l', 'o', 'w', 'e', 's', 't']
Step 2: ['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', 'lo', 'w', 'e', 's', 't']  # 'l'+'o' 병합
Step 3: ['low', ' ', 'low', 'e', 'r', ' ', 'low', 'e', 's', 't']  # 'lo'+'w' 병합
...
```

---

## 3. BPE 학습 전체 흐름도

```
data/samples.txt
       │
       ▼
┌──────────────────────────────────┐
│  1. Whitespace Pre-tokenizer     │
│     "Google 채용" → ["Google", "채용"] │
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  2. 초기 어휘 생성                │
│     모든 문자를 개별 토큰으로      │
│     어휘 = ['G','o','g','l','e',...] │
│     (약 256개)                    │
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  3. BPE 반복 루프                 │
│     3-1. 가장 빈번한 쌍 찾기      │
│     3-2. 쌍을 새 토큰으로 병합    │
│     3-3. vocab_size 도달?        │
│          → No: 반복              │
│          → Yes: 종료             │
└──────────────────────────────────┘
       │
       ▼
data/tokenizer.json
```

### 흐름도 단계별 설명

#### 1단계: Pre-tokenizer (전처리)

텍스트를 공백 기준으로 1차 분리:

```
입력: "Google 채용 정보"
출력: ["Google", "채용", "정보"]
```

#### 2단계: 초기 어휘 생성

모든 개별 문자를 토큰으로 등록:

```
"Google" → ['G', 'o', 'o', 'g', 'l', 'e']
"채용"   → ['채', '용']

초기 어휘: ['G', 'o', 'g', 'l', 'e', '채', '용', ...]  (약 256개)
```

#### 3단계: BPE 반복 루프

| 단계 | 동작 | 예시 |
|------|------|------|
| 3-1 | 가장 빈번한 쌍 찾기 | `('o','o')` 100회 등장 |
| 3-2 | 쌍을 새 토큰으로 병합 | `'o'+'o'` → `'oo'` 추가 |
| 3-3 | vocab_size 도달 확인 | 24,000개? → No → 반복 |

**반복 횟수 계산:**
```
vocab_size = 24,000
초기 어휘 = 256개
병합 횟수 = 24,000 - 256 = 23,744회
```

---

## 4. 토크나이저 학습 스크립트

### scripts/train_tokenizer.py

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# 설정
IN_PATH = "data/samples.txt"
OUT_PATH = "data/tokenizer.json"
VOCAB_SIZE = 24000

# 토크나이저 초기화
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 트레이너 설정
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    show_progress=True,
)

# 학습 실행
tokenizer.train([IN_PATH], trainer=trainer)
tokenizer.save(OUT_PATH)
```

### Special Tokens 설명

| 토큰 | 용도 |
|------|------|
| `[PAD]` | 패딩 (배치 내 길이 맞춤) |
| `[UNK]` | 미지의 토큰 (vocabulary에 없는 토큰) |
| `[BOS]` | 문장 시작 (Beginning of Sentence) |
| `[EOS]` | 문장 끝 (End of Sentence) |

---

## 5. 바이너리 변환 스크립트

### scripts/build_bin_dataset.py

토크나이저 학습이 완료되면, 전체 텍스트를 토큰 ID 배열로 변환:

```python
import numpy as np
from tokenizers import Tokenizer

# 설정
TEXT_PATH = "data/samples.txt"
TOK_PATH = "data/tokenizer.json"
TRAIN_OUT = "data/train.bin"
VAL_OUT = "data/val.bin"
VAL_RATIO = 0.01

# 토크나이저 로드
tokenizer = Tokenizer.from_file(TOK_PATH)

# 텍스트 로드 및 토큰화
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

ids = tokenizer.encode(text).ids
arr = np.array(ids, dtype=np.uint16)

# Train/Val 분할
n = len(arr)
n_val = int(n * VAL_RATIO)
train = arr[:-n_val]
val = arr[-n_val:]

# 바이너리로 저장
train.tofile(TRAIN_OUT)
val.tofile(VAL_OUT)
```

---

## 6. 실행 방법

```bash
# 토크나이저 학습
uv run python scripts/train_tokenizer.py

# 바이너리 데이터셋 생성
uv run python scripts/build_bin_dataset.py
```

### 예상 출력

```
토크나이저 학습 시작...
입력 파일: data/samples.txt
어휘 크기: 24000
[00:00:30] Tokenizing ██████████████████████████████ 100%
토크나이저 저장 완료: data/tokenizer.json

학습 데이터: 14,850,000 토큰 → data/train.bin
검증 데이터: 150,000 토큰 → data/val.bin
```

---

## 7. vocab_size 선택 가이드

| vocab_size | 장점 | 단점 | 권장 상황 |
|------------|------|------|-----------|
| 8,000 | 빠른 학습 | 한국어 표현력 부족 | 영어 위주 데이터 |
| 16,000 | 균형 잡힘 | - | 일반적인 경우 |
| 24,000 | 한국어 표현력 좋음 | 학습 약간 느림 | 한국어 + 전문용어 |
| 32,000 | 표현력 최고 | 학습 느림, 메모리 많음 | 대용량 데이터 |

### 데이터 크기별 권장 vocab_size

| 데이터 크기 | 권장 vocab_size |
|------------|-----------------|
| 10KB 이하 | 500 ~ 1,000 |
| 10KB ~ 100KB | 1,000 ~ 4,000 |
| 100KB ~ 1MB | 4,000 ~ 16,000 |
| 1MB ~ 10MB | 16,000 ~ 32,000 |
| 10MB 이상 | 32,000 ~ 50,000 |

---

## 8. vocab_size 미달 문제

### 문제 상황

데이터가 너무 작으면 vocab_size에 도달하기 전에 병합이 끝날 수 있습니다.

```
목표: vocab_size = 24,000
데이터: "abc def ghi" (아주 작음)

결과: vocab_size = 10 (목표 미달)
```

### 해결 방법

1. **데이터 늘리기** (권장): 최소 1MB 이상
2. **vocab_size 줄이기**: 데이터에 맞게 조정

### 확인 방법

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/tokenizer.json")
actual_vocab_size = tokenizer.get_vocab_size()

print(f"목표 vocab_size: 24000")
print(f"실제 vocab_size: {actual_vocab_size}")

if actual_vocab_size < 24000:
    print("⚠️ 경고: 데이터가 부족하여 목표에 미달했습니다.")
```

---

## 9. 토크나이저 사용 예시

### 인코딩 (텍스트 → 토큰 ID)

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/tokenizer.json")

text = "안녕하세요. 연락처를 알려드립니다."
encoded = tokenizer.encode(text)

print(encoded.ids)     # [123, 456, 789, ...]
print(encoded.tokens)  # ['안녕하세요', '.', '연락처를', ...]
```

### 디코딩 (토큰 ID → 텍스트)

```python
ids = [123, 456, 789]
text = tokenizer.decode(ids)
print(text)  # "안녕하세요. 연락처를..."
```

---

## 요약

| 질문 | 답변 |
|------|------|
| BPE란? | 자주 등장하는 문자 쌍을 병합하여 어휘 생성 |
| 왜 직접 학습? | 내 데이터에 최적화된 토큰화 |
| 권장 vocab_size? | 한국어 포함 시 16,000~24,000 |
| 핵심 원칙? | vocab_size는 데이터 크기에 비례 |

---

## 관련 문서

- [vocab-size.md](vocab-size.md) - vocab_size 상세 설명
- [why-tokenize.md](why-tokenize.md) - 왜 토큰화가 필요한가
- [how-to-tokenize.md](how-to-tokenize.md) - 토큰화 방법
- [embedding-and-prediction.md](embedding-and-prediction.md) - 임베딩과 다음 토큰 예측
