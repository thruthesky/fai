# 텍스트 생성

학습된 모델로 텍스트를 생성합니다.
JAI는 **요약/정리형 구인 정보**를 생성하도록 설계되었습니다.

---

## 한 줄 요약

> **"프롬프트 → 토큰화 → 다음 토큰 예측 반복 → 디코딩"**

---

## 1. 생성 원리

GPT는 **Autoregressive** 방식으로 텍스트를 생성합니다:

```
입력: "안녕"
→ 모델이 다음 토큰 예측: "하"
→ 입력 업데이트: "안녕하"
→ 모델이 다음 토큰 예측: "세"
→ 입력 업데이트: "안녕하세"
→ ... 반복
```

---

## 2. 프롬프트 형식 (중요!)

학습 데이터와 **동일한 형식**으로 프롬프트를 작성해야 좋은 결과가 나옵니다.

### 권장 프롬프트 템플릿

```
[QUESTION]
{질문 내용}
[/QUESTION]

[ANSWER]
요약:
-
```

`[ANSWER] 요약: -` 까지 입력하면, 모델이 자연스럽게 이어서 작성합니다.

---

## 3. 생성 함수

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    텍스트 생성 (Autoregressive)

    Args:
        idx: 시작 토큰 ID (B, T)
        max_new_tokens: 생성할 토큰 수
        temperature: 샘플링 온도 (높을수록 다양함)
        top_k: Top-K 샘플링 (None이면 전체 vocab 사용)

    Returns:
        idx: 생성된 토큰 시퀀스
    """
    for _ in range(max_new_tokens):
        # 컨텍스트 길이 제한
        idx_cond = idx[:, -self.block_size:]

        # 순전파
        logits = self(idx_cond)

        # 마지막 토큰의 로짓만 사용
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        # Top-K 샘플링
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        # 확률 분포로 변환
        probs = F.softmax(logits, dim=-1)

        # 샘플링
        next_id = torch.multinomial(probs, num_samples=1)

        # 시퀀스에 추가
        idx = torch.cat((idx, next_id), dim=1)

    return idx
```

---

## 4. 샘플링 파라미터 이해

### Temperature

출력의 **다양성/창의성**을 조절합니다.

| 값 | 효과 |
|----|------|
| 0.1~0.5 | 보수적, 예측 가능, 반복적 |
| 0.7~0.9 | 균형 잡힌 다양성 (권장) |
| 1.0+ | 창의적, 예측 불가, 가끔 엉뚱함 |

```python
# 보수적 (정확한 정보 원할 때)
generate_text(prompt, temperature=0.5)

# 창의적 (다양한 표현 원할 때)
generate_text(prompt, temperature=1.2)
```

### Top-K

상위 K개의 토큰만 샘플링 후보로 사용합니다.

| 값 | 효과 |
|----|------|
| 10~20 | 매우 보수적, 반복 위험 |
| 40~60 | 균형 (권장) |
| 100+ | 다양하지만 품질 저하 위험 |

```python
# 보수적
generate_text(prompt, top_k=20)

# 다양한
generate_text(prompt, top_k=100)
```

### 권장 조합

| 용도 | temperature | top_k |
|------|-------------|-------|
| 정확한 구인 정보 | 0.7 | 30 |
| 일반적 사용 | 0.9 | 50 |
| 창의적 글쓰기 | 1.1 | 100 |

---

## 5. 생성 스크립트 예시

```python
# scripts/generate.py
import torch
from tokenizers import Tokenizer

# 토크나이저 & 모델 로드
tokenizer = Tokenizer.from_file("data/tokenizer.json")
model = GPT(...).to(device)
ckpt = torch.load("checkpoints/ckpt.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

def generate_text(prompt, max_new_tokens=400, temperature=0.9, top_k=50):
    """
    프롬프트에 이어서 텍스트 생성
    """
    # 토큰화
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

    # 생성
    y = model.generate(x, max_new_tokens=max_new_tokens,
                       temperature=temperature, top_k=top_k)

    # 디코딩
    return tokenizer.decode(y[0].tolist())

# 사용 예시
prompt = """[QUESTION]
미국 주요 도시의 한국 대사관/영사관 연락처를 정리해줘.
[/QUESTION]

[ANSWER]
요약:
-"""

result = generate_text(prompt, max_new_tokens=400, temperature=0.9, top_k=50)
print(result)
```

---

## 6. generate.py 스크립트 상세 사용법

`scripts/generate.py`는 학습된 모델로 텍스트를 생성하는 CLI 도구입니다.

### 6.1 CLI 옵션

| 옵션 | 단축 | 기본값 | 설명 |
|------|------|--------|------|
| `--prompt` | `-p` | None | 시작 프롬프트 (미지정 시 대화형 모드) |
| `--max_tokens` | `-m` | 500 | 생성할 최대 토큰 수 |
| `--temperature` | `-t` | 0.9 | 샘플링 온도 (0.1~1.5) |
| `--top_k` | `-k` | 50 | Top-K 샘플링 (10~200) |
| `--interactive` | `-i` | False | 대화형 모드 강제 실행 |
| `--no-streaming` | - | False | 스트리밍 비활성화 (한번에 출력) |

### 6.2 기본 실행 (대화형 모드)

```bash
uv run python scripts/generate.py
```

프롬프트 입력창이 나타나며, 텍스트를 입력하면 즉시 생성이 시작됩니다.
`quit`, `exit`, `q` 중 하나를 입력하면 종료됩니다.

```
============================================================
JAI 텍스트 생성기 (대화형 모드)
============================================================
프롬프트를 입력하면 텍스트를 생성합니다.
스트리밍 모드: 토큰이 생성될 때마다 즉시 출력
종료: 'quit' 또는 'exit' 입력
============================================================

프롬프트> [QUESTION]
싱가포르 개발자 채용 정보
[/QUESTION]

[DOC]
```

### 6.3 단일 프롬프트 실행

```bash
# 기본 스트리밍 모드 (토큰이 생성될 때마다 즉시 출력)
uv run python scripts/generate.py --prompt "[QUESTION]
싱가포르 개발자 채용
[/QUESTION]

[DOC]"

# 스트리밍 비활성화 (생성 완료 후 한번에 출력)
uv run python scripts/generate.py --prompt "..." --no-streaming
```

### 6.4 파라미터 조절

```bash
# 보수적 생성 (정확한 정보)
uv run python scripts/generate.py --prompt "..." --temperature 0.5 --top_k 30

# 창의적 생성 (다양한 표현)
uv run python scripts/generate.py --prompt "..." --temperature 1.1 --top_k 100

# 긴 응답 생성
uv run python scripts/generate.py --prompt "..." --max_tokens 1000
```

### 6.5 스트리밍 모드 (기본)

스트리밍 모드에서는 토큰이 생성될 때마다 즉시 화면에 출력됩니다.
마치 ChatGPT처럼 글자가 하나씩 나타나는 효과를 볼 수 있습니다.

**특징:**
- 토큰 생성 즉시 출력 (버퍼 없음)
- `[/ANSWER]` 태그 감지 시 자동 종료
- 사용자 경험 향상 (대기 시간 체감 감소)

### 6.6 종료 조건

생성은 다음 조건 중 하나가 만족되면 종료됩니다:

1. **max_tokens 도달**: 지정한 최대 토큰 수에 도달
2. **종료 패턴 감지**: `[/ANSWER]` 태그가 출력되면 자동 종료
3. **EOS 토큰**: `[EOS]` 토큰이 생성되면 종료

### 6.7 권장 프롬프트 형식

학습 데이터와 동일한 형식을 사용해야 좋은 결과가 나옵니다:

```
[QUESTION]
{질문 내용}
[/QUESTION]

[DOC]
{참고 정보 - 선택사항}
[/DOC]

[ANSWER]
요약:
-
```

**팁**: `[ANSWER] 요약: -` 까지 입력하면 모델이 자연스럽게 이어서 작성합니다.

---

## 7. 생성 결과 예시

### 입력 프롬프트

```
[QUESTION]
일본 도쿄 주재 한국 대사관 연락처와 업무 안내를 정리해줘.
[/QUESTION]

[ANSWER]
요약:
-
```

### 기대 출력

```
[ANSWER]
요약:
- 주일 대한민국 대사관은 도쿄 미나토구에 위치
- 영사과 업무는 별도 번호로 문의
- 평일 09:00-17:00 운영

체크리스트:
- 해야 할 일:
  - (1) 방문 전 전화 예약
  - (2) 필요 서류 사전 확인
- 준비물:
  - (1) 여권 원본
  - (2) 신청서
- 주의사항:
  - (1) 공휴일 휴무

연락처(공공정보):
- 주일 대한민국 대사관
  - TEL: +81-3-3452-7611
  - ADDR: 東京都港区南麻布1-2-5
  - WEB: https://overseas.mofa.go.kr/jp-ko/

상세 설명:
주일 대한민국 대사관은 도쿄 미나토구에 위치하고 있습니다...
[/ANSWER]
```

---

## 8. 생성 품질 향상 팁

### 1. 학습 데이터 형식과 동일하게 프롬프트 작성

모델은 학습 데이터에서 본 패턴을 따라합니다.
`[QUESTION]...[/QUESTION]` 형식을 일관되게 사용하세요.

### 2. Temperature 조절

- 반복이 심하면 temperature를 높이세요 (0.9 → 1.1)
- 엉뚱한 출력이면 temperature를 낮추세요 (0.9 → 0.7)

### 3. 더 많은 컨텍스트 제공

`[DOC]` 섹션에 참고 정보를 넣으면 더 정확한 답변이 생성됩니다.

---

## 요약

| 질문 | 답변 |
|------|------|
| 생성 방식? | Autoregressive (한 토큰씩 예측) |
| 권장 temperature? | 0.7~0.9 |
| 권장 top_k? | 40~60 |
| 프롬프트 형식? | 학습 데이터와 동일하게 |

---

## 관련 문서

- [training.md](training.md) - 모델 학습
- [server.md](server.md) - API 서버 구축
- [core-concepts.md](core-concepts.md) - 핵심 개념 9가지
