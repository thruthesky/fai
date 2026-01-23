# 모델 학습

GPT 모델의 학습 목표는 단 하나입니다:

> **"앞의 토큰들을 보고 다음 토큰을 맞춰라" (Next-token prediction)**

---

## 한 줄 요약

> **"다음 토큰 예측 → CrossEntropy 손실 → 역전파 → 가중치 업데이트"**

---

## 1. 학습 설정 (Config)

```python
from dataclasses import dataclass

@dataclass
class CFG:
    """학습 설정"""
    # 파일 경로
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    tok_path: str = "data/tokenizer.json"
    out_dir: str = "checkpoints"

    # 모델 설정
    block_size: int = 256      # 컨텍스트 길이
    n_layer: int = 6           # Transformer 레이어 수
    n_head: int = 6            # Attention Head 수
    n_embd: int = 384          # 임베딩 차원
    dropout: float = 0.1       # 드롭아웃 비율

    # 학습 설정
    batch_size: int = 16       # 배치 크기
    lr: float = 3e-4           # 학습률
    max_steps: int = 20000     # 총 학습 스텝
    eval_interval: int = 500   # 평가 간격
    eval_iters: int = 100      # 평가 시 반복 횟수
    grad_clip: float = 1.0     # Gradient Clipping

    # 샘플링 설정 (디버그용)
    sample_every_eval: bool = True
    sample_max_new_tokens: int = 250
    temperature: float = 0.9
    top_k: int = 50
```

---

## 2. 데이터 로더

```python
import numpy as np
import torch

# 메모리 매핑으로 대용량 데이터 효율적으로 처리
train_data = np.memmap(cfg.train_bin, dtype=np.uint16, mode="r")
val_data = np.memmap(cfg.val_bin, dtype=np.uint16, mode="r")

def get_batch(split: str):
    """
    랜덤 배치 생성

    Args:
        split: "train" 또는 "val"

    Returns:
        x: 입력 토큰 (batch_size, block_size)
        y: 타겟 토큰 (batch_size, block_size) - x를 한 칸 shift
    """
    data = train_data if split == "train" else val_data

    # 랜덤 시작 위치 선택
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))

    # 입력과 타겟 생성
    x = torch.stack([
        torch.from_numpy((data[i:i+cfg.block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i+1:i+1+cfg.block_size]).astype(np.int64))
        for i in ix
    ])

    return x.to(device), y.to(device)
```

**핵심 개념:**
- **입력 x**: 토큰 시퀀스 [t1, t2, t3, ..., tn]
- **타겟 y**: x를 한 칸 shift한 시퀀스 [t2, t3, t4, ..., tn+1]
- 모델은 x를 보고 y를 예측하도록 학습

---

## 3. 학습 루프

```python
from tqdm import tqdm

model = GPT(...).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

pbar = tqdm(range(cfg.max_steps))
for step in pbar:
    # 배치 가져오기
    x, y = get_batch("train")

    # 순전파
    logits, loss = model(x, y)

    # 역전파
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient Clipping (학습 안정화)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    # 가중치 업데이트
    optimizer.step()

    # 진행률 업데이트
    if step % 50 == 0:
        pbar.set_description(f"step {step} loss {loss.item():.4f}")
```

---

## 4. 평가 함수

```python
@torch.no_grad()
def estimate_loss():
    """학습/검증 손실 추정"""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
```

---

## 5. 체크포인트 저장/로드

### 저장

```python
ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")

torch.save({
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
    "step": step,
    "cfg": cfg.__dict__,
}, ckpt_path)
```

### 로드 (학습 재개)

```python
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    start_step = ckpt.get("step", 0)
    print(f"스텝 {start_step}부터 재개")
```

---

## 6. 실행 방법

```bash
uv run python scripts/train_gpt.py
```

### 예상 출력

```
사용 디바이스: mps
어휘 크기: 24000
학습 데이터: 14,850,000 토큰
검증 데이터: 150,000 토큰
모델 파라미터 수: 28,123,456
학습 시작...
step 0 loss 10.1234: 100%|██████████████| 50/20000
step 50 loss 8.5432: 100%|██████████████| 100/20000
...
step 500 train_loss=5.2341 val_loss=5.3456
체크포인트 저장: checkpoints/ckpt.pt
```

---

## 7. 손실(Loss) 변화 해석

| 손실 값 | 의미 |
|---------|------|
| 10+ | 초기 상태, 랜덤 |
| 5~7 | 문법 패턴 학습 시작 |
| 3~5 | 형식 학습, 구인 정보 패턴 인식 |
| 2~3 | 의미 있는 텍스트 생성 |
| <2 | 좋은 품질 (과적합 주의) |

---

## 8. 학습 팁

### 손실이 줄지 않을 때

```python
# 학습률 낮추기
cfg.lr = 1e-4  # 3e-4 → 1e-4

# 배치 크기 늘리기 (메모리 허용 시)
cfg.batch_size = 32
```

### 메모리 부족 (OOM)

```python
# 배치 크기 줄이기
cfg.batch_size = 8  # 16 → 8

# 컨텍스트 길이 줄이기
cfg.block_size = 128  # 256 → 128

# 모델 크기 줄이기
cfg.n_layer = 4   # 6 → 4
cfg.n_embd = 256  # 384 → 256
```

### 학습 중단 후 재개

체크포인트가 자동으로 저장되므로, 스크립트를 다시 실행하면 자동으로 재개됩니다.

```bash
uv run python scripts/train_gpt.py
```

---

## 9. MPS Fallback 설정

Mac에서 학습 시 일부 연산이 MPS에서 지원되지 않을 수 있습니다:

```python
import os
# MPS fallback 설정 (torch import 전에 반드시 설정)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
```

---

## 요약

| 질문 | 답변 |
|------|------|
| 학습 목표? | 다음 토큰 예측 (Next-token prediction) |
| 손실 함수? | CrossEntropy |
| 옵티마이저? | AdamW |
| 권장 학습률? | 3e-4 (손실 안 줄면 1e-4) |
| 체크포인트? | 자동 저장, 재실행 시 자동 재개 |

---

## 관련 문서

- [model-architecture.md](model-architecture.md) - GPT 모델 아키텍처
- [generation.md](generation.md) - 텍스트 생성
- [troubleshooting.md](troubleshooting.md) - 트러블슈팅
