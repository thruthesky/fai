# GPT 모델 아키텍처

JAI는 **Decoder-only Transformer** 구조를 사용합니다.
이는 GPT, LLaMA, Claude 등 현대 LLM의 기본 구조입니다.

---

## 한 줄 요약

> **"Token Embedding → N개 Transformer Block → Output Head"**

---

## 1. 전체 구조도

```
입력 토큰 [I1, I2, I3, ..., In]
           ↓
    ┌──────────────┐
    │  Token       │  토큰 → 벡터 변환
    │  Embedding   │  (vocab_size, n_embd)
    └──────┬───────┘
           │
           + ← Position Embedding (위치 정보 추가)
           │
    ┌──────┴───────┐
    │   Dropout    │
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │              │
    │   Block ×N   │  ← Transformer Block 반복
    │              │
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │  LayerNorm   │  최종 정규화
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │  Linear      │  (n_embd → vocab_size)
    │  (Head)      │
    └──────┬───────┘
           │
           ↓
    출력 로짓 [O1, O2, O3, ..., On]
```

---

## 2. Transformer Block 구조

```
입력 x
    │
    ├──────────────────┐
    │                  │ (Residual Connection)
    ↓                  │
┌────────────┐         │
│ LayerNorm  │         │
└─────┬──────┘         │
      │                │
      ↓                │
┌─────────────────┐    │
│ Causal Self     │    │
│ Attention       │    │
└─────┬───────────┘    │
      │                │
      + ←──────────────┘
      │
      ├──────────────────┐
      │                  │ (Residual Connection)
      ↓                  │
┌────────────┐           │
│ LayerNorm  │           │
└─────┬──────┘           │
      │                  │
      ↓                  │
┌─────────────────┐      │
│ Feed Forward    │      │
│ (MLP)           │      │
└─────┬───────────┘      │
      │                  │
      + ←────────────────┘
      │
      ↓
    출력
```

---

## 3. 권장 하이퍼파라미터 (M4 기준)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| vocab_size | 24,000 | 토크나이저와 맞춤 |
| block_size | 256 | 컨텍스트 길이 |
| n_layer | 6 | Transformer 블록 수 |
| n_head | 6 | Attention Head 수 |
| n_embd | 384 | 임베딩 차원 |
| dropout | 0.1 | 드롭아웃 비율 |

기본 설정(6 layer, 384 dim)으로 약 **28M** 파라미터입니다.

---

## 4. 핵심 컴포넌트

### 4.1 Causal Self-Attention

```python
class CausalSelfAttention(nn.Module):
    """
    Causal (인과적) Self-Attention
    - 현재 위치에서 미래의 토큰을 볼 수 없도록 마스킹
    - GPT의 핵심 메커니즘
    """
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Q, K, V를 한 번에 계산 (효율성)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal Mask (하삼각 행렬)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size))
                 .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Q, K, V 계산
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        # 멀티헤드로 reshape
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention Score: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal Masking: 미래 위치를 -inf로 설정
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Softmax → Dropout → Value 곱하기
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        # 헤드 합치기 → 출력 프로젝션
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))

        return y
```

**핵심 개념:**
- **Q, K, V**: Query, Key, Value 행렬
- **Causal Mask**: 미래 토큰을 볼 수 없게 하는 하삼각 행렬
- **Multi-head**: 여러 시각에서 attention을 계산

### 4.2 Feed Forward Network (MLP)

```python
class MLP(nn.Module):
    """
    Feed Forward Network
    - 2층 MLP with GELU 활성화
    - 중간 차원은 4배로 확장
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)    # 확장
        self.proj = nn.Linear(4 * n_embd, n_embd)  # 축소
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)       # 확장
        x = F.gelu(x)        # GELU 활성화
        x = self.proj(x)     # 축소
        x = self.drop(x)
        return x
```

**핵심 개념:**
- **4배 확장**: n_embd → 4*n_embd → n_embd
- **GELU 활성화**: ReLU보다 부드러운 비선형 변환

### 4.3 Transformer Block

```python
class Block(nn.Module):
    """
    Transformer Block
    - Pre-LayerNorm 구조 (GPT-2 스타일)
    - Residual Connection 포함
    """
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)  # Attention 전 정규화
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)  # MLP 전 정규화
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Residual + Self-Attention
        x = x + self.mlp(self.ln2(x))   # Residual + MLP
        return x
```

**핵심 개념:**
- **Pre-LayerNorm**: LayerNorm을 먼저 적용 (학습 안정화)
- **Residual Connection**: 입력을 출력에 더함 (기울기 소실 방지)

### 4.4 전체 GPT 모델

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size

        # 임베딩 레이어
        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # 토큰 임베딩
        self.pos_emb = nn.Embedding(block_size, n_embd)  # 위치 임베딩
        self.drop = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, block_size)
            for _ in range(n_layer)
        ])

        # 최종 레이어
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        # 토큰 임베딩 + 위치 임베딩
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        # Transformer Blocks 통과
        for block in self.blocks:
            x = block(x)

        # 최종 정규화 + 출력 헤드
        x = self.ln_f(x)
        logits = self.head(x)

        # 손실 계산 (학습 시)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
```

---

## 5. 모델 파라미터 수 계산

```python
def count_parameters(model):
    """모델의 학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = GPT(
    vocab_size=24000,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1
)

params = count_parameters(model)
print(f"총 파라미터 수: {params:,}")  # 약 28M
```

---

## 6. JAI가 직접 구현하는 것

JAI 프로젝트에서는 **외부 GPT 라이브러리를 사용하지 않고** PyTorch의 기본 nn.* 레이어로 직접 구현합니다.

| 사용하는 것 | 직접 구현하는 것 |
|-------------|------------------|
| `nn.Embedding` | GPT 클래스 전체 |
| `nn.Linear` | CausalSelfAttention |
| `nn.LayerNorm` | MLP |
| `nn.Dropout` | Block |

---

## 요약

| 질문 | 답변 |
|------|------|
| GPT 구조? | Decoder-only Transformer |
| 핵심 컴포넌트? | Token Embedding + N×Block + Output Head |
| Block 구성? | LayerNorm + Attention + LayerNorm + MLP |
| 외부 라이브러리? | PyTorch nn.* 기본 레이어만 사용 |

---

## 관련 문서

- [embedding-and-prediction.md](embedding-and-prediction.md) - 임베딩과 다음 토큰 예측
- [training.md](training.md) - 모델 학습 방법
- [core-concepts.md](core-concepts.md) - 핵심 개념 9가지
