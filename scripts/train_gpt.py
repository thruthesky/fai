# scripts/train_gpt.py
# ============================================================================
# GPT 모델 학습 스크립트
# ============================================================================
#
# 【개요】
# Decoder-only Transformer (GPT) 모델을 처음부터 학습합니다.
# 바이너리 데이터셋(train.bin, val.bin)을 사용하여 다음 토큰 예측 학습을 수행합니다.
#
# 【입력/출력】
# - 입력: data/train.bin (학습 데이터)
#         data/val.bin (검증 데이터)
#         data/tokenizer.json (토크나이저)
# - 출력: checkpoints/ckpt.pt (학습된 모델)
#
# ============================================================================
# GPT 아키텍처 개요
# ============================================================================
#
# 【전체 구조】
#
#   입력 토큰 [I1, I2, I3, ..., In]
#              ↓
#   ┌──────────────────────────┐
#   │  Token Embedding         │  (vocab_size → n_embd)
#   │  + Position Embedding    │  (block_size → n_embd)
#   └────────────┬─────────────┘
#                ↓
#   ┌──────────────────────────┐
#   │  Dropout                 │
#   └────────────┬─────────────┘
#                ↓
#   ┌──────────────────────────┐
#   │  Transformer Block × N   │  ← N개 블록 반복
#   │  ├─ LayerNorm            │
#   │  ├─ Causal Self-Attention│
#   │  ├─ Residual Connection  │
#   │  ├─ LayerNorm            │
#   │  ├─ MLP (Feed Forward)   │
#   │  └─ Residual Connection  │
#   └────────────┬─────────────┘
#                ↓
#   ┌──────────────────────────┐
#   │  Final LayerNorm         │
#   └────────────┬─────────────┘
#                ↓
#   ┌──────────────────────────┐
#   │  Output Head (Linear)    │  (n_embd → vocab_size)
#   └────────────┬─────────────┘
#                ↓
#   출력 로짓 [O1, O2, O3, ..., On]
#
# ============================================================================
# 학습 방식: 다음 토큰 예측 (Next-token Prediction)
# ============================================================================
#
# 【핵심 개념】
# - 입력: [t1, t2, t3, ..., tn]
# - 타겟: [t2, t3, t4, ..., tn+1]  (한 칸 shift)
# - 목표: 각 위치에서 다음에 올 토큰을 예측
#
# 【손실 함수】
# - Cross Entropy Loss
# - 모델 출력(로짓)과 정답 토큰 간의 차이를 최소화
#
# 【중요】
# GPT는 벡터 유사도 검색이 아닌, 확률 분포에서 샘플링합니다!
# - 출력: vocab_size 크기의 확률 분포
# - 각 토큰이 다음에 올 확률을 예측
#
# ============================================================================
# 실행 방법
# ============================================================================
#
# uv run python scripts/train_gpt.py
#
# 체크포인트가 있으면 자동으로 학습을 재개합니다.
#
# ============================================================================

# ============================================
# MPS Fallback 설정 (torch import 전에 필수)
# ============================================
# Apple Silicon (M1/M2/M3/M4)에서 일부 연산이 MPS에서 지원되지 않을 경우
# 자동으로 CPU로 폴백하여 실행합니다.
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ============================================
# 라이브러리 임포트
# ============================================
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm


# ============================================================================
# 설정 (Configuration)
# ============================================================================
@dataclass
class CFG:
    """
    학습 설정을 담는 데이터클래스

    【파일 경로】
    - train_bin: 학습 데이터 (바이너리)
    - val_bin: 검증 데이터 (바이너리)
    - tok_path: BPE 토크나이저
    - out_dir: 체크포인트 저장 경로

    【모델 설정】
    - block_size: 컨텍스트 길이 (최대 입력 시퀀스 길이)
    - n_layer: Transformer 블록 수
    - n_head: Attention Head 수 (n_embd를 n_head로 나눈 값이 head_dim)
    - n_embd: 임베딩 차원 (토큰을 표현하는 벡터의 크기)
    - dropout: 드롭아웃 비율 (과적합 방지)

    【학습 설정】
    - batch_size: 한 번에 처리하는 샘플 수
    - lr: 학습률 (AdamW 옵티마이저)
    - max_steps: 총 학습 스텝 수
    - eval_interval: 평가 및 체크포인트 저장 간격
    - eval_iters: 평가 시 손실 계산에 사용할 배치 수
    - grad_clip: Gradient Clipping 값 (학습 안정화)

    【샘플링 설정】
    - sample_every_eval: 평가 시마다 샘플 텍스트 생성 여부
    - sample_max_new_tokens: 생성할 최대 토큰 수
    - temperature: 샘플링 온도 (높을수록 다양, 낮을수록 결정적)
    - top_k: Top-K 샘플링 (상위 K개 토큰에서만 샘플링)
    """
    # 파일 경로
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    tok_path: str = "data/tokenizer.json"
    out_dir: str = "checkpoints"

    # 모델 설정 (M4 Mac 기준 권장값)
    block_size: int = 256      # 컨텍스트 길이
    n_layer: int = 6           # Transformer 블록 수
    n_head: int = 6            # Attention Head 수
    n_embd: int = 384          # 임베딩 차원 (n_embd % n_head == 0 필수)
    dropout: float = 0.1       # 드롭아웃 비율

    # 학습 설정
    batch_size: int = 16       # 배치 크기
    lr: float = 3e-4           # 학습률
    max_steps: int = 5000      # 총 학습 스텝 (데이터가 적으므로 조정)
    eval_interval: int = 500   # 평가 간격
    eval_iters: int = 50       # 평가 시 반복 횟수
    grad_clip: float = 1.0     # Gradient Clipping

    # 샘플링 설정 (학습 중 생성 테스트)
    sample_every_eval: bool = True
    sample_max_new_tokens: int = 200
    temperature: float = 0.9
    top_k: int = 50


# 설정 인스턴스 생성
cfg = CFG()

# 체크포인트 디렉토리 생성
os.makedirs(cfg.out_dir, exist_ok=True)


# ============================================================================
# 디바이스 설정
# ============================================================================
def get_device() -> torch.device:
    """
    최적의 디바이스를 자동 선택합니다.

    우선순위:
    1. MPS (Apple Silicon GPU)
    2. CUDA (NVIDIA GPU)
    3. CPU (폴백)

    Returns:
        torch.device: 사용할 디바이스
    """
    if torch.backends.mps.is_available():
        print("디바이스: MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("디바이스: CUDA (NVIDIA GPU)")
        return torch.device("cuda")
    else:
        print("디바이스: CPU")
        return torch.device("cpu")


device = get_device()


# ============================================================================
# 데이터 로딩
# ============================================================================
# 메모리 매핑(Memory Mapping)으로 대용량 데이터를 효율적으로 로드합니다.
# - 전체 데이터를 RAM에 로드하지 않고, 필요한 부분만 읽음
# - uint16 타입 (0~65535, vocab_size < 65535이므로 충분)

print(f"\n데이터 로딩 중...")
train_data = np.memmap(cfg.train_bin, dtype=np.uint16, mode="r")
val_data = np.memmap(cfg.val_bin, dtype=np.uint16, mode="r")

print(f"  학습 데이터: {len(train_data):,} 토큰")
print(f"  검증 데이터: {len(val_data):,} 토큰")


# ============================================================================
# 토크나이저 로드
# ============================================================================
tokenizer = Tokenizer.from_file(cfg.tok_path)
vocab_size = tokenizer.get_vocab_size()
print(f"  어휘 크기: {vocab_size:,}")


# ============================================================================
# 배치 생성 함수
# ============================================================================
def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    학습/검증 데이터에서 랜덤 배치를 생성합니다.

    【동작 원리】
    1. 데이터에서 랜덤한 시작 위치 선택 (batch_size개)
    2. 각 위치에서 block_size 길이의 시퀀스 추출
    3. 입력(x)과 타겟(y) 생성 (y는 x를 한 칸 shift)

    【예시】
    데이터: [10, 20, 30, 40, 50, 60, 70]
    block_size: 3

    시작 위치 i=1인 경우:
    x = [20, 30, 40]  (입력)
    y = [30, 40, 50]  (타겟: 각 위치의 "다음 토큰")

    Args:
        split: "train" 또는 "val"

    Returns:
        x: 입력 토큰 (batch_size, block_size)
        y: 타겟 토큰 (batch_size, block_size)
    """
    data = train_data if split == "train" else val_data

    # 랜덤 시작 위치 선택
    # -block_size-1: 시퀀스 끝에서 y를 위한 여유 확보
    max_start = len(data) - cfg.block_size - 1
    ix = torch.randint(max_start, (cfg.batch_size,))

    # 입력과 타겟 생성
    x = torch.stack([
        torch.from_numpy(data[i:i+cfg.block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+1+cfg.block_size].astype(np.int64))
        for i in ix
    ])

    return x.to(device), y.to(device)


# ============================================================================
# 모델 컴포넌트
# ============================================================================

# ────────────────────────────────────────────────────────────────────────────
# Causal Self-Attention (인과적 자기 주의)
# ────────────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention 레이어

    【역할】
    - 시퀀스 내 토큰들 간의 관계를 학습합니다.
    - "Causal"은 현재 위치에서 미래 토큰을 볼 수 없도록 마스킹합니다.

    【동작 원리】
    1. Query(Q), Key(K), Value(V) 계산
       - Q: "내가 찾고 싶은 정보"
       - K: "내가 가진 정보의 인덱스"
       - V: "실제 정보"

    2. Attention Score 계산
       - score = (Q @ K^T) / sqrt(d_k)
       - 미래 위치는 -inf로 마스킹

    3. Softmax로 가중치 계산
       - 각 위치에 얼마나 주의를 기울일지 결정

    4. Value와 가중 합산
       - 주의를 기울인 정보를 결합

    【수식】
    Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V

    【Causal Mask 예시】
    block_size=4인 경우:
    [[1, 0, 0, 0],    ← 위치 0: 자기 자신만 볼 수 있음
     [1, 1, 0, 0],    ← 위치 1: 0, 1 볼 수 있음
     [1, 1, 1, 0],    ← 위치 2: 0, 1, 2 볼 수 있음
     [1, 1, 1, 1]]    ← 위치 3: 모두 볼 수 있음
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()

        # n_embd는 n_head로 나누어 떨어져야 함
        assert n_embd % n_head == 0, f"n_embd({n_embd})는 n_head({n_head})로 나누어 떨어져야 합니다"

        self.n_head = n_head
        self.head_dim = n_embd // n_head  # 각 헤드의 차원

        # Q, K, V를 한 번에 계산 (효율성)
        # 입력: (batch, seq_len, n_embd)
        # 출력: (batch, seq_len, 3 * n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)

        # 출력 프로젝션
        # 멀티헤드 결과를 다시 n_embd 차원으로 변환
        self.proj = nn.Linear(n_embd, n_embd)

        # 드롭아웃 (정규화)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal Mask 등록 (학습되지 않는 버퍼)
        # 하삼각 행렬: 미래 토큰을 볼 수 없도록 마스킹
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size))
                 .view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (batch, seq_len, n_embd)

        Returns:
            출력 텐서 (batch, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch, seq_len, n_embd

        # ────────────────────────────────────────────
        # 1. Q, K, V 계산
        # ────────────────────────────────────────────
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # 각각 (B, T, C)

        # ────────────────────────────────────────────
        # 2. 멀티헤드 reshape
        # ────────────────────────────────────────────
        # (B, T, C) → (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ────────────────────────────────────────────
        # 3. Attention Score 계산
        # ────────────────────────────────────────────
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) = (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ────────────────────────────────────────────
        # 4. Causal Masking (미래 토큰 차단)
        # ────────────────────────────────────────────
        # bias가 0인 위치(미래)를 -inf로 설정
        # softmax 후 이 위치들은 0이 됨
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # ────────────────────────────────────────────
        # 5. Softmax → 가중치로 변환
        # ────────────────────────────────────────────
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # ────────────────────────────────────────────
        # 6. Value와 가중 합산
        # ────────────────────────────────────────────
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) = (B, n_head, T, head_dim)
        y = att @ v

        # ────────────────────────────────────────────
        # 7. 헤드 결합 및 출력 프로젝션
        # ────────────────────────────────────────────
        # (B, n_head, T, head_dim) → (B, T, n_head, head_dim) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))

        return y


# ────────────────────────────────────────────────────────────────────────────
# MLP (Feed Forward Network)
# ────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """
    Feed Forward Network (MLP)

    【역할】
    - Attention이 찾은 관계를 바탕으로 정보를 비선형 변환합니다.
    - 각 위치에 독립적으로 적용됩니다.

    【구조】
    입력 (n_embd)
        ↓ Linear (확장)
    중간 (4 * n_embd)
        ↓ GELU (비선형 활성화)
    중간 (4 * n_embd)
        ↓ Linear (축소)
    출력 (n_embd)

    【GELU 활성화 함수】
    - ReLU보다 부드러운 활성화
    - GPT-2에서 사용된 활성화 함수
    - GELU(x) = x * Φ(x), Φ는 정규분포 CDF
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()

        # 확장: n_embd → 4*n_embd
        self.fc = nn.Linear(n_embd, 4 * n_embd)

        # 축소: 4*n_embd → n_embd
        self.proj = nn.Linear(4 * n_embd, n_embd)

        # 드롭아웃
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (batch, seq_len, n_embd)

        Returns:
            출력 텐서 (batch, seq_len, n_embd)
        """
        x = self.fc(x)        # 확장
        x = F.gelu(x)         # GELU 활성화
        x = self.proj(x)      # 축소
        x = self.drop(x)
        return x


# ────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ────────────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    """
    Transformer Block (Pre-LayerNorm 방식)

    【구조】
    입력 x
        │
        ├─────────────────┐
        │                 │ (Residual)
        ↓                 │
    LayerNorm             │
        ↓                 │
    Attention             │
        ↓                 │
        + ←───────────────┘
        │
        ├─────────────────┐
        │                 │ (Residual)
        ↓                 │
    LayerNorm             │
        ↓                 │
    MLP                   │
        ↓                 │
        + ←───────────────┘
        │
        ↓
    출력

    【Residual Connection (잔차 연결)】
    - x = x + layer(x) 형태
    - 기울기 소실 문제 해결
    - 깊은 네트워크 학습 가능

    【Pre-LayerNorm vs Post-LayerNorm】
    - Pre-LayerNorm: LayerNorm → Attention/MLP (GPT-2 스타일)
    - Post-LayerNorm: Attention/MLP → LayerNorm (원래 Transformer)
    - Pre-LayerNorm이 학습이 더 안정적
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()

        # Layer Normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # Attention과 MLP
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (batch, seq_len, n_embd)

        Returns:
            출력 텐서 (batch, seq_len, n_embd)
        """
        # Residual + Attention (Pre-LayerNorm)
        x = x + self.attn(self.ln1(x))

        # Residual + MLP (Pre-LayerNorm)
        x = x + self.mlp(self.ln2(x))

        return x


# ============================================================================
# GPT 모델
# ============================================================================
class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) 모델

    【전체 흐름】
    1. 토큰 ID 입력
    2. 토큰 임베딩 + 위치 임베딩
    3. N개의 Transformer Block 통과
    4. LayerNorm
    5. 출력 헤드 (vocab_size 차원의 로짓)

    【임베딩 (Embedding)】
    - 토큰 임베딩: 각 토큰 ID를 n_embd 차원 벡터로 변환
    - 위치 임베딩: 각 위치를 n_embd 차원 벡터로 변환
    - 최종 입력 = 토큰 임베딩 + 위치 임베딩

    【출력 헤드】
    - n_embd → vocab_size 변환
    - 각 위치에서 다음에 올 토큰의 확률 분포를 출력

    【파라미터 수 계산 (대략적)】
    - 토큰 임베딩: vocab_size × n_embd
    - 위치 임베딩: block_size × n_embd
    - 각 Block: ~12 × n_embd²
    - 출력 헤드: n_embd × vocab_size
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float
    ):
        super().__init__()

        self.block_size = block_size

        # ────────────────────────────────────────────
        # 임베딩 레이어
        # ────────────────────────────────────────────
        # 토큰 임베딩: 토큰 ID → 벡터
        # 예: ID 802 ("채용") → [0.1, -0.3, 0.5, ...] (384차원)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        # 위치 임베딩: 위치 → 벡터
        # 예: 위치 5 → [0.2, 0.1, -0.4, ...] (384차원)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        # 입력 드롭아웃
        self.drop = nn.Dropout(dropout)

        # ────────────────────────────────────────────
        # Transformer 블록
        # ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, block_size)
            for _ in range(n_layer)
        ])

        # ────────────────────────────────────────────
        # 출력층
        # ────────────────────────────────────────────
        # 최종 LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # 출력 헤드: n_embd → vocab_size
        # bias=False: 토큰 임베딩과 가중치 공유 시 사용 (여기선 공유 안 함)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # ────────────────────────────────────────────
        # 가중치 초기화
        # ────────────────────────────────────────────
        self.apply(self._init_weights)

        # 파라미터 수 출력
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  모델 파라미터 수: {n_params:,}")

    def _init_weights(self, module: nn.Module):
        """
        GPT-2 스타일 가중치 초기화

        - Linear, Embedding: N(0, 0.02) 정규분포
        - bias: 0으로 초기화
        - LayerNorm: 기본값 사용
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        순전파

        Args:
            idx: 입력 토큰 ID (batch, seq_len)
            targets: 타겟 토큰 ID (batch, seq_len), 학습 시 사용

        Returns:
            logits: 출력 로짓 (batch, seq_len, vocab_size)
            loss: Cross Entropy 손실 (targets가 주어진 경우)
        """
        B, T = idx.size()
        assert T <= self.block_size, f"시퀀스 길이({T})가 block_size({self.block_size})를 초과"

        # ────────────────────────────────────────────
        # 1. 위치 인덱스 생성
        # ────────────────────────────────────────────
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        # ────────────────────────────────────────────
        # 2. 토큰 임베딩 + 위치 임베딩
        # ────────────────────────────────────────────
        tok_emb = self.tok_emb(idx)  # (B, T, n_embd)
        pos_emb = self.pos_emb(pos)  # (1, T, n_embd) → 브로드캐스팅

        x = self.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # ────────────────────────────────────────────
        # 3. Transformer 블록 통과
        # ────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)

        # ────────────────────────────────────────────
        # 4. 최종 LayerNorm + 출력 헤드
        # ────────────────────────────────────────────
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        # ────────────────────────────────────────────
        # 5. 손실 계산 (학습 시)
        # ────────────────────────────────────────────
        loss = None
        if targets is not None:
            # Cross Entropy Loss
            # logits: (B*T, vocab_size)
            # targets: (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """
        텍스트 생성 (Autoregressive)

        【동작 원리】
        1. 현재까지의 토큰으로 다음 토큰 확률 예측
        2. 확률 분포에서 다음 토큰 샘플링
        3. 생성된 토큰을 시퀀스에 추가
        4. 반복

        【샘플링 방법】
        - temperature: 확률 분포의 날카로움 조절
          - 낮을수록(0.1): 결정적, 반복적
          - 높을수록(1.5): 다양하지만 품질 저하 가능

        - top_k: 상위 K개 토큰에서만 샘플링
          - 저확률 토큰을 제외하여 품질 향상

        Args:
            idx: 시작 토큰 (batch, seq_len)
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_k: Top-K 샘플링

        Returns:
            생성된 토큰 시퀀스 (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # block_size 제한 (컨텍스트 길이 초과 방지)
            idx_cond = idx[:, -self.block_size:]

            # 순전파
            logits, _ = self(idx_cond)

            # 마지막 위치의 로짓만 사용
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Temperature 적용
            logits = logits / max(temperature, 1e-6)

            # Top-K 샘플링
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Softmax → 확률 분포
            probs = F.softmax(logits, dim=-1)

            # 샘플링
            next_id = torch.multinomial(probs, num_samples=1)

            # 시퀀스에 추가
            idx = torch.cat((idx, next_id), dim=1)

        return idx


# ============================================================================
# 모델 생성
# ============================================================================
print(f"\n모델 생성 중...")
model = GPT(
    vocab_size=vocab_size,
    block_size=cfg.block_size,
    n_layer=cfg.n_layer,
    n_head=cfg.n_head,
    n_embd=cfg.n_embd,
    dropout=cfg.dropout
).to(device)


# ============================================================================
# 옵티마이저
# ============================================================================
# AdamW: 가중치 감쇠(weight decay)가 포함된 Adam
# - 적응형 학습률 (파라미터별로 다른 학습률)
# - 가중치 감쇠로 정규화 효과
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)


# ============================================================================
# 체크포인트 로드 (학습 재개)
# ============================================================================
ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
start_step = 0

if os.path.exists(ckpt_path):
    print(f"\n체크포인트 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    start_step = ckpt.get("step", 0)

    print(f"  스텝 {start_step}부터 재개")

    # 옵티마이저 상태를 디바이스로 이동
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# ============================================================================
# 평가 함수
# ============================================================================
@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    """
    학습/검증 손실을 추정합니다.

    【동작】
    1. 모델을 평가 모드로 전환 (드롭아웃 비활성화)
    2. 여러 배치에서 손실 계산
    3. 평균 손실 반환
    4. 모델을 학습 모드로 복원

    Returns:
        {"train": 학습 손실, "val": 검증 손실}
    """
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


# ============================================================================
# 샘플 생성 함수
# ============================================================================
def quick_sample() -> str:
    """
    학습 중 모델의 생성 능력을 테스트합니다.

    Returns:
        생성된 텍스트
    """
    model.eval()

    # 시작 토큰: [QUESTION]
    prompt = "[QUESTION]"
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # 생성
    out = model.generate(
        idx,
        max_new_tokens=cfg.sample_max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k
    )

    # 디코딩
    text = tokenizer.decode(out[0].tolist())

    model.train()
    return text


# ============================================================================
# 학습 루프
# ============================================================================
print(f"\n{'='*60}")
print(f"학습 시작")
print(f"{'='*60}")
print(f"  총 스텝: {cfg.max_steps:,}")
print(f"  배치 크기: {cfg.batch_size}")
print(f"  컨텍스트 길이: {cfg.block_size}")
print(f"  학습률: {cfg.lr}")
print(f"{'='*60}\n")

# 학습 모드 설정
model.train()

# tqdm 진행바
pbar = tqdm(range(start_step, cfg.max_steps), desc="학습 중")

for step in pbar:
    # ────────────────────────────────────────────────────────────────
    # 평가 및 체크포인트 저장
    # ────────────────────────────────────────────────────────────────
    if step > 0 and step % cfg.eval_interval == 0:
        losses = estimate_loss()

        print(f"\n[step {step:,}] train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")

        # 체크포인트 저장
        torch.save({
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
            "cfg": cfg.__dict__,
        }, ckpt_path)
        print(f"  체크포인트 저장: {ckpt_path}")

        # 샘플 생성
        if cfg.sample_every_eval:
            print(f"\n--- 샘플 생성 ---")
            sample_text = quick_sample()
            # 처음 500자만 출력
            print(sample_text[:500])
            if len(sample_text) > 500:
                print("...")
            print(f"--- 샘플 끝 ---\n")

    # ────────────────────────────────────────────────────────────────
    # 학습 스텝
    # ────────────────────────────────────────────────────────────────
    # 1. 배치 로드
    x, y = get_batch("train")

    # 2. 순전파
    logits, loss = model(x, y)

    # 3. 역전파
    optimizer.zero_grad(set_to_none=True)  # 메모리 효율적
    loss.backward()

    # 4. Gradient Clipping (학습 안정화)
    # 기울기가 너무 커지면 잘라냄 (exploding gradient 방지)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    # 5. 가중치 업데이트
    optimizer.step()

    # 진행바 업데이트
    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

# ============================================================================
# 최종 저장
# ============================================================================
print(f"\n{'='*60}")
print(f"학습 완료!")
print(f"{'='*60}")

# 최종 체크포인트 저장
torch.save({
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
    "step": cfg.max_steps,
    "cfg": cfg.__dict__,
}, ckpt_path)
print(f"최종 체크포인트 저장: {ckpt_path}")

# 최종 손실 출력
final_losses = estimate_loss()
print(f"\n최종 손실:")
print(f"  학습 손실: {final_losses['train']:.4f}")
print(f"  검증 손실: {final_losses['val']:.4f}")

# 최종 샘플 생성
print(f"\n--- 최종 샘플 생성 ---")
print(quick_sample()[:1000])
print(f"--- 끝 ---")
