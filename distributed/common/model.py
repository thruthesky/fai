# distributed/common/model.py
# ============================================================================
# GPT 모델 정의 (공통 모듈)
# ============================================================================
#
# 서버(병합), 워커(학습), 추론 등 모든 곳에서 사용하는 GPT 모델입니다.
# scripts/train_gpt.py의 모델 클래스를 추출하여 공통 모듈로 분리했습니다.
#
# 【주의】
# - 기존 체크포인트(checkpoints/ckpt.pt)와 state_dict 키 100% 호환
# - 레이어 이름(tok_emb, pos_emb, blocks, ln_f, head 등)을 변경하면 안 됨
# ============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# GPT 모델 설정
# ============================================================================
@dataclass
class GPTConfig:
    """
    GPT 모델 하이퍼파라미터 설정

    【역할】
    모델 아키텍처를 정의하는 설정값입니다.
    기존 CFG 클래스에서 모델 관련 필드만 추출했습니다.

    【기본값】 (M4 Mac 기준)
    - vocab_size: 24,000 (BPE 토크나이저 어휘 크기)
    - block_size: 256 (컨텍스트 길이)
    - n_layer: 6 (Transformer 블록 수)
    - n_head: 6 (Attention Head 수)
    - n_embd: 384 (임베딩 차원)
    - dropout: 0.1 (드롭아웃 비율)
    """

    vocab_size: int = 24000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

    def to_dict(self) -> dict:
        """JSONB 저장용 직렬화 (experiments 테이블의 config 컬럼에 저장)"""
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GPTConfig:
        """JSONB에서 복원"""
        return cls(
            vocab_size=d.get("vocab_size", 24000),
            block_size=d.get("block_size", 256),
            n_layer=d.get("n_layer", 6),
            n_head=d.get("n_head", 6),
            n_embd=d.get("n_embd", 384),
            dropout=d.get("dropout", 0.1),
        )


# ============================================================================
# Causal Self-Attention
# ============================================================================
class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention 레이어

    【역할】
    시퀀스 내 토큰들 간의 관계를 학습합니다.
    "Causal"은 현재 위치에서 미래 토큰을 볼 수 없도록 마스킹합니다.

    【수식】
    Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()

        assert n_embd % n_head == 0, (
            f"n_embd({n_embd})는 n_head({n_head})로 나누어 떨어져야 합니다"
        )

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Q, K, V를 한 번에 계산 (효율성)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)

        # 출력 프로젝션 (멀티헤드 결과를 n_embd 차원으로 변환)
        self.proj = nn.Linear(n_embd, n_embd)

        # 드롭아웃
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal Mask (하삼각 행렬: 미래 토큰을 볼 수 없도록 마스킹)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Q, K, V 계산
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        # 2. 멀티헤드 reshape: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Attention Score 계산
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. Causal Masking (미래 토큰 차단)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 5. Softmax → 가중치
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # 6. Value와 가중 합산
        y = att @ v

        # 7. 헤드 결합 및 출력 프로젝션
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))

        return y


# ============================================================================
# MLP (Feed Forward Network)
# ============================================================================
class MLP(nn.Module):
    """
    Feed Forward Network

    【구조】
    n_embd → Linear → 4*n_embd → GELU → Linear → n_embd → Dropout
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


# ============================================================================
# Transformer Block (Pre-LayerNorm)
# ============================================================================
class Block(nn.Module):
    """
    Transformer Block (Pre-LayerNorm 방식)

    【구조】
    x → LayerNorm → Attention → + (Residual)
      → LayerNorm → MLP       → + (Residual)
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================================
# GPT 모델 (Decoder-only Transformer)
# ============================================================================
class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) 모델

    【전체 흐름】
    토큰 ID → 토큰 임베딩 + 위치 임베딩 → N개 Transformer Block → LayerNorm → 출력 헤드

    【레이어 이름 (state_dict 키 호환)】
    - tok_emb: 토큰 임베딩
    - pos_emb: 위치 임베딩
    - drop: 입력 드롭아웃
    - blocks.{i}: i번째 Transformer Block
    - ln_f: 최종 LayerNorm
    - head: 출력 Linear (n_embd → vocab_size)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.block_size = config.block_size

        # 임베딩 레이어
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer 블록
        self.blocks = nn.ModuleList([
            Block(config.n_embd, config.n_head, config.dropout, config.block_size)
            for _ in range(config.n_layer)
        ])

        # 출력층
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # GPT-2 스타일 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """GPT-2 스타일 가중치 초기화: N(0, 0.02)"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
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
        assert T <= self.block_size, (
            f"시퀀스 길이({T})가 block_size({self.block_size})를 초과"
        )

        # 위치 인덱스
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        # 토큰 임베딩 + 위치 임베딩
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer 블록 통과
        for block in self.blocks:
            x = block(x)

        # 최종 LayerNorm + 출력 헤드
        x = self.ln_f(x)
        logits = self.head(x)

        # 손실 계산 (학습 시)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        텍스트 생성 (Autoregressive)

        Args:
            idx: 시작 토큰 (batch, seq_len)
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도 (낮을수록 결정적)
            top_k: Top-K 샘플링 (상위 K개 토큰에서만 샘플링)

        Returns:
            생성된 토큰 시퀀스 (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # block_size 제한
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Temperature 적용
            logits = logits / max(temperature, 1e-6)

            # Top-K 샘플링
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

        return idx

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: str = "cpu",
    ) -> tuple[GPT, dict]:
        """
        체크포인트에서 모델 복원

        Args:
            ckpt_path: 체크포인트 파일 경로
            device: 로드할 디바이스

        Returns:
            (model, checkpoint_dict) 튜플
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("cfg", {})

        # 체크포인트의 설정에서 GPTConfig 생성
        config = GPTConfig(
            vocab_size=cfg_dict.get("vocab_size", 24000),
            block_size=cfg_dict.get("block_size", 256),
            n_layer=cfg_dict.get("n_layer", 6),
            n_head=cfg_dict.get("n_head", 6),
            n_embd=cfg_dict.get("n_embd", 384),
            dropout=cfg_dict.get("dropout", 0.1),
        )

        model = cls(config)
        model.load_state_dict(ckpt["model"])
        model = model.to(device)

        return model, ckpt

    def param_count(self) -> int:
        """모델 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters())
