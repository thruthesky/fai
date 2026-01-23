# scripts/generate.py
# ============================================================================
# 텍스트 생성 스크립트
# ============================================================================
#
# 【개요】
# 학습된 GPT 모델을 사용하여 텍스트를 생성합니다.
# 프롬프트(시작 텍스트)를 입력하면 모델이 이어서 텍스트를 생성합니다.
#
# 【입력/출력】
# - 입력: checkpoints/ckpt.pt (학습된 모델)
#         data/tokenizer.json (토크나이저)
#         프롬프트 텍스트
# - 출력: 생성된 텍스트
#
# ============================================================================
# 텍스트 생성 원리 (Autoregressive Generation)
# ============================================================================
#
# 【핵심 개념】
# GPT는 "다음에 올 토큰"을 예측합니다.
# 한 토큰씩 생성하고, 생성된 토큰을 다시 입력으로 사용하여 반복합니다.
#
# 【예시】
# 프롬프트: "[QUESTION]"
#
# 1단계: "[QUESTION]" → 다음 토큰 예측 → "싱가포르"
# 2단계: "[QUESTION] 싱가포르" → 다음 토큰 예측 → "에서"
# 3단계: "[QUESTION] 싱가포르에서" → 다음 토큰 예측 → "개발자"
# ...
# 반복하여 전체 텍스트 생성
#
# 【중요】
# GPT는 벡터 유사도 검색이 아닙니다!
# 각 단계에서 vocab_size 크기의 확률 분포를 생성하고,
# 그 분포에서 다음 토큰을 샘플링합니다.
#
# ============================================================================
# 실행 방법
# ============================================================================
#
# 기본 실행 (대화형):
#   uv run python scripts/generate.py
#
# 프롬프트 지정:
#   uv run python scripts/generate.py --prompt "[QUESTION]\n싱가포르 개발자 채용"
#
# 생성 옵션:
#   uv run python scripts/generate.py --max_tokens 300 --temperature 0.8
#
# ============================================================================

# ============================================
# MPS Fallback 설정 (torch import 전에 필수)
# ============================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ============================================
# 라이브러리 임포트
# ============================================
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


# ============================================================================
# 설정
# ============================================================================
CKPT_PATH = "checkpoints/ckpt.pt"      # 체크포인트 경로
TOK_PATH = "data/tokenizer.json"       # 토크나이저 경로


# ============================================================================
# 모델 정의 (train_gpt.py와 동일)
# ============================================================================
# 참고: 실제 프로젝트에서는 모델 클래스를 별도 모듈로 분리하는 것이 좋습니다.

class CausalSelfAttention(nn.Module):
    """Causal Self-Attention 레이어"""

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size))
                 .view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    """Feed Forward Network"""

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


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT 모델"""

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

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, block_size)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.block_size

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        stop_tokens: list = None,
        callback: callable = None
    ) -> torch.Tensor:
        """
        텍스트 생성 (Autoregressive)

        【샘플링 방법】
        1. Temperature: 확률 분포의 날카로움 조절
           - 낮을수록 (0.1~0.5): 결정적, 반복적
           - 높을수록 (1.0~1.5): 다양하지만 품질 저하 가능
           - 권장: 0.7~1.0

        2. Top-K: 상위 K개 토큰에서만 샘플링
           - 저확률 토큰을 제외하여 품질 향상
           - 권장: 40~100

        Args:
            idx: 시작 토큰 (batch, seq_len)
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_k: Top-K 샘플링
            stop_tokens: 생성을 중단할 토큰 ID 리스트
            callback: 토큰 생성 시 호출할 콜백 함수 (스트리밍용)

        Returns:
            생성된 토큰 시퀀스
        """
        for _ in range(max_new_tokens):
            # 컨텍스트 길이 제한
            idx_cond = idx[:, -self.block_size:]

            # 순전파
            logits = self(idx_cond)

            # 마지막 위치의 로짓만 사용
            logits = logits[:, -1, :]

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

            # 콜백 호출 (스트리밍 출력)
            if callback is not None:
                callback(next_id.item())

            # 종료 토큰 체크
            if stop_tokens is not None and next_id.item() in stop_tokens:
                break

        return idx


# ============================================================================
# 디바이스 설정
# ============================================================================
def get_device() -> torch.device:
    """최적의 디바이스 자동 선택"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ============================================================================
# 모델 로드
# ============================================================================
def load_model(device: torch.device) -> tuple[GPT, Tokenizer]:
    """
    학습된 모델과 토크나이저를 로드합니다.

    Returns:
        model: 학습된 GPT 모델
        tokenizer: BPE 토크나이저
    """
    # 체크포인트 확인
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {CKPT_PATH}\n"
            f"먼저 'uv run python scripts/train_gpt.py'로 모델을 학습하세요."
        )

    # 토크나이저 로드
    tokenizer = Tokenizer.from_file(TOK_PATH)
    vocab_size = tokenizer.get_vocab_size()

    # 체크포인트 로드
    print(f"체크포인트 로드: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # 설정 복원
    cfg = ckpt.get("cfg", {})
    block_size = cfg.get("block_size", 256)
    n_layer = cfg.get("n_layer", 6)
    n_head = cfg.get("n_head", 6)
    n_embd = cfg.get("n_embd", 384)
    dropout = cfg.get("dropout", 0.0)  # 추론 시 드롭아웃 비활성화

    print(f"  block_size: {block_size}")
    print(f"  n_layer: {n_layer}")
    print(f"  n_head: {n_head}")
    print(f"  n_embd: {n_embd}")
    print(f"  vocab_size: {vocab_size}")

    # 모델 생성
    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0  # 추론 시 드롭아웃 비활성화
    )

    # 가중치 로드
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()  # 평가 모드

    step = ckpt.get("step", "?")
    print(f"  학습 스텝: {step}")

    return model, tokenizer


# ============================================================================
# 텍스트 생성 함수
# ============================================================================
def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 500,
    temperature: float = 0.9,
    top_k: int = 50,
    device: torch.device = None,
    streaming: bool = False
) -> str:
    """
    프롬프트를 기반으로 텍스트를 생성합니다.

    Args:
        model: GPT 모델
        tokenizer: 토크나이저
        prompt: 시작 텍스트
        max_new_tokens: 생성할 최대 토큰 수
        temperature: 샘플링 온도 (높을수록 다양)
        top_k: Top-K 샘플링
        device: 디바이스
        streaming: 스트리밍 출력 여부

    Returns:
        생성된 텍스트
    """
    import sys

    # 프롬프트 토큰화
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # 종료 토큰 찾기: [/ANSWER], [EOS] 등
    # 토크나이저에서 해당 토큰의 ID를 찾음
    stop_tokens = []
    vocab = tokenizer.get_vocab()

    # [EOS] 토큰
    if "[EOS]" in vocab:
        stop_tokens.append(vocab["[EOS]"])

    # [/ANSWER] 종료 패턴을 위한 상태 추적
    answer_end_pattern = "[/ANSWER]"
    generated_tokens = []

    # 스트리밍 콜백 함수
    def streaming_callback(token_id: int):
        """토큰 생성 시 즉시 출력"""
        generated_tokens.append(token_id)

        # 토큰을 텍스트로 변환
        token_text = tokenizer.decode([token_id])

        if streaming:
            # 스트리밍 출력 (버퍼 없이 즉시)
            sys.stdout.write(token_text)
            sys.stdout.flush()

    # 생성
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens if stop_tokens else None,
            callback=streaming_callback
        )

    if streaming:
        print()  # 줄바꿈

    # 디코딩
    text = tokenizer.decode(out[0].tolist())

    # [/ANSWER] 이후 텍스트 제거 (깔끔한 종료)
    if answer_end_pattern in text:
        text = text.split(answer_end_pattern)[0] + answer_end_pattern

    return text


def generate_text_streaming(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 500,
    temperature: float = 0.9,
    top_k: int = 50,
    device: torch.device = None
):
    """
    스트리밍 방식으로 텍스트를 생성합니다.
    토큰이 생성될 때마다 즉시 화면에 출력합니다.

    Args:
        model: GPT 모델
        tokenizer: 토크나이저
        prompt: 시작 텍스트
        max_new_tokens: 생성할 최대 토큰 수
        temperature: 샘플링 온도
        top_k: Top-K 샘플링
        device: 디바이스
    """
    import sys

    # 프롬프트 토큰화
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # 프롬프트 먼저 출력
    sys.stdout.write(prompt)
    sys.stdout.flush()

    # 종료 패턴 체크용
    generated_text = ""
    answer_end_pattern = "[/ANSWER]"

    # 종료 토큰 찾기
    stop_tokens = []
    vocab = tokenizer.get_vocab()
    if "[EOS]" in vocab:
        stop_tokens.append(vocab["[EOS]"])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 컨텍스트 길이 제한
            idx_cond = idx[:, -model.block_size:]

            # 순전파
            logits = model(idx_cond)
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / max(temperature, 1e-6)

            # Top-K
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # 샘플링
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # 시퀀스에 추가
            idx = torch.cat((idx, next_id), dim=1)

            # 토큰을 텍스트로 변환하여 즉시 출력
            token_text = tokenizer.decode([next_id.item()])
            sys.stdout.write(token_text)
            sys.stdout.flush()

            # 생성된 텍스트 누적
            generated_text += token_text

            # 종료 조건 체크
            if answer_end_pattern in generated_text:
                break

            if stop_tokens and next_id.item() in stop_tokens:
                break

    print()  # 마지막 줄바꿈
    return tokenizer.decode(idx[0].tolist())


# ============================================================================
# 대화형 모드
# ============================================================================
def interactive_mode(model: GPT, tokenizer: Tokenizer, device: torch.device, args):
    """대화형 텍스트 생성"""
    print("\n" + "=" * 60)
    print("JAI 텍스트 생성기 (대화형 모드)")
    print("=" * 60)
    print("프롬프트를 입력하면 텍스트를 생성합니다.")
    print("스트리밍 모드: 토큰이 생성될 때마다 즉시 출력")
    print("종료: 'quit' 또는 'exit' 입력")
    print("=" * 60 + "\n")

    while True:
        try:
            # 프롬프트 입력
            prompt = input("프롬프트> ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("종료합니다.")
                break

            if not prompt:
                print("프롬프트를 입력하세요.")
                continue

            # 스트리밍 모드로 텍스트 생성
            print("\n" + "-" * 40)
            generate_text_streaming(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break


# ============================================================================
# 메인 함수
# ============================================================================
def main():
    # 인자 파서
    parser = argparse.ArgumentParser(description="JAI GPT 텍스트 생성")
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="생성할 텍스트의 시작 프롬프트"
    )
    parser.add_argument(
        "--max_tokens", "-m",
        type=int,
        default=500,
        help="생성할 최대 토큰 수 (기본값: 500)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.9,
        help="샘플링 온도 (기본값: 0.9)"
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=50,
        help="Top-K 샘플링 (기본값: 50)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드 실행"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="스트리밍 비활성화 (한번에 전체 출력)"
    )

    args = parser.parse_args()

    # 디바이스 설정
    device = get_device()
    print(f"디바이스: {device}")

    # 모델 로드
    model, tokenizer = load_model(device)

    # 실행 모드 결정
    if args.prompt:
        # 단일 프롬프트 모드
        print(f"\n프롬프트: {args.prompt}")
        print(f"설정: max_tokens={args.max_tokens}, temperature={args.temperature}, top_k={args.top_k}")

        if args.no_streaming:
            # 스트리밍 비활성화: 한번에 전체 출력
            print("\n생성 중...")
            text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
                streaming=False
            )

            print("\n" + "=" * 60)
            print("생성 결과")
            print("=" * 60)
            print(text)
            print("=" * 60)
        else:
            # 스트리밍 모드: 토큰이 생성될 때마다 즉시 출력
            print("\n" + "=" * 60)
            print("생성 결과 (스트리밍)")
            print("=" * 60)
            generate_text_streaming(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            print("=" * 60)

    else:
        # 대화형 모드
        interactive_mode(model, tokenizer, device, args)


# ============================================================================
# 스크립트 실행
# ============================================================================
if __name__ == "__main__":
    main()
