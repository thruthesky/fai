# 한국어 중심 ByteLevel BPE 토크나이저 — 파이 인공지능(FAI) 재학습용.
#
# 기존 scripts/train_tokenizer.py(Whitespace pre-tokenizer, 영어 전용)와 달리
# ByteLevel BPE(GPT-2 방식)를 사용한다:
#   - [UNK] 없이 모든 유니코드(한글·이모지 포함)를 바이트 단위로 처리
#   - 교착어인 한국어의 어절 폭발 문제를 회피
# 출력은 HuggingFace tokenizers 표준 tokenizer.json — 브라우저(JS)에서 그대로 재사용한다.
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .config import SPECIAL_TOKENS, VOCAB_SIZE


def train_korean_tokenizer(samples_path: Path, out_path: Path, vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """samples.txt(한·영 혼합 코퍼스)로 ByteLevel BPE 토크나이저를 학습해 저장한다."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        # ByteLevel 은 기본 알파벳(256 바이트)을 반드시 포함해야 [UNK] 없이 동작한다.
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(samples_path)], trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
