# 스냅샷(JSONL) → 학습 코퍼스(samples.txt) → 바이너리 데이터셋(train.bin/val.bin).
#
# 스냅샷 형식(서버 /api/fai/snapshot 이 내려주는 한 줄):
#   {"id": 7, "title": "제목", "text": "본문", "contributor": "uid"}
# 학습 포맷(일반 텍스트 코퍼스 — 기존 [QUESTION]/[DOC]/[ANSWER] QA 포맷을 대체):
#   [BOS] 제목 \n 본문 [EOS]
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

VAL_RATIO = 0.01  # 99:1 train/val 분할


def snapshot_to_samples(snapshot_path: Path, samples_path: Path) -> int:
    """JSONL 스냅샷을 학습 코퍼스 텍스트로 변환한다. 반환값은 문서 수."""
    count = 0
    with snapshot_path.open(encoding="utf-8") as src, samples_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()
            if not text:
                continue
            dst.write(f"[BOS] {title}\n{text} [EOS]\n")
            count += 1
    return count


def build_bin_dataset(samples_path: Path, tokenizer: Tokenizer, out_dir: Path) -> tuple[int, int]:
    """코퍼스 전체를 인코딩해 uint16 바이너리(train.bin/val.bin)로 저장한다.

    vocab 32,000 < 65,536 이므로 uint16 으로 충분하다.
    반환값은 (train 토큰 수, val 토큰 수).
    """
    text = samples_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text).ids
    arr = np.array(ids, dtype=np.uint16)

    split = int(len(arr) * (1 - VAL_RATIO))
    out_dir.mkdir(parents=True, exist_ok=True)
    arr[:split].tofile(out_dir / "train.bin")
    arr[split:].tofile(out_dir / "val.bin")
    return split, len(arr) - split
