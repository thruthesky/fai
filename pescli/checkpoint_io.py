# 체크포인트 입출력 — SafeTensors 전용.
#
# ⚠️ 왜 .pt(pickle) 를 쓰지 않는가:
#   torch.load 는 pickle 을 역직렬화하면서 임의 코드를 실행할 수 있다. 파이는 인터넷의
#   불특정 다수가 체크포인트를 업로드하는 구조이므로, 신뢰할 수 없는 파일을 여는 쪽
#   (오너의 fai merge)이 코드 실행 위험에 노출되면 안 된다.
#   SafeTensors 는 텐서 데이터만 담는 포맷이라 로드 중 코드가 실행되지 않는다.
#
# 메타데이터(step/loss/cfg)는 SafeTensors 의 __metadata__ 에 문자열로 저장한다.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

CKPT_SUFFIX = ".safetensors"


def save_checkpoint(
    state_dict: dict[str, torch.Tensor],
    path: Path,
    *,
    step: int,
    cfg: dict[str, Any],
    train_loss: float | None = None,
    val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """state_dict 와 메타데이터를 SafeTensors 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "step": str(int(step)),
        "cfg": json.dumps(cfg),
        "format": "fai-safetensors-v1",
    }
    if train_loss is not None:
        meta["train_loss"] = f"{float(train_loss):.6f}"
    if val_loss is not None:
        meta["val_loss"] = f"{float(val_loss):.6f}"
    if extra:
        meta.update({k: json.dumps(v) for k, v in extra.items()})

    # weight tying 을 쓰면 head.weight 와 tok_emb.weight 가 같은 저장소를 공유한다.
    # SafeTensors 는 메모리를 공유하는 텐서를 거부하므로 연속 복사본으로 분리해 저장한다.
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in state_dict.items()}
    save_file(tensors, str(path), metadata=meta)


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """SafeTensors 체크포인트를 로드한다 (코드 실행 위험 없음).

    반환: (state_dict, meta) — meta 는 step/cfg/train_loss/val_loss 가 파싱된 dict.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        raw_meta = f.metadata() or {}
        state = {key: f.get_tensor(key) for key in f.keys()}

    meta: dict[str, Any] = {
        "step": int(raw_meta.get("step", 0) or 0),
        "cfg": json.loads(raw_meta.get("cfg", "{}")),
        "format": raw_meta.get("format", ""),
    }
    for key in ("train_loss", "val_loss"):
        if raw_meta.get(key) is not None:
            meta[key] = float(raw_meta[key])
    if raw_meta.get("merged_from"):
        meta["merged_from"] = json.loads(raw_meta["merged_from"])
    return state, meta
