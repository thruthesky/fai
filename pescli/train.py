# 로컬 트레이닝 루프 — 개인 컴퓨터에서 실행된다 (서버는 관여하지 않는다).
#
# distributed/common/model.py 의 GPT 를 그대로 사용하되,
# 한국어 재학습 결정 사항(docs/pai-analysis.md)에 따라 weight tying 을 적용한다:
#   head.weight 와 tok_emb.weight 를 공유해 파라미터 약 35M → 23M 로 축소.
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from distributed.common.model import GPT, GPTConfig

from .config import BLOCK_SIZE, VOCAB_SIZE


@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_steps: int = 5000
    eval_interval: int = 500
    eval_iters: int = 50
    grad_clip: float = 1.0


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_model(config: GPTConfig | None = None, tie_weights: bool = True) -> GPT:
    """weight tying 이 적용된 FAI GPT 를 생성한다."""
    cfg = config or GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE)
    model = GPT(cfg)
    if tie_weights:
        model.head.weight = model.tok_emb.weight
    return model


def load_checkpoint_model(ckpt_path: Path, device: str = "cpu") -> tuple[GPT, dict]:
    """체크포인트에서 모델 복원 — 외부에서 받은 파일일 수 있으므로 weights_only 로만 로드한다."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = GPTConfig.from_dict(ckpt.get("cfg", {}))
    model = create_model(cfg)
    model.load_state_dict(ckpt["model"])
    # tying 재적용 (state_dict 로드가 공유를 끊을 수 있음)
    model.head.weight = model.tok_emb.weight
    return model.to(device), ckpt


def _get_batch(data: np.memmap, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_loss(model: GPT, data: np.memmap, tc: TrainConfig, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(tc.eval_iters):
        x, y = _get_batch(data, tc.batch_size, model.block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(
    data_dir: Path,
    ckpt_path: Path,
    resume_from: Path | None = None,
    tc: TrainConfig | None = None,
) -> dict:
    """트레이닝 실행. Ctrl+C 로 중단해도 마지막 저장 시점까지의 체크포인트가 남는다.

    반환: {"steps": 학습 스텝 수, "train_loss": ..., "val_loss": ..., "device": ...}
    """
    tc = tc or TrainConfig()
    device = pick_device()
    print(f"[FAI] 디바이스: {device}")

    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    start_step = 0
    if resume_from and resume_from.exists():
        model, ckpt = load_checkpoint_model(resume_from, device)
        start_step = int(ckpt.get("step", 0))
        print(f"[FAI] 체크포인트에서 이어 학습: step {start_step}")
    else:
        model = create_model().to(device)
        print(f"[FAI] 새 모델 생성: {model.param_count():,} 파라미터 (weight tying 적용)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate)
    model.train()

    train_loss = val_loss = float("nan")
    step = start_step
    t0 = time.time()
    try:
        for step in range(start_step + 1, start_step + tc.max_steps + 1):
            x, y = _get_batch(train_data, tc.batch_size, model.block_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optimizer.step()

            if step % tc.eval_interval == 0 or step == start_step + tc.max_steps:
                train_loss = _estimate_loss(model, train_data, tc, device)
                val_loss = _estimate_loss(model, val_data, tc, device)
                dt = time.time() - t0
                print(f"[FAI] step {step} | train {train_loss:.4f} | val {val_loss:.4f} | {dt:.0f}s")
                _save(model, step, train_loss, val_loss, ckpt_path)
    except KeyboardInterrupt:
        print("\n[FAI] 중단됨 — 마지막 체크포인트는 저장되어 있습니다.")

    return {
        "steps": step - start_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "device": device,
    }


def _save(model: GPT, step: int, train_loss: float, val_loss: float, ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # 옵티마이저 상태는 저장하지 않는다 — 업로드 크기를 1/3로 줄이고,
    # weights_only=True 로 로드 가능한 순수 텐서 딕셔너리를 유지한다.
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "cfg": model.config.to_dict(),
        },
        ckpt_path,
    )
