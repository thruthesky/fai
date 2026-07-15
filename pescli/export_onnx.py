# PyTorch 체크포인트 → ONNX 변환 + int8 양자화 — 오너/개발자의 로컬 컴퓨터에서 실행 (fai build).
#
# 브라우저(onnxruntime-web)에서 실행하기 위한 변환:
#   - forward 를 logits-only 로 래핑 (targets/loss 분기 제거)
#   - seq_len 은 dynamic axis (causal mask 버퍼는 block_size 크기로 export 되어 슬라이싱은 상수 범위)
#   - 샘플링 루프(generate)는 export 하지 않는다 — 브라우저 JS 에서 구현
# 의존성: uv sync --extra build  (onnx, onnxruntime)
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .train import load_checkpoint_model


class LogitsOnly(nn.Module):
    """ONNX export 용 래퍼 — (B, T) 토큰 → (B, T, vocab) logits 만 반환."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx)
        return logits


def export_onnx(ckpt_path: Path, out_dir: Path) -> dict:
    """체크포인트를 ONNX(fp32) + int8 양자화 모델로 변환하고 logits 일치를 검증한다."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model, ckpt = load_checkpoint_model(ckpt_path, device="cpu")
    model.eval()
    wrapper = LogitsOnly(model)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "fai.onnx"
    int8_path = out_dir / "fai.int8.onnx"

    dummy = torch.randint(0, model.config.vocab_size, (1, 32), dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(fp32_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )

    # 검증 1: fp32 ONNX 와 PyTorch logits 일치 (허용 오차 내)
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    test = torch.randint(0, model.config.vocab_size, (1, 16), dtype=torch.long)
    with torch.no_grad():
        pt_logits = wrapper(test).numpy()
    onnx_logits = sess.run(["logits"], {"input_ids": test.numpy()})[0]
    max_diff = float(np.abs(pt_logits - onnx_logits).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX/PyTorch logits 불일치: max diff {max_diff}")

    # int8 동적 양자화 (~1/4 크기) — 브라우저 다운로드 부담 축소
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    # 검증 2: int8 모델이 로드·실행되는지 smoke test
    sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    sess8.run(["logits"], {"input_ids": test.numpy()})

    return {
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "max_diff": max_diff,
        "step": int(ckpt.get("step", 0)),
    }
