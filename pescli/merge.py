# 로컬 FedAvg 병합 — 오너의 컴퓨터에서 실행된다 (fai merge). 서버는 병합하지 않는다.
#
# distributed/server/services/merger.py 의 FedAvg 로직을 서버 없이 동작하도록 이식했다.
# 보안: 외부에서 받은 .pt 는 반드시 weights_only=True 로만 로드한다
#       (torch.load 기본 pickle 은 임의 코드 실행 위험 — PyTorch 공식 경고).
from __future__ import annotations

import math
from pathlib import Path

import torch

from distributed.common.model import GPTConfig

from .train import create_model


def _validate_state(state: dict, reference_keys: set[str]) -> str | None:
    """NaN/Inf·키 불일치 검사. 문제가 있으면 사유 문자열, 정상이면 None."""
    if set(state.keys()) != reference_keys:
        return "state_dict 키 불일치"
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            return f"{name} 에 NaN/Inf 존재"
    return None


def merge_checkpoints(ckpt_paths: list[Path], out_path: Path) -> dict:
    """체크포인트 목록을 steps 가중 FedAvg 로 병합한다.

    각 파일은 pescli/train.py 가 저장한 형식이어야 한다:
      { "model": state_dict, "step": int, "train_loss": float, "val_loss": float, "cfg": dict }
    반환: {"merged": 병합된 수, "rejected": [(파일, 사유)], "total_steps": ...}
    """
    if not ckpt_paths:
        raise ValueError("병합할 체크포인트가 없습니다")

    accepted: list[tuple[dict, float, Path]] = []
    rejected: list[tuple[str, str]] = []
    reference_keys: set[str] | None = None
    cfg_dict: dict | None = None

    for path in ckpt_paths:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # 손상·비호환 파일
            rejected.append((path.name, f"로드 실패: {e}"))
            continue
        state = ckpt.get("model")
        steps = float(ckpt.get("step", 0) or 0)
        val_loss = ckpt.get("val_loss")
        if not isinstance(state, dict) or steps <= 0:
            rejected.append((path.name, "형식 오류 (model/step 없음)"))
            continue
        if val_loss is not None and (not math.isfinite(float(val_loss)) or float(val_loss) > 20):
            rejected.append((path.name, f"val_loss 이상값: {val_loss}"))
            continue
        if reference_keys is None:
            reference_keys = set(state.keys())
            cfg_dict = ckpt.get("cfg", {})
        reason = _validate_state(state, reference_keys)
        if reason:
            rejected.append((path.name, reason))
            continue
        accepted.append((state, steps, path))

    if not accepted:
        raise ValueError(f"유효한 체크포인트가 없습니다: {rejected}")

    # FedAvg: global = Σ(stepsᵢ · stateᵢ) / Σ stepsᵢ
    total_steps = sum(s for _, s, _ in accepted)
    merged = {
        key: sum(state[key].to(torch.float32) * (steps / total_steps) for state, steps, _ in accepted)
        for key in accepted[0][0].keys()
    }

    # 병합 결과가 실제 모델에 로드되는지 확인 (키·shape 검증)
    model = create_model(GPTConfig.from_dict(cfg_dict or {}))
    model.load_state_dict(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": merged,
            "step": int(total_steps),
            "train_loss": None,
            "val_loss": None,
            "cfg": cfg_dict,
            "merged_from": [p.name for _, _, p in accepted],
        },
        out_path,
    )
    return {
        "merged": len(accepted),
        "rejected": rejected,
        "total_steps": int(total_steps),
        "out": str(out_path),
    }
