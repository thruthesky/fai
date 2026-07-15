# 로컬 FedAvg 병합 — 오너의 컴퓨터에서 실행된다 (fai merge). 서버는 병합하지 않는다.
#
# distributed/server/services/merger.py 의 FedAvg 로직을 서버 없이 동작하도록 이식하고,
# 신뢰할 수 없는 기여 파일을 다루기 위해 포맷을 SafeTensors 로 고정했다
# (pickle 기반 .pt 는 로드만으로 임의 코드가 실행될 수 있어 사용하지 않는다 — checkpoint_io.py).
from __future__ import annotations

import math
from pathlib import Path

import torch

from distributed.common.model import GPTConfig

from .checkpoint_io import load_checkpoint
from .train import create_model


def _validate_state(state: dict, reference_keys: set[str]) -> str | None:
    """NaN/Inf·키 불일치 검사. 문제가 있으면 사유 문자열, 정상이면 None."""
    if set(state.keys()) != reference_keys:
        return "state_dict 키 불일치"
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            return f"{name} 에 NaN/Inf 존재"
    return None


def _l2_norm(state: dict[str, torch.Tensor]) -> float:
    return float(torch.sqrt(sum((t.to(torch.float64) ** 2).sum() for t in state.values())).item())


def merge_checkpoints(
    ckpt_paths: list[Path],
    out_path: Path,
    *,
    norm_ratio_limit: float = 5.0,
) -> dict:
    """체크포인트 목록을 steps 가중 FedAvg 로 병합한다.

    각 파일은 pescli 가 저장한 SafeTensors 형식이어야 한다.
    이상 기여 배제: 로드 실패, 키 불일치, NaN/Inf, val_loss 이상값,
    그리고 다른 기여 대비 L2 노름이 norm_ratio_limit 배를 넘는 경우(모델 오염 방어).

    반환: {"merged": 병합 수, "rejected": [(파일, 사유)], "total_steps": ..., "out": ...}
    """
    if not ckpt_paths:
        raise ValueError("병합할 체크포인트가 없습니다")

    candidates: list[tuple[dict, float, Path, float]] = []  # (state, steps, path, norm)
    rejected: list[tuple[str, str]] = []
    reference_keys: set[str] | None = None
    cfg_dict: dict | None = None

    for path in ckpt_paths:
        try:
            state, meta = load_checkpoint(path)
        except Exception as e:  # 손상·비호환·비 SafeTensors 파일
            rejected.append((path.name, f"로드 실패: {e}"))
            continue

        steps = float(meta.get("step", 0) or 0)
        if not state or steps <= 0:
            rejected.append((path.name, "형식 오류 (텐서/step 없음)"))
            continue

        val_loss = meta.get("val_loss")
        if val_loss is not None and (not math.isfinite(val_loss) or val_loss > 20):
            rejected.append((path.name, f"val_loss 이상값: {val_loss}"))
            continue

        if reference_keys is None:
            reference_keys = set(state.keys())
            cfg_dict = meta.get("cfg", {})
        reason = _validate_state(state, reference_keys)
        if reason:
            rejected.append((path.name, reason))
            continue

        candidates.append((state, steps, path, _l2_norm(state)))

    if not candidates:
        raise ValueError(f"유효한 체크포인트가 없습니다: {rejected}")

    # 노름 이상치 배제 — 중앙값 대비 과도하게 큰 가중치는 오염 기여로 본다.
    norms = sorted(c[3] for c in candidates)
    median_norm = norms[len(norms) // 2]
    accepted: list[tuple[dict, float, Path]] = []
    for state, steps, path, norm in candidates:
        if median_norm > 0 and norm > median_norm * norm_ratio_limit:
            rejected.append((path.name, f"가중치 노름 이상 ({norm:.1f} > 중앙값 {median_norm:.1f}×{norm_ratio_limit})"))
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

    from .checkpoint_io import save_checkpoint

    save_checkpoint(
        merged,
        out_path,
        step=int(total_steps),
        cfg=cfg_dict or {},
        extra={"merged_from": [p.name for _, _, p in accepted]},
    )
    return {
        "merged": len(accepted),
        "rejected": rejected,
        "total_steps": int(total_steps),
        "out": str(out_path),
    }
