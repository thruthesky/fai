# fai doctor — 이 컴퓨터가 파이 트레이닝을 할 수 있는지 점검한다.
#
# 참여자가 install.sh 직후 가장 먼저 실행하는 명령. 학습이 실패할 조건(메모리 부족,
# 디스크 부족, PyTorch 미설치)을 미리 알려주고 예상 학습 시간을 안내한다.
from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG_PATH, DATA_DIR, WORK_DIR


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = False  # True 면 학습 불가


def _bytes_gb(n: float) -> float:
    return n / (1024**3)


def run_checks() -> list[Check]:
    checks: list[Check] = []

    # 1. Python — pyproject.toml 의 requires-python(>=3.13) 과 일치해야 한다.
    v = sys.version_info
    py_ok = (v.major, v.minor) >= (3, 13)
    checks.append(
        Check(
            "Python",
            py_ok,
            f"{v.major}.{v.minor}.{v.micro}" + ("" if py_ok else " (3.13 이상 필요 — `uv sync` 로 자동 설치)"),
            fatal=not py_ok,
        )
    )

    # 2. OS / CPU
    checks.append(Check("운영체제", True, f"{platform.system()} {platform.release()} ({platform.machine()})"))

    # 3. PyTorch + 가속 장치
    try:
        import torch

        if torch.backends.mps.is_available():
            device, detail = "mps", "Apple Silicon GPU (MPS) 가속 사용"
        elif torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = _bytes_gb(torch.cuda.get_device_properties(0).total_memory)
            device, detail = "cuda", f"NVIDIA {name} ({vram:.1f}GB) CUDA 가속 사용"
        else:
            device, detail = "cpu", "GPU 없음 — CPU 로 학습 (느립니다)"
        checks.append(Check("PyTorch", True, f"{torch.__version__} / {detail}"))
        checks.append(Check("학습 장치", True, device))
    except ImportError:
        checks.append(
            Check("PyTorch", False, "설치되지 않음 — `uv sync` 를 실행하세요", fatal=True)
        )
        device = "none"

    # 4. RAM
    try:
        import psutil

        ram = _bytes_gb(psutil.virtual_memory().total)
        ram_ok = ram >= 7.0
        checks.append(
            Check("메모리(RAM)", ram_ok, f"{ram:.1f}GB" + ("" if ram_ok else " (8GB 이상 권장)"))
        )
    except ImportError:
        checks.append(Check("메모리(RAM)", True, "확인 불가 (psutil 없음)"))

    # 5. 디스크 (작업 디렉토리)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    free = _bytes_gb(shutil.disk_usage(WORK_DIR).free)
    disk_ok = free >= 5.0
    checks.append(
        Check("디스크 여유", disk_ok, f"{free:.1f}GB 사용 가능" + ("" if disk_ok else " (5GB 이상 필요)"), fatal=not disk_ok)
    )

    # 6. 로그인 상태
    from .config import CliConfig

    cfg = CliConfig.load()
    checks.append(
        Check(
            "로그인",
            bool(cfg.token),
            f"{cfg.server} 에 토큰 등록됨" if cfg.token else "토큰 없음 — `fai login` 을 실행하세요",
        )
    )

    # 7. 데이터셋 준비 여부
    has_data = (DATA_DIR / "train.bin").exists()
    checks.append(
        Check("학습 데이터", has_data, "준비됨" if has_data else "없음 — `fai pull` 을 실행하세요")
    )

    # 8. 토크나이저 (모든 참여자가 동일한 것을 써야 가중치를 합칠 수 있다)
    tok = DATA_DIR / "tokenizer.json"
    if tok.exists():
        from .release import sha256_file

        checks.append(Check("토크나이저", True, f"fai-ko-v1 sha256:{sha256_file(tok)[:12]}…"))
    else:
        checks.append(Check("토크나이저", False, "없음 — `fai pull` 이 내려받습니다"))

    return checks


def estimate_speed() -> str | None:
    """이 컴퓨터의 대략적인 학습 속도를 측정해 안내 문구로 반환한다."""
    try:
        import time

        import torch

        from .train import create_model, pick_device

        device = pick_device()
        model = create_model().to(device)
        x = torch.randint(0, model.config.vocab_size, (4, model.block_size), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for _ in range(2):  # 워밍업
            _, loss = model(x, x)
            opt.zero_grad(); loss.backward(); opt.step()

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        steps = 5
        for _ in range(steps):
            _, loss = model(x, x)
            opt.zero_grad(); loss.backward(); opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        per_step = (time.time() - t0) / steps
        est_min = per_step * 5000 / 60
        return f"약 {per_step * 1000:.0f}ms/step → 5,000 step 기준 약 {est_min:.0f}분 예상 ({device})"
    except Exception:
        return None
