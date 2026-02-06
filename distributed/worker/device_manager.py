# distributed/worker/device_manager.py
# ============================================================================
# GPU/CPU 자동 감지 및 시스템 정보 수집
# ============================================================================
# CUDA > MPS > CPU 순으로 디바이스를 자동 감지합니다.
# GPU 메모리, RAM, CPU 코어 수 등 시스템 정보를 수집합니다.

from __future__ import annotations

import os
from dataclasses import dataclass

import psutil
import torch


@dataclass
class DeviceInfo:
    """시스템 하드웨어 정보"""
    device_type: str          # 'cuda', 'mps', 'cpu'
    device_name: str          # GPU 모델명 또는 'CPU'
    gpu_memory_mb: int | None # GPU 전용 메모리 (MB)
    ram_mb: int               # 시스템 RAM (MB)
    cpu_cores: int            # CPU 코어 수
    torch_device: torch.device


def detect_device(preferred: str | None = None) -> DeviceInfo:
    """
    최적의 디바이스를 자동 감지합니다.

    【우선순위】
    1. preferred가 지정되면 해당 디바이스 사용
    2. CUDA (NVIDIA GPU)
    3. MPS (Apple Silicon GPU)
    4. CPU (폴백)

    Args:
        preferred: 선호 디바이스 ('cuda', 'mps', 'cpu', None)

    Returns:
        DeviceInfo: 감지된 디바이스 정보
    """
    # MPS Fallback 설정 (Apple Silicon에서 미지원 연산 시 CPU로 자동 전환)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # 시스템 정보 (공통)
    ram_mb = psutil.virtual_memory().total // (1024 * 1024)
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()

    # 디바이스 감지
    if preferred:
        device_type = preferred
    elif torch.cuda.is_available():
        device_type = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    # 디바이스별 정보 수집
    if device_type == "cuda":
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        gpu_memory_mb = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    elif device_type == "mps":
        device = torch.device("mps")
        # Apple Silicon은 통합 메모리이므로 RAM의 일부를 GPU로 사용
        # 정확한 GPU 전용 메모리는 알 수 없으므로 RAM 기반 추정
        device_name = _get_apple_chip_name()
        gpu_memory_mb = ram_mb  # 통합 메모리
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        gpu_memory_mb = None

    return DeviceInfo(
        device_type=device_type,
        device_name=device_name,
        gpu_memory_mb=gpu_memory_mb,
        ram_mb=ram_mb,
        cpu_cores=cpu_cores,
        torch_device=device,
    )


def _get_apple_chip_name() -> str:
    """Apple Silicon 칩 이름을 가져옵니다."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "Apple Silicon"
    except Exception:
        return "Apple Silicon"
