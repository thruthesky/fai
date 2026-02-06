# distributed/server/services/scheduler.py
# ============================================================================
# 워커 작업 스케줄러
# ============================================================================
# 워커의 하드웨어 사양에 따라 최적의 배치 크기와 로컬 스텝 수를 결정합니다.
#
# 【스케줄링 전략】
# 1. GPU 메모리 기반 배치 크기 결정
# 2. CPU 코어 기반 폴백
# 3. 벤치마크 결과 우선 적용

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# GPU 메모리 → 배치 크기 매핑
# ============================================================================
# (최소 GPU 메모리 MB, 추천 배치 크기)
GPU_BATCH_MAP = [
    (24000, 64),   # RTX 4090, A100 등
    (12000, 32),   # RTX 4070 Ti, RTX 3080 등
    (8000, 16),    # RTX 4060, RTX 3060 등
    (4000, 8),     # GTX 1650, MX 시리즈 등
    (0, 4),        # 기타 저사양 GPU
]

# MPS (Apple Silicon) 통합 메모리 기반
MPS_BATCH_MAP = [
    (32000, 32),   # M2 Max/Ultra 32GB+
    (16000, 16),   # M1/M2 Pro 16GB+
    (8000, 8),     # M1/M2 8GB
    (0, 4),        # 기타
]


def recommend_settings(
    device_type: str,
    gpu_memory_mb: int | None = None,
    ram_mb: int | None = None,
    cpu_cores: int | None = None,
    benchmark_score: float | None = None,
) -> dict:
    """
    워커의 하드웨어에 최적화된 학습 설정을 추천합니다.

    Args:
        device_type: 디바이스 유형 ('cuda', 'mps', 'cpu')
        gpu_memory_mb: GPU 메모리 (MB)
        ram_mb: 시스템 RAM (MB)
        cpu_cores: CPU 코어 수
        benchmark_score: 벤치마크 점수 (있으면 우선 적용)

    Returns:
        dict: {
            "recommended_batch_size": int,
            "recommended_local_steps": int,
        }
    """
    # 벤치마크 결과가 있으면 그에 따라 결정
    if benchmark_score is not None:
        return _from_benchmark(benchmark_score)

    # 디바이스별 추천
    if device_type == "cuda":
        batch_size = _lookup_batch(GPU_BATCH_MAP, gpu_memory_mb or 0)
        local_steps = _compute_local_steps(batch_size, "cuda")
    elif device_type == "mps":
        batch_size = _lookup_batch(MPS_BATCH_MAP, gpu_memory_mb or ram_mb or 0)
        local_steps = _compute_local_steps(batch_size, "mps")
    else:
        # CPU 폴백
        batch_size = min(4, (cpu_cores or 4) // 2)
        batch_size = max(batch_size, 1)
        local_steps = 30  # CPU는 적은 스텝

    logger.info(
        f"스케줄링 추천: device={device_type}, "
        f"batch_size={batch_size}, local_steps={local_steps}, "
        f"gpu_mem={gpu_memory_mb}MB"
    )

    return {
        "recommended_batch_size": batch_size,
        "recommended_local_steps": local_steps,
    }


def _lookup_batch(batch_map: list[tuple[int, int]], memory_mb: int) -> int:
    """메모리 크기에 맞는 배치 크기를 찾습니다."""
    for min_mem, batch_size in batch_map:
        if memory_mb >= min_mem:
            return batch_size
    return batch_map[-1][1]


def _compute_local_steps(batch_size: int, device_type: str) -> int:
    """
    배치 크기에 따른 로컬 스텝 수를 결정합니다.

    【원칙】
    - 큰 배치 → 적은 스텝 (1라운드 데이터 처리량 일정하게 유지)
    - GPU는 더 많은 스텝, CPU는 적은 스텝
    - 1라운드 총 토큰 수 ≈ batch_size × block_size × local_steps
    """
    # 목표: 라운드당 약 200K 토큰 처리
    target_tokens = 200_000
    block_size = 256  # GPTConfig 기본값

    steps = target_tokens // (batch_size * block_size)

    # 디바이스별 조정
    if device_type == "cuda":
        steps = max(steps, 50)   # CUDA 최소 50 스텝
        steps = min(steps, 500)  # 최대 500 스텝
    elif device_type == "mps":
        steps = max(steps, 30)   # MPS 최소 30 스텝
        steps = min(steps, 200)  # 최대 200 스텝
    else:
        steps = max(steps, 20)   # CPU 최소 20 스텝
        steps = min(steps, 100)  # 최대 100 스텝

    return steps


def _from_benchmark(score: float) -> dict:
    """
    벤치마크 점수를 기반으로 설정을 결정합니다.

    【벤치마크 점수 기준】
    - score > 100: 고성능 GPU (batch_size=64, steps=200)
    - score > 50: 중간 GPU (batch_size=32, steps=150)
    - score > 20: 저사양 GPU/MPS (batch_size=16, steps=100)
    - score > 5: CPU (batch_size=4, steps=50)
    """
    if score > 100:
        return {"recommended_batch_size": 64, "recommended_local_steps": 200}
    elif score > 50:
        return {"recommended_batch_size": 32, "recommended_local_steps": 150}
    elif score > 20:
        return {"recommended_batch_size": 16, "recommended_local_steps": 100}
    elif score > 5:
        return {"recommended_batch_size": 8, "recommended_local_steps": 50}
    else:
        return {"recommended_batch_size": 4, "recommended_local_steps": 30}
