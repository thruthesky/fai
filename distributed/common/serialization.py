# distributed/common/serialization.py
# ============================================================================
# 모델 가중치 직렬화/역직렬화
# ============================================================================
# 체크포인트 저장, 워커 ↔ 서버 간 가중치 전송에 사용됩니다.
# Delta 전송 (차분 전송) + gzip 압축으로 네트워크 대역폭을 절약합니다.

from __future__ import annotations

import gzip
import io
from collections import OrderedDict

import torch


def serialize_state_dict(state_dict: dict[str, torch.Tensor]) -> bytes:
    """
    모델 state_dict를 바이트로 직렬화 + gzip 압축

    Args:
        state_dict: 모델 가중치 (model.state_dict())

    Returns:
        gzip 압축된 바이트 데이터
    """
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    raw_bytes = buffer.getvalue()

    # gzip 압축
    return gzip.compress(raw_bytes, compresslevel=6)


def deserialize_state_dict(data: bytes) -> OrderedDict:
    """
    바이트 데이터를 모델 state_dict로 역직렬화

    Args:
        data: gzip 압축된 바이트 데이터

    Returns:
        모델 가중치 OrderedDict
    """
    # gzip 압축 해제 시도 (비압축 데이터도 처리)
    try:
        raw_bytes = gzip.decompress(data)
    except gzip.BadGzipFile:
        raw_bytes = data

    buffer = io.BytesIO(raw_bytes)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def compute_delta(
    base_state_dict: dict[str, torch.Tensor],
    new_state_dict: dict[str, torch.Tensor],
) -> OrderedDict:
    """
    두 state_dict의 차이(delta)를 계산합니다.
    전체 가중치 대신 변화분만 전송하여 네트워크 대역폭을 절약합니다.

    Args:
        base_state_dict: 기준 가중치 (학습 시작 시점의 체크포인트)
        new_state_dict: 새 가중치 (학습 완료 후)

    Returns:
        delta: 가중치 차이 (new - base)
    """
    delta = OrderedDict()
    for key in new_state_dict:
        if key in base_state_dict:
            delta[key] = new_state_dict[key] - base_state_dict[key]
        else:
            # 새 키가 추가된 경우 (보통 없지만 안전 처리)
            delta[key] = new_state_dict[key]
    return delta


def apply_delta(
    base_state_dict: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
) -> OrderedDict:
    """
    기준 가중치에 delta를 적용하여 새 가중치를 복원합니다.

    Args:
        base_state_dict: 기준 가중치
        delta: 가중치 차이

    Returns:
        복원된 가중치 (base + delta)
    """
    result = OrderedDict()
    for key in base_state_dict:
        if key in delta:
            result[key] = base_state_dict[key] + delta[key]
        else:
            result[key] = base_state_dict[key]
    return result


def state_dict_size_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    """state_dict의 전체 크기(바이트) 계산"""
    total = 0
    for tensor in state_dict.values():
        total += tensor.nelement() * tensor.element_size()
    return total
