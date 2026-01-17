# 환경 설정

## 개요

CAI 프로젝트는 **Macbook M4 (MPS GPU)**를 기준으로 설계되었습니다.
Apple Silicon은 PyTorch의 MPS(Metal Performance Shaders) GPU 가속을 지원하여 개인이 LLM을 학습하기에 매우 좋은 환경입니다.

## 1. Python 가상환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd cai

# Python 가상환경 생성
python -m venv .venv

# 가상환경 활성화
source .venv/bin/activate
```

## 2. 필수 패키지 설치

```bash
pip install torch torchvision torchaudio
pip install tokenizers tqdm numpy
```

### 패키지 설명

| 패키지 | 용도 |
|--------|------|
| `torch` | PyTorch 딥러닝 프레임워크 |
| `torchvision` | 이미지 관련 유틸리티 (선택) |
| `torchaudio` | 오디오 관련 유틸리티 (선택) |
| `tokenizers` | Hugging Face BPE 토크나이저 |
| `tqdm` | 진행률 표시 |
| `numpy` | 수치 연산 |

## 3. MPS (Metal Performance Shaders) 확인

PyTorch는 Mac에서 Metal 기반 GPU 가속으로 학습을 돌릴 수 있습니다.

```python
import torch

# MPS 사용 가능 여부 확인
print(torch.backends.mps.is_available())  # True면 OK

# 디바이스 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
```

## 4. MPS Fallback 설정 (중요!)

MPS는 빠르지만, 일부 연산이 미구현일 수 있어 에러가 날 때가 있습니다.
이때 CPU로 자동 fallback시키는 환경 변수를 설정합니다.

### 터미널에서 설정

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### .bashrc 또는 .zshrc에 영구 설정

```bash
# ~/.zshrc 또는 ~/.bashrc에 추가
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
source ~/.zshrc
```

### 주의사항

- 이 환경 변수는 **터미널에서 export**하는 것이 권장됩니다.
- 파이썬 코드 내부에서 설정하지 마세요.
- Fallback이 발생하면 해당 연산은 CPU에서 실행되므로 약간 느려질 수 있습니다.

## 5. 디바이스 자동 선택 코드

학습 코드에서 사용할 디바이스 선택 함수:

```python
def get_device():
    """
    사용 가능한 최적의 디바이스를 자동으로 선택합니다.
    우선순위: MPS (Mac) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"사용 디바이스: {device}")
```

## 6. 설치 확인 스크립트

전체 환경이 올바르게 설정되었는지 확인하는 스크립트:

```python
# check_env.py
import torch
from tokenizers import Tokenizer
import numpy as np

print("=" * 50)
print("CAI 환경 설정 확인")
print("=" * 50)

# PyTorch 버전
print(f"PyTorch 버전: {torch.__version__}")

# MPS 확인
mps_available = torch.backends.mps.is_available()
print(f"MPS 사용 가능: {mps_available}")

# CUDA 확인 (있는 경우)
cuda_available = torch.cuda.is_available()
print(f"CUDA 사용 가능: {cuda_available}")

# 디바이스 선택
if mps_available:
    device = "mps"
elif cuda_available:
    device = "cuda"
else:
    device = "cpu"
print(f"선택된 디바이스: {device}")

# 간단한 텐서 연산 테스트
x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x @ y
print(f"텐서 연산 테스트: 성공")

# NumPy 확인
print(f"NumPy 버전: {np.__version__}")

# Tokenizers 확인
print(f"Tokenizers 라이브러리: 설치됨")

print("=" * 50)
print("모든 환경 설정이 완료되었습니다!")
print("=" * 50)
```

실행:
```bash
python check_env.py
```

## 7. 메모리 관련 팁

### M4 Mac에서 권장 설정

| 설정 | 권장값 | 설명 |
|------|--------|------|
| batch_size | 8~16 | 메모리 상황에 따라 조절 |
| block_size | 256 | 컨텍스트 길이 |
| n_layer | 6 | Transformer 레이어 수 |
| n_embd | 384 | 임베딩 차원 |

### OOM (Out of Memory) 발생 시

1. `batch_size`를 줄이기 (16 → 8 → 4)
2. `block_size`를 줄이기 (256 → 128)
3. 모델 크기 줄이기 (n_layer, n_embd)

## 다음 단계

- [02-project-structure.md](02-project-structure.md) - 프로젝트 폴더 구조
