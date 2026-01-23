# 환경 설정

JAI 프로젝트를 위한 개발 환경 설정 방법입니다.

---

## 한 줄 요약

> **uv로 패키지 관리, MPS로 Mac GPU 가속**

---

## 1. 대상 환경

JAI 프로젝트는 **Macbook M4 (MPS GPU)**를 기준으로 설계되었습니다.

| 환경 | 지원 여부 |
|------|----------|
| Mac (Apple Silicon) | ✅ MPS GPU 가속 |
| Linux (NVIDIA GPU) | ✅ CUDA 지원 |
| Windows | ✅ CUDA 또는 CPU |
| Mac (Intel) | ✅ CPU만 |

---

## 2. uv 설치

[uv](https://github.com/astral-sh/uv)는 빠른 Python 패키지 관리자입니다.

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew
brew install uv
```

설치 확인:
```bash
uv --version
```

---

## 3. 프로젝트 생성

```bash
# 새 프로젝트 생성
uv init jai
cd jai

# 필요한 폴더 생성
mkdir -p data scripts checkpoints docs faq
```

---

## 4. 필수 패키지 설치

```bash
uv add torch tokenizers tqdm numpy
```

### 패키지 설명

| 패키지 | 용도 |
|--------|------|
| `torch` | PyTorch 딥러닝 프레임워크 |
| `tokenizers` | Hugging Face BPE 토크나이저 |
| `tqdm` | 진행률 표시 |
| `numpy` | 수치 연산 |

### 서버 구축 시 추가 패키지

```bash
uv add fastapi uvicorn
```

---

## 5. MPS (Mac GPU) 확인

MPS(Metal Performance Shaders)는 Apple Silicon Mac에서 GPU 가속을 사용하는 기능입니다.

### 터미널에서 확인

```bash
uv run python -c "import torch; print('MPS 사용 가능:', torch.backends.mps.is_available())"
```

`True`가 출력되면 GPU 가속 사용 가능합니다.

### Python 코드에서 확인

```python
import torch

# MPS 사용 가능 여부 확인
print(torch.backends.mps.is_available())  # True면 OK

# 자동 디바이스 선택
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"사용 디바이스: {device}")
```

---

## 6. 디바이스 자동 선택 함수

학습/추론 코드에서 사용할 디바이스 선택 함수:

```python
import torch

def get_device():
    """최적의 디바이스 자동 선택"""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Mac GPU
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    return torch.device("cpu")

device = get_device()
print(f"사용 디바이스: {device}")
```

---

## 7. MPS Fallback 설정

일부 PyTorch 연산이 MPS에서 지원되지 않을 수 있습니다.
이 경우 자동으로 CPU로 폴백하도록 설정합니다.

```python
import os
# MPS fallback 설정 (torch import 전에 반드시 설정)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
# ... 이후 코드
```

---

## 요약

| 항목 | 명령어/설정 |
|------|------------|
| uv 설치 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| 프로젝트 생성 | `uv init jai && cd jai` |
| 패키지 설치 | `uv add torch tokenizers tqdm numpy` |
| MPS 확인 | `torch.backends.mps.is_available()` |
| Fallback 설정 | `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` |

---

## 관련 문서

- [project-structure.md](project-structure.md) - 프로젝트 폴더 구조
- [data-flow.md](data-flow.md) - 데이터 흐름 파이프라인
