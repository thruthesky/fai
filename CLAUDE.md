# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 언어

모든 응답은 한국어로 작성합니다. 코드 주석도 한국어로 작성합니다.

## 프로젝트 개요

CAI (Contact AI)는 전 세계 연락처 정보를 제공하는 LLM을 처음부터(from scratch) 구현하는 학습 프로젝트입니다. 파인튜닝이 아닌, 토크나이저부터 GPT 모델까지 직접 구현합니다.

## 실행 명령어

```bash
# 환경 설정
python -m venv venv
source venv/bin/activate
pip install torch tokenizers tqdm numpy

# MPS fallback (Mac M4 필수)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 순차 실행 (순서 중요)
python 01_prepare_samples.py      # 데이터 전처리 → data/samples.txt
python 02_train_tokenizer.py      # BPE 토크나이저 → data/tokenizer.json
python 03_build_bin_dataset.py    # 바이너리 변환 → data/train.bin, data/val.bin
python 04_train_gpt_from_scratch.py  # GPT 학습 → checkpoints/ckpt.pt
python 05_generate.py             # 텍스트 생성
```

## 아키텍처

### 데이터 흐름
```
data/raw.txt → 전처리 → samples.txt → 토큰화 → train.bin/val.bin → GPT 학습 → 생성
```

### GPT 모델 구조 (Decoder-only Transformer)
- `CausalSelfAttention`: Q, K, V를 한번에 계산, causal mask로 미래 토큰 차단
- `MLP`: 2층 Feed Forward (n_embd → 4*n_embd → n_embd), GELU 활성화
- `Block`: Pre-LayerNorm + Residual Connection
- `GPT`: 토큰/위치 임베딩 → N개 Block → 출력 헤드

### 학습 데이터 형식
```
[QUESTION]
질문 내용
[/QUESTION]

[DOC]
원문 내용
[/DOC]

[ANSWER]
요약:
- ...
체크리스트:
- ...
연락처(공공정보):
- TEL: ...
상세 설명:
...
[/ANSWER]
```

## 하이퍼파라미터 (M4 기준)

| 파라미터 | 기본값 |
|----------|--------|
| vocab_size | 24,000 |
| block_size | 256 |
| n_layer | 6 |
| n_head | 6 |
| n_embd | 384 |
| batch_size | 16 |
| learning_rate | 3e-4 |

## 문서 참조

상세 내용은 `docs/` 폴더 참조:
- 모델 구조: `docs/05-model-architecture.md`
- 학습 코드: `docs/06-training.md`
- 핵심 개념: `docs/08-concepts.md`
- 트러블슈팅: `docs/09-tips.md`
