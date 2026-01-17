좋아요. 조건이 **“한국어 100MB txt + Macbook M4(MPS GPU)”**면, 개인이 처음부터 학습하는 초간단 LLM 프로젝트를 진짜로 완주할 수 있는 매우 좋은 환경입니다. (특히 Apple Silicon은 PyTorch의 MPS GPU 가속을 쓸 수 있어요.  ￼)

아래는 제가 추천하는 “완전 초간단 내 LLM from scratch” 1주 코스입니다.

⸻

✅ 목표: mydata.txt → 토크나이저 학습 → GPT 학습 → 글 생성

전체 흐름 (딱 4개만 하면 됨)
	1.	txt 정리
	2.	토크나이저(BPE) 직접 학습
	3.	작은 GPT(Decoder Transformer) 구현
	4.	pretrain 후 generate

⸻

1) 환경 세팅 (Mac M4 + MPS)

PyTorch는 Mac에서 Metal 기반 GPU 가속(MPS)으로 학습을 돌릴 수 있습니다.  ￼

설치

python -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install tokenizers tqdm numpy

MPS 잘 잡히는지 확인

import torch
print(torch.backends.mps.is_available())  # True면 OK
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

참고: MPS는 일부 연산이 미구현일 수 있어 에러가 날 때가 있습니다. 그런 경우 CPU fallback 하거나 모델/연산을 단순화하면 해결되는 경우가 많아요.  ￼

⸻

2) 내 txt 준비 (한국어 100MB면 충분!)

✅ 가장 중요한 규칙 3개
	•	인코딩 UTF-8
	•	너무 긴 한 줄(예: 로그)을 줄바꿈으로 적당히 끊기
	•	특수문자/제어문자 제거(가능하면)

파일 예:

data/mydata.txt


⸻

3) 토크나이저를 “내 데이터로” 직접 학습 (BPE 추천)

LLM은 “문자”가 아니라 “토큰”으로 학습합니다.
가장 안정적인 입문은 BPE(Byte-Pair Encoding) 입니다.  ￼

train_tokenizer.py (초간단 버전)

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=16000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)

tokenizer.train(["data/mydata.txt"], trainer)
tokenizer.save("data/tokenizer.json")

print("Saved tokenizer to data/tokenizer.json")

실행:

python train_tokenizer.py

BPE가 GPT 계열 토크나이징에 널리 쓰인다는 설명은 Hugging Face LLM Course에도 잘 정리돼 있어요.  ￼
“토크나이저를 직접 학습시키는 방법”은 tokenizers 기반 튜토리얼들이 참고가 됩니다.  ￼

⸻

4) “초간단 GPT”를 처음부터 구현해서 학습하기

여기서부터가 진짜 **“내가 LLM을 만든다”**의 핵심입니다.

가장 추천하는 레퍼런스는 Karpathy의 build-nanogpt예요.
빈 파일에서 시작해서 GPT-2 수준 모델까지 “커밋 단계별로” 만들어가는 자료입니다.  ￼

⸻

✅ 내가 추천하는 “초간단 GPT 설정”(M4에서 잘 돌아감)

처음엔 작게 해야 성공합니다.
	•	vocab_size: 16,000 (위 토크나이저와 맞춤)
	•	block_size(context): 256
	•	n_layer: 6
	•	n_head: 6
	•	n_embd: 384
	•	batch_size: 16 (MPS 상황에 따라 8~32 조절)

⸻

5) 학습 목표는 딱 하나: Next-token prediction

LLM pretraining은 결국 이겁니다:

“앞의 토큰들을 보고 다음 토큰을 맞춰라”

이 방식이 GPT류 모델의 기본 학습 방식입니다. (build-nanogpt도 이 구조로 갑니다)  ￼

⸻

6) “내 txt는 한국어인데 괜찮나?”

네, 충분히 가능합니다. 다만 주의점이 있어요:

✅ 잘 되는 데이터 형태
	•	일기 / 에세이 / 뉴스 / 소설 / 위키식 글
	•	자연어 문장 위주

⚠️ 품질이 떨어질 수 있는 데이터
	•	코드/로그만 잔뜩 (문장 패턴이 너무 단조롭거나 깨짐)
	•	이모지/URL/해시태그가 과도

⸻

7) 지금 당장 제가 만들어드릴 것 (다음 단계)

원하시면 제가 바로:

✅ **“실행 가능한 초간단 프로젝트 폴더”**를 통째로 잡아드릴게요.

예:

my_llm/
  data/mydata.txt
  data/tokenizer.json
  train_tokenizer.py
  train_gpt.py
  generate.py

그리고 M4(MPS)에서 잘 돌아가게
	•	device 자동 선택(mps/cpu)
	•	메모리 터지면 batch 자동 줄이는 옵션
	•	학습 중간 저장/재개까지

까지 포함해드릴게요.

⸻


좋습니다. 이제 목표가 아주 명확해졌어요 ✅
	•	한국어(필리핀 교민/재필동포 정보) 100MB txt
	•	처음부터(from scratch) 학습하는 “초간단 LLM”
	•	결과물: 정보 요약/정리 잘하는 생성형 모델

아래대로 하면 **진짜 “내가 만든 LLM”**이 됩니다. (파인튜닝 아님)

⸻

✅ 전체 구조(초간단 LLM 프로젝트 설계)

최종 목표

mydata.txt 하나 넣으면
	1.	토크나이저 직접 학습(BPE)
	2.	작은 GPT(Decoder Transformer) 직접 학습
	3.	학습된 모델로 생성
	4.	생성 결과를 요약/정리 스타일로 나오게 “데이터 방식으로” 유도

구현 참고로는 Karpathy의 build-nanogpt가 “빈 파일에서 GPT 만들기”를 가장 잘 보여줍니다.  ￼

⸻

0) Macbook M4(MPS)에서 학습할 때 중요한 점 1개

PyTorch MPS는 빠르지만, 가끔 지원되지 않는 연산이 나오면 에러가 날 수 있어요.
이때는 환경변수로 CPU fallback을 켜서 진행할 수 있습니다.  ￼

터미널에서:

export PYTORCH_ENABLE_MPS_FALLBACK=1


⸻

1) (핵심) “요약/정리 잘하는 LLM”을 from scratch로 만들려면?

여기서 중요한 포인트가 있어요.

✅ LLM을 from scratch로 학습하면 기본적으로 “다음 단어 예측”만 잘해집니다.
요약 능력은 데이터에 ‘요약 형식’이 포함돼야 생깁니다.

즉, 파인튜닝 없이도 가능하지만 **txt를 요약형 데이터로 ‘재구성’**해야 합니다.

⸻

2) 데이터(txt) 가공 전략 (요약 능력을 만드는 핵심)

당신의 txt가 “필리핀 생활/교민 정보”라면,
그걸 아래처럼 “섹션/항목형 구조”로 분해해주는 게 매우 중요합니다.

✅ 추천 포맷(LLM이 요약/정리 학습하기 쉬움)

txt를 아래처럼 바꿔서 학습 데이터로 쓰면 “요약형 모델”이 됩니다.

예시 (한 샘플):

[DOC]
제목: 필리핀 마닐라 BGC 지역 생활 장단점
내용: ...원문...

[SUMMARY]
- 핵심요약: ...
- 장점: ...
- 단점: ...
- 추천대상: ...
- 주의사항: ...
[/SUMMARY]

이걸 수천~수만 개 만들면,
모델은 자연스럽게 “요약하는 패턴” 자체를 언어로 학습합니다.

이 방식이 좋은 이유: LLM의 기본 학습(다음 토큰 예측)만으로도 “요약 텍스트 형식”을 만들어내게 유도할 수 있어요.

⸻

3) 토크나이저는 “내 데이터로 BPE 직접 학습” 추천

한국어 + 필리핀 지명/기관/커뮤니티 단어들이 많을 거라서
기존 토크나이저보다 내 단어에 맞춘 토크나이저가 유리합니다.

Hugging Face tokenizers는 BPE 학습을 공식 문서에서 안내합니다.  ￼

✅ 권장값:
	•	vocab_size: 16,000 ~ 32,000
	•	special tokens: [PAD] [UNK] [BOS] [EOS]

⸻

4) “초간단 GPT” 모델 크기 추천 (M4에서 잘 돌아가는 값)

처음 성공하기 좋은 설정입니다.
	•	context length(block_size): 256
	•	n_layer: 6
	•	n_head: 6
	•	n_embd: 384
	•	batch_size: 8~32 (메모리에 맞게 조절)

⸻

5) 초간단 LLM 프로젝트 폴더 구조(추천)

이렇게 가면 깔끔합니다.

my_ph_llm/
  data/
    mydata.txt
    tokenizer.json
    train.bin
    val.bin
  01_prepare_dataset.py
  02_train_tokenizer.py
  03_train_gpt.py
  04_generate.py


⸻

6) 생성 프롬프트 템플릿(요약/정리용)

학습 때도 이런 패턴을 많이 넣고,
생성할 때도 같은 형태로 물어보면 “요약 특화 모델”이 됩니다.

예:

[DOC]
필리핀에서 ACR I-Card 연장할 때 필요한 서류는...
[/DOC]

[SUMMARY]

그러면 모델이 다음 토큰을 예측하면서
- 준비물: ... 이런 정리형으로 계속 이어가게 됩니다.

⸻

7) 공부 루트(“진짜 from scratch”로 완주하는 추천 레퍼런스)

✅ 1순위: build-nanogpt

Karpathy가 “빈 파일부터 GPT-2까지” 커밋 단위로 만들어서 공부하기 최고입니다.  ￼

✅ 토크나이저: Hugging Face tokenizers Quicktour

BPE 토크나이저 학습/저장을 공식 문서로 확인할 수 있어요.  ￼

✅ (선택) 작은 모델 학습을 빠르게 감 잡기: TinyStories 논문

“작은 모델도 말 되게 만들 수 있냐?”를 보여주는 대표 자료입니다.  ￼

⸻


완전 좋습니다. 지금 상황(한국어 100MB + M4 MPS)은 **“내 텍스트로 from-scratch LLM 만들기”**를 제대로 공부하기에 최적이에요.
아래는 최대한 상세하게, 그리고 실제로 끝까지 돌아가는 초간단 LLM 파이프라인을 “프로젝트 형태”로 드릴게요.

참고로 “빈 파일에서 GPT를 직접 구현 → 학습” 흐름은 Karpathy의 build-nanogpt가 가장 교육적으로 유명합니다.  ￼

⸻

✅ 목표(당신의 LLM이 할 일)

**“필리핀 재필교민 정보 txt”**를 학습해서,
	•	“대사관/한인회/병원/경찰서” 같은 정보를 정리된 형식으로 답하고
	•	“요약 + 체크리스트 + 핵심 포인트”를 자동 생성하는 모델

즉, “대화형 챗봇”이 아니라
✅ 요약/정리형 정보 생성 엔진을 내 LLM로 만든다고 보면 됩니다.

⸻

0) 가장 중요한 현실 체크 (from scratch의 한계와 해결책)

LLM을 처음부터 학습하면 기본적으로 **다음 토큰 예측(next-token prediction)**만 합니다.

그래서 “요약을 잘 하게” 만들려면 반드시:

✅ 학습 데이터 안에 “요약/정리된 답변 형식”이 많이 존재해야 합니다.

정리:

요약 능력 = 알고리즘이 아니라 “데이터 포맷”으로 만든다.

⸻

1) Macbook M4(MPS) 세팅 (학습이 잘 되게)

1-1. 설치

python -m venv .venv
source .venv/bin/activate

pip install torch tokenizers tqdm numpy

1-2. MPS 사용 + fallback 설정(강력추천)

Mac의 MPS는 빠르지만, 일부 연산이 미지원이면 에러가 날 수 있어요.
이럴 때 CPU로 자동 fallback시키는 환경 변수가 있습니다.  ￼

터미널에서:

export PYTORCH_ENABLE_MPS_FALLBACK=1

PyTorch 이슈에서 “임시 해결로 이 변수를 쓰면 CPU fallback 된다(느려짐)”고 명시합니다.  ￼
또한 PyTorch 포럼에서도 “파이썬 코드 내부에서 설정하지 말고 터미널에서 export 추천” 언급이 있습니다.  ￼

⸻

2) 프로젝트 폴더 구조 (이대로 만들면 됨)

my_ph_llm/
  data/
    raw.txt                # 원본 100MB
    cleaned.txt            # 전처리 후
    tokenizer.json
    train.bin
    val.bin
  01_clean_and_mask.py
  02_train_tokenizer.py
  03_build_dataset_bin.py
  04_train_gpt.py
  05_generate.py


⸻

3) 개인정보(공공 연락처 포함) 처리 전략 ✅

당신 데이터는 “대사관/병원/경찰서 연락처”가 포함되어 있고, 이건 공공정보 성격이 큽니다.

여기서 선택지가 2개예요.

✅ 전략 A (추천): “연락처는 보존”하되 표준화

연락처는 LLM이 실제로 답변에 써야 하는 핵심정보입니다.

그러므로 지우는 게 아니라 아래처럼 “표준 형식”으로 맞추는 게 좋아요.

예:
	•	전화: Tel: ☎️ → 전부 TEL: 로 통일
	•	카톡: Kakao: → KAKAO:
	•	주소/웹사이트도 ADDR:, WEB: 형태로 통일

이렇게 하면 모델이 “출력 포맷”을 배우기 쉬워집니다.

전략 B: 연락처를 익명화(학습용) + 별도 DB로 제공
	•	LLM을 “요약/정리만”
	•	연락처는 별도 검색(DB/JSON)에서 꺼내기

하지만 지금 목표가 “초간단 LLM 학습”이므로
✅ 전략 A가 공부/성과 모두 좋습니다.

⸻

4) (핵심) 요약/정리형 LLM을 만드는 데이터 포맷

원본이 그냥 긴 글이라면, LLM은 그냥 “긴 글 흉내”만 냅니다.

그래서 아래처럼 학습 텍스트 자체를 “문제→정리답변” 형태로 변환하세요.

✅ 추천 템플릿 (가장 강력)

당신의 데이터가 “필리핀 교민 정보”니까 이런 구조가 거의 만능입니다.

[QUESTION]
필리핀 마닐라에서 한국인이 여권 분실했을 때 해야 할 일 정리해줘.
[/QUESTION]

[ANSWER]
요약:
- 가장 먼저: 현지 경찰서 분실 신고(Police Report)
- 다음: 대사관/총영사관 연락 및 안내 받기

체크리스트:
- 준비물: 여권사본, 신분증, 사진
- 방문처: 경찰서 → 대사관/총영사관
- 주의사항: ...

연락처(공공정보):
- 대사관 TEL: ...
- 경찰서 TEL: ...
[/ANSWER]

이런 포맷이 많을수록 모델은 “요약을 잘하는 패턴”을 스스로 학습합니다.

⸻

5) 01_clean_and_mask.py (전처리 + 표준화)

아래는 초간단 버전입니다.

✅ 기능
	•	제어문자 제거
	•	공백 정리
	•	전화번호 표준화(일부)
	•	URL 표준화

# 01_clean_and_mask.py
import re

INPUT = "data/raw.txt"
OUTPUT = "data/cleaned.txt"

def normalize(text: str) -> str:
    # 제어문자 제거
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    # 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 전화번호 패턴(아주 대충) -> TEL: 로 표준화
    # 예: 0917-123-4567 / +63 917 123 4567 / 02-1234-5678
    text = re.sub(r"(☎️|전화|TEL|Tel|tel)\s*[:：]?\s*", "TEL: ", text)

    # URL 표준화
    text = re.sub(r"(홈페이지|사이트|웹)\s*[:：]?\s*", "WEB: ", text)

    return text.strip()

with open(INPUT, "r", encoding="utf-8") as f:
    raw = f.read()

clean = normalize(raw)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(clean)

print("Saved:", OUTPUT, "chars:", len(clean))

실행:

python 01_clean_and_mask.py


⸻

6) 02_train_tokenizer.py (내 데이터로 토크나이저 학습)

Hugging Face tokenizers는 BPE 토크나이저 학습/트레이너(vocab_size, special_tokens 등)를 공식 문서로 제공합니다.  ￼

# 02_train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

INPUT = "data/cleaned.txt"
OUT = "data/tokenizer.json"

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=24000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    show_progress=True,
)

tokenizer.train([INPUT], trainer=trainer)
tokenizer.save(OUT)

print("Saved tokenizer:", OUT)

실행:

python 02_train_tokenizer.py


⸻

7) 03_build_dataset_bin.py (토큰화 → bin 저장)

LLM 학습은 “토큰 ID 배열”로 하는 게 제일 간단합니다.

# 03_build_dataset_bin.py
import numpy as np
from tokenizers import Tokenizer

TEXT_PATH = "data/cleaned.txt"
TOK_PATH = "data/tokenizer.json"

TRAIN_OUT = "data/train.bin"
VAL_OUT = "data/val.bin"

VAL_RATIO = 0.01  # 1%

tokenizer = Tokenizer.from_file(TOK_PATH)

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# BOS/EOS 없이도 되지만 넣어도 좋음
ids = tokenizer.encode(text).ids
arr = np.array(ids, dtype=np.uint16)  # vocab_size 65535 이하일 때 OK

n = len(arr)
n_val = int(n * VAL_RATIO)

train = arr[:-n_val]
val = arr[-n_val:]

train.tofile(TRAIN_OUT)
val.tofile(VAL_OUT)

print("train tokens:", len(train), "->", TRAIN_OUT)
print("val tokens:", len(val), "->", VAL_OUT)


⸻

8) 04_train_gpt.py (초간단 GPT from scratch 학습)

이 파트는 Karpathy의 build-nanogpt 방식과 거의 동일한 구성(Decoder-only Transformer)입니다.  ￼

여기서는 “완전 초간단”으로 핵심만 넣은 구조로 안내할게요.

✅ 학습 목표:
	•	입력 토큰 x → 다음 토큰 y 예측
	•	CrossEntropyLoss

✅ M4(MPS) 동작:
	•	device 자동 선택: "mps" 가능하면 mps 사용
	•	안 되면 cpu

(원하시면 제가 **완전한 코드 전체(약 200~300줄)**를 다음 메시지에 통째로 드릴게요. 지금은 개념/구성 위주로 설명합니다.)

⸻

9) 05_generate.py (요약/정리 출력 만들기)

중요: 프롬프트를 구조화하면, 작은 모델도 “정리된 답변”을 잘 흉내냅니다.

추천 프롬프트:

[QUESTION]
마닐라에서 응급실 잘하는 병원 추천과 주의사항 정리해줘.
[/QUESTION]

[ANSWER]
요약:
-

이렇게 시작해두면 모델은 다음 토큰 예측을 하다가
	•	체크리스트
	•	항목별 정리
	•	연락처 섹션

같은 형식으로 계속 “이어 쓰는 능력”을 학습합니다.

⸻

10) 학습 파라미터 추천 (M4 기준 “성공 확률 높은 값”)

처음은 무조건 작게 시작하세요.

✅ 1차 성공용
	•	block_size: 256
	•	n_layer: 6
	•	n_head: 6
	•	n_embd: 384
	•	batch_size: 8~16
	•	max_steps: 20,000 정도 (상황 봐서)

✅ 더 고급(느려지지만 좋아짐)
	•	block_size: 512
	•	n_layer: 8
	•	n_embd: 512

⸻

11) “nanoGPT vs build-nanogpt” 선택 가이드
	•	nanoGPT는 강력하지만, repo 자체에서 “old/deprecated” 언급이 있고(최근엔 nanochat 추천)  ￼
	•	“공부 목적(내 손으로 만들기)”는 build-nanogpt가 훨씬 좋습니다.  ￼

⸻

12) 당신의 케이스에서 제일 중요한 성공 포인트 5개
	1.	요약/정리 포맷으로 학습 데이터를 만들어라 (진짜 핵심)
	2.	연락처는 삭제보다 표준화가 더 유용하다
	3.	처음 모델은 작게(6-layer 정도) → 성공 후 키우기
	4.	MPS는 빠르지만 에러 대비해서 PYTORCH_ENABLE_MPS_FALLBACK=1  ￼
	5.	생성 프롬프트는 항상 [QUESTION]...[ANSWER] 형태 유지

⸻


좋습니다. 그럼 이제 “A) 요약 + 체크리스트 + 연락처” 형태로 정리하되, D) 긴 설명형으로 아주 상세하게 풀어주는
✅ 필리핀 교민 정보 제공용 초간단 LLM(from scratch) 제작 파이프라인을 완성본 수준으로 안내할게요.

아래 내용대로 따라하면, 진짜로:
	•	내 txt로 토크나이저 직접 학습
	•	작은 GPT(Transformer Decoder) 처음부터 학습
	•	“요약/정리 포맷”으로 정보를 생성하는 모델까지 완주합니다.

“빈 파일에서 시작해서 GPT-2까지 단계별로 구현하는 학습 루트”는 Karpathy의 build-nanogpt가 가장 교육적으로 유명합니다.  ￼

⸻

0) 당신이 만들 LLM의 최종 출력 포맷(목표)

모델이 이런 스타일로 답하도록 만들 겁니다:

✅ 요약(핵심만)
✅ 체크리스트(실행 순서 / 준비물 / 주의사항)
✅ 연락처(공공기관, 병원, 대사관 등)
✅ 긴 설명형(왜/어떻게/주의점까지 자세히)

예시(모델 출력 목표):

요약
	•	마닐라에서 여권 분실 시: 경찰 신고서 → 대사관 연락 → 여행증명서/재발급 진행 순서로 처리

체크리스트
	•	경찰서 방문해 Police Report 받기
	•	여권 사본/사진 준비
	•	대사관/총영사관 안내 확인
	•	항공권 일정 조정

연락처(공공정보)
	•	주필리핀 대한민국 대사관 TEL: …
	•	관할 경찰서 TEL: …
	•	긴급 병원 TEL: …

상세 설명
	•	왜 Police Report가 먼저 필요한지…
	•	대사관 업무시간/휴일에 따른 플랜B…
	•	재발급과 여행증명서 차이…

⸻

1) Macbook M4(MPS)에서 학습 환경 만들기

1-1. 설치

python -m venv .venv
source .venv/bin/activate

pip install torch tokenizers tqdm numpy

1-2. MPS(GPU) 사용 확인

PyTorch는 Mac에서 MPS 디바이스를 통해 GPU 가속 학습을 지원합니다.  ￼

import torch
print(torch.backends.mps.is_available())
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device =", device)

1-3. (중요) MPS fallback 켜기

MPS에서 지원되지 않는 연산이 나오면 에러가 날 수 있어서, 임시 해결로 CPU fallback을 켤 수 있습니다.  ￼

터미널에서:

export PYTORCH_ENABLE_MPS_FALLBACK=1


⸻

2) 프로젝트 폴더(이대로 만들면 됨)

my_ph_llm/
  data/
    raw.txt                 # 원본 txt (100MB)
    cleaned.txt             # 정리된 txt
    tokenizer.json          # 내 데이터로 학습한 토크나이저
    train.bin               # 학습용 토큰 시퀀스
    val.bin                 # 검증용 토큰 시퀀스
  01_clean_normalize.py
  02_train_tokenizer.py
  03_build_bin_dataset.py
  04_train_gpt_from_scratch.py
  05_generate.py


⸻

3) 데이터 전략: “요약/정리 능력”은 데이터 포맷이 만든다 (핵심)

처음부터 학습하는 LLM은 기본적으로:

“다음 토큰 맞추기”를 잘하게 됩니다.

따라서 “요약을 잘하는 모델”을 만들려면
학습 텍스트 자체에 요약/정리된 답변이 많이 존재해야 해요.

✅ 해결법: txt를 아래 형태로 “학습 가능한 문서 포맷”으로 바꿔줍니다.

⸻

3-1) 추천 학습 포맷 (당신에게 최적)

당신 데이터가 “필리핀 교민 정보”니까 아래 템플릿이 최고입니다:

[DOC]
제목: (내용의 주제)
본문:
(원문 내용)
[/DOC]

[ANSWER]
요약:
- (핵심 3~7개)

체크리스트:
- 준비물:
  - ...
- 해야 할 일:
  - ...
- 주의사항:
  - ...

연락처(공공정보):
- 기관명: ...
  - TEL: ...
  - ADDR: ...
  - WEB: ...

상세 설명:
(여기부터 길게 설명)
[/ANSWER]

📌 여기서 포인트는:
	•	모델이 “요약/체크리스트/연락처/설명” 패턴을 언어 패턴으로 통째로 학습하게 됨
	•	파인튜닝 없이도 출력 형식이 습관처럼 고정됩니다.

⸻

4) 01_clean_normalize.py (전처리 + 표준화)

공공기관 연락처는 “개인정보”라기보다 “정보 제공용 핵심 데이터”이므로
✅ 삭제하지 말고 표준화해서 모델이 배우기 쉽게 만드는 게 좋습니다.

예: 전화, Tel, ☎️ → 전부 TEL: 로 통일

# 01_clean_normalize.py
import re

IN_PATH = "data/raw.txt"
OUT_PATH = "data/cleaned.txt"

def clean_text(s: str) -> str:
    # 제어문자 제거
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", s)

    # 공백 정리
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # 라벨 표준화
    s = re.sub(r"(☎️|전화|TEL|Tel|tel)\s*[:：]?\s*", "TEL: ", s)
    s = re.sub(r"(주소|ADDR|Addr|addr)\s*[:：]?\s*", "ADDR: ", s)
    s = re.sub(r"(홈페이지|웹사이트|사이트|WEB|Web|web)\s*[:：]?\s*", "WEB: ", s)

    return s.strip()

with open(IN_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

cleaned = clean_text(raw)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(cleaned)

print("saved:", OUT_PATH, "chars:", len(cleaned))

실행:

python 01_clean_normalize.py


⸻

5) 02_train_tokenizer.py (BPE 토크나이저를 내 데이터로 직접 학습)

토크나이저 학습은 Hugging Face tokenizers의 Trainer API로 매우 간단합니다.  ￼
(vocab_size, special_tokens 등 옵션 설명이 문서에 나옵니다.)

# 02_train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

IN_PATH = "data/cleaned.txt"
OUT_PATH = "data/tokenizer.json"

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=24000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    show_progress=True,
)

tokenizer.train([IN_PATH], trainer=trainer)
tokenizer.save(OUT_PATH)

print("saved tokenizer:", OUT_PATH)

실행:

python 02_train_tokenizer.py

✅ 한국어 + “필리핀 지명/기관명/용어”가 많기 때문에
내 데이터로 학습한 토크나이저가 매우 유리합니다.

⸻

6) 03_build_bin_dataset.py (토큰화 → 학습용 바이너리 저장)

# 03_build_bin_dataset.py
import numpy as np
from tokenizers import Tokenizer

TEXT_PATH = "data/cleaned.txt"
TOK_PATH = "data/tokenizer.json"

TRAIN_OUT = "data/train.bin"
VAL_OUT = "data/val.bin"

VAL_RATIO = 0.01

tokenizer = Tokenizer.from_file(TOK_PATH)

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

ids = tokenizer.encode(text).ids
arr = np.array(ids, dtype=np.uint16)

n = len(arr)
n_val = int(n * VAL_RATIO)

train = arr[:-n_val]
val = arr[-n_val:]

train.tofile(TRAIN_OUT)
val.tofile(VAL_OUT)

print("train tokens:", len(train))
print("val tokens:", len(val))

실행:

python 03_build_bin_dataset.py


⸻

7) 04_train_gpt_from_scratch.py (진짜 “초간단 GPT” 학습)

여기서부터가 “완전한 LLM”입니다.
구조는 **Decoder-only Transformer (GPT 형태)**이고, 학습은 next-token prediction입니다.

교육용으로는 build-nanogpt가 빈 파일에서 시작해 완성까지 보여줘서 정말 좋습니다.  ￼

7-1) 추천 하이퍼파라미터 (M4 성공 확률 최상)

처음엔 무조건 작게:
	•	block_size = 256
	•	n_layer = 6
	•	n_head = 6
	•	n_embd = 384
	•	batch_size = 8~16
	•	steps = 10k~30k

이 설정이면 MPS에서 “학습이 되는 느낌”을 꽤 빨리 봅니다.

참고: PyTorch에는 MPS 메모리/워터마크 관련 환경변수 문서도 있습니다(너무 자주 OOM 나면 참고).  ￼

⸻

8) 05_generate.py (요약+체크리스트+연락처+상세설명 생성)

생성은 “요약형 LLM”에서 프롬프트 형식이 거의 반 이상입니다.

✅ 추천 프롬프트(고정):

[QUESTION]
필리핀에서 한국인이 병원 가야 할 때 체크리스트와 추천 기준을 자세히 정리해줘.
[/QUESTION]

[ANSWER]
요약:
-

이렇게 [ANSWER] 요약:까지 써주면
모델은 다음 토큰을 이어 쓰면서:
	•	요약 bullet
	•	체크리스트
	•	연락처 섹션
	•	상세설명

을 “습관처럼” 이어 작성하게 됩니다.

⸻

9) 공부할 때 꼭 이해해야 하는 핵심 개념 9개 (진짜 중요)
	1.	Tokenizer

	•	“텍스트 → 정수 토큰” 변환기
	•	vocab_size가 너무 작으면 한국어가 깨지고, 너무 크면 학습이 어려워짐

	2.	Embedding

	•	토큰 ID를 벡터로 바꾸는 표

	3.	Positional Encoding/Embedding

	•	토큰 순서를 모델이 이해하게 만드는 요소(GPT는 위치 임베딩 자주 사용)

	4.	Self-Attention

	•	문장 안에서 “어떤 단어가 중요한지”를 참조하는 핵심 메커니즘

	5.	Feed Forward

	•	attention 다음에 정보를 비선형 변환해서 표현력을 늘림

	6.	Residual + LayerNorm

	•	안정적으로 학습되게 해주는 장치

	7.	Next-token prediction

	•	GPT의 학습 목표: “다음 토큰 맞추기”

	8.	Sampling

	•	temperature / top-k / top-p로 출력 다양성 조절

	9.	데이터 포맷이 곧 모델 능력

	•	요약형으로 학습시키면 요약형으로 말함
	•	체크리스트형이면 체크리스트를 뽑음

⸻

10) 당신의 데이터(필리핀 교민정보)에서 특히 추천하는 “학습 데이터 강화” 팁

✅ “한 문서당 1개의 요약 답변”을 만들어라

raw.txt를 그대로 학습시키면 “텍스트 이어쓰기”만 잘합니다.

대신 아래처럼 “질문→답변” 구조 샘플을 많이 만들면 좋아요:
	•	[QUESTION] 필리핀 ACR I-Card 연장 방법 정리해줘
	•	[QUESTION] 마닐라에서 경찰 신고서 받는 법
	•	[QUESTION] 교민이 자주 쓰는 긴급 연락처 모음

이걸 수천 개 만들어도 되고, 자동으로 생성해도 됩니다(규칙 기반).

✅ 연락처는 “정확한 포맷”으로 고정

예:

연락처(공공정보):
- 주필리핀 대한민국 대사관
  - TEL: ...
  - ADDR: ...
  - WEB: ...

이렇게 하면 모델이 출력도 비슷하게 하려 합니다.

⸻

11) 다음 단계: 제가 “학습 코드 전체 완성본”을 드릴까요?

당신이 원한 건 “최대한 상세 정보” + “완전 초간단 LLM 완성”이니까,

원하시면 다음 답변에서 제가:

✅ 04_train_gpt_from_scratch.py 전체 코드(학습/체크포인트/재개 포함)
✅ 05_generate.py 전체 코드(temperature/top-k 포함)
✅ 실행 커맨드 + 예상 소요시간 + 튜닝 포인트

를 그대로 복붙하면 돌아가게 완성본으로 드릴게요.

⸻

좋습니다 👍
아래는 “raw.txt(한 파일에 줄줄이 섞임)” → 전처리/분할 → 토크나이저(BPE) 학습 → GPT(Decoder Transformer) from scratch 학습 → 요약/정리+상세설명 생성까지 되는 완성본 코드입니다.

Macbook M4(MPS)에서 돌아가도록 만들었고, PyTorch MPS에서 미지원 연산이 나오면 CPU fallback을 켤 수도 있습니다.  ￼
토크나이저는 Hugging Face tokenizers의 BPE Trainer 기반입니다.  ￼
(구현 스타일은 Karpathy의 from-scratch GPT 구현 흐름을 참고하면 학습에 도움됩니다.  ￼)

⸻

✅ 0) 프로젝트 폴더 구조

my_ph_llm/
  data/
    raw.txt
  01_prepare_samples.py
  02_train_tokenizer.py
  03_build_bin_dataset.py
  04_train_gpt_from_scratch.py
  05_generate.py


⸻

✅ 1) 설치 & 실행 준비

cd my_ph_llm
python -m venv .venv
source .venv/bin/activate

pip install torch tokenizers tqdm numpy

(권장) MPS fallback 켜기:  ￼

export PYTORCH_ENABLE_MPS_FALLBACK=1


⸻

✅ 2) 01_prepare_samples.py

raw.txt가 “줄줄이 섞인 형태”이므로 → 자동으로 문서 덩어리로 쪼개고 → 요약/체크리스트/연락처/상세설명 포맷으로 “학습 텍스트”를 만듭니다.

from-scratch 학습에서는 “요약 능력”이 모델에 저절로 생기지 않고, 이런 출력 포맷을 학습 텍스트로 만들어줘야 합니다.

# 01_prepare_samples.py
import re
import random

RAW_PATH = "data/raw.txt"
OUT_PATH = "data/samples.txt"

random.seed(42)

def normalize_text(s: str) -> str:
    # 제어문자 제거
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", s)
    # 공백 정리
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # 라벨 표준화(연락처를 학습시키기 좋게)
    s = re.sub(r"(☎️|전화|TEL|Tel|tel)\s*[:：]?\s*", "TEL: ", s)
    s = re.sub(r"(주소|ADDR|Addr|addr)\s*[:：]?\s*", "ADDR: ", s)
    s = re.sub(r"(홈페이지|웹사이트|사이트|WEB|Web|web)\s*[:：]?\s*", "WEB: ", s)

    return s.strip()

def split_into_chunks(text: str, min_len=600, max_len=1800):
    """
    줄줄이 섞인 텍스트를 문서 덩어리로 쪼개는 초간단 휴리스틱.
    - 빈 줄 2개 이상을 경계로 1차 분리
    - 너무 짧은 건 합치고, 너무 긴 건 잘라서 나눔
    """
    parts = re.split(r"\n\s*\n", text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks = []
    buf = ""

    for p in parts:
        if len(buf) < min_len:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            chunks.append(buf)
            buf = p

    if buf:
        chunks.append(buf)

    # 너무 긴 chunk는 강제로 쪼갬
    final = []
    for c in chunks:
        if len(c) <= max_len:
            final.append(c)
        else:
            for i in range(0, len(c), max_len):
                final.append(c[i:i+max_len].strip())
    return [c for c in final if len(c) >= min_len]

QUESTION_BANK = [
    "이 내용을 교민이 이해하기 쉽게 요약해줘.",
    "필리핀에서 한국인 입장에서 핵심만 정리해줘.",
    "체크리스트와 주의사항 중심으로 정리해줘.",
    "연락처가 있다면 함께 정리해줘.",
    "초보 교민을 위한 안내문처럼 상세히 설명해줘.",
]

def build_training_sample(doc: str) -> str:
    # 연락처 후보 추출(너무 정교할 필요 없음)
    tels = re.findall(r"TEL:\s*([0-9+\-\s()]{6,})", doc)
    webs = re.findall(r"WEB:\s*(\S+)", doc)

    # “요약/체크리스트/연락처/상세설명” 포맷을 강제로 학습시키는 텍스트
    q = random.choice(QUESTION_BANK)

    # 아주 간단한 “요약 문장” 힌트(학습 안정성을 위해 일부는 남김)
    # ※ from scratch라서 여기서 요약을 완벽히 만들 필요는 없고,
    #    "형식"을 강하게 주입하는 게 목적.
    hint = doc[:200].replace("\n", " ").strip()

    contacts = ""
    if tels or webs:
        contacts += "연락처(공공정보):\n"
        if tels:
            for t in list(dict.fromkeys(tels))[:3]:
                contacts += f"- TEL: {t.strip()}\n"
        if webs:
            for w in list(dict.fromkeys(webs))[:2]:
                contacts += f"- WEB: {w.strip()}\n"
    else:
        contacts = "연락처(공공정보):\n- (문서에 명시된 연락처가 없거나 확인 필요)\n"

    sample = f"""[QUESTION]
{q}
[/QUESTION]

[DOC]
{doc}
[/DOC]

[ANSWER]
요약:
- {hint} ...

체크리스트:
- 해야 할 일:
  - (1) 핵심 행동 1
  - (2) 핵심 행동 2
- 준비물:
  - (1) 필요한 서류/정보
- 주의사항:
  - (1) 실수하기 쉬운 점

{contacts}
상세 설명:
이 문서는 필리핀(재필 교민/동포) 생활 정보 관점에서 중요한 내용을 담고 있습니다.
위 요약과 체크리스트를 기준으로 실제 상황에 맞게 단계별로 진행하세요.
문서에 포함된 규정/비용/업무시간/관할지역 정보가 있다면 반드시 최신 정보를 재확인하는 것이 좋습니다.
[/ANSWER]

"""
    return sample

def main():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = normalize_text(raw)
    chunks = split_into_chunks(raw)

    # 너무 많으면 일부만(학습 속도 위해)
    # 100MB면 chunk가 많을 수 있으니, 우선 30k개 이하를 추천
    if len(chunks) > 30000:
        chunks = chunks[:30000]

    print("chunks:", len(chunks))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(build_training_sample(c))

    print("saved:", OUT_PATH)

if __name__ == "__main__":
    main()

실행:

python 01_prepare_samples.py

✅ 결과: data/samples.txt (요약/체크리스트/연락처/상세설명 학습용)

⸻

✅ 3) 02_train_tokenizer.py (내 데이터로 BPE 토크나이저 학습)

BPE Trainer의 핵심 옵션(vocab_size, special_tokens 등)은 Hugging Face 문서에 있습니다.  ￼

# 02_train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

IN_PATH = "data/samples.txt"
OUT_PATH = "data/tokenizer.json"

VOCAB_SIZE = 24000

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    show_progress=True,
)

tokenizer.train([IN_PATH], trainer=trainer)
tokenizer.save(OUT_PATH)

print("saved tokenizer:", OUT_PATH)

실행:

python 02_train_tokenizer.py


⸻

✅ 4) 03_build_bin_dataset.py (토큰화 → train/val 바이너리)

# 03_build_bin_dataset.py
import numpy as np
from tokenizers import Tokenizer

TEXT_PATH = "data/samples.txt"
TOK_PATH = "data/tokenizer.json"

TRAIN_OUT = "data/train.bin"
VAL_OUT = "data/val.bin"

VAL_RATIO = 0.01

tokenizer = Tokenizer.from_file(TOK_PATH)

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

ids = tokenizer.encode(text).ids
arr = np.array(ids, dtype=np.uint16)

n = len(arr)
n_val = int(n * VAL_RATIO)

train = arr[:-n_val]
val = arr[-n_val:]

train.tofile(TRAIN_OUT)
val.tofile(VAL_OUT)

print("train tokens:", len(train), "->", TRAIN_OUT)
print("val tokens:", len(val), "->", VAL_OUT)

실행:

python 03_build_bin_dataset.py


⸻

✅ 5) 04_train_gpt_from_scratch.py (핵심: GPT from scratch 학습 코드)

아래는 완전한 초간단 GPT 학습 코드입니다.
	•	MPS 자동 사용
	•	체크포인트 저장/재개
	•	학습 중간 generate 샘플 출력

# 04_train_gpt_from_scratch.py
import os
import math
import time
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    tok_path: str = "data/tokenizer.json"
    out_dir: str = "checkpoints"

    # model
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

    # train
    batch_size: int = 16
    lr: float = 3e-4
    max_steps: int = 20000
    eval_interval: int = 500
    eval_iters: int = 100
    grad_clip: float = 1.0

    # sampling (debug)
    sample_every_eval: bool = True
    sample_max_new_tokens: int = 250
    temperature: float = 0.9
    top_k: int = 50

cfg = CFG()

os.makedirs(cfg.out_dir, exist_ok=True)

# -----------------------------
# Device
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print("device =", device)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = Tokenizer.from_file(cfg.tok_path)
vocab_size = tokenizer.get_vocab_size()
print("vocab_size =", vocab_size)

# -----------------------------
# Data loader (bin)
# -----------------------------
train_data = np.memmap(cfg.train_bin, dtype=np.uint16, mode="r")
val_data = np.memmap(cfg.val_bin, dtype=np.uint16, mode="r")

def get_batch(split: str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+cfg.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+cfg.block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# -----------------------------
# Model (GPT)
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for b in self.blocks:
            x = b(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

model = GPT(
    vocab_size=vocab_size,
    block_size=cfg.block_size,
    n_layer=cfg.n_layer,
    n_head=cfg.n_head,
    n_embd=cfg.n_embd,
    dropout=cfg.dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

# -----------------------------
# Checkpoint load/save
# -----------------------------
ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
start_step = 0

if os.path.exists(ckpt_path):
    print("loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    start_step = ckpt.get("step", 0)
    print("resume from step", start_step)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def decode_ids(ids):
    return tokenizer.decode(ids)

def quick_sample():
    prompt = """[QUESTION]
필리핀에서 한국인이 병원 가야 할 때 체크리스트와 주의사항을 상세히 정리해줘.
[/QUESTION]

[ANSWER]
요약:
-"""
    enc = tokenizer.encode(prompt).ids
    x = torch.tensor(enc, dtype=torch.long, device=device)[None, :]
    y = model.generate(
        x,
        max_new_tokens=cfg.sample_max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
    )
    out = decode_ids(y[0].tolist())
    return out

print("start training...")
t0 = time.time()

pbar = tqdm(range(start_step, cfg.max_steps))
for step in pbar:
    x, y = get_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    if step % 50 == 0:
        pbar.set_description(f"step {step} loss {loss.item():.4f}")

    if step > 0 and step % cfg.eval_interval == 0:
        losses = estimate_loss()
        print(f"\nstep {step} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}")

        # save ckpt
        torch.save({
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
            "cfg": cfg.__dict__,
        }, ckpt_path)
        print("saved:", ckpt_path)

        if cfg.sample_every_eval:
            print("\n--- sample ---")
            print(quick_sample()[:2500])
            print("--------------\n")

print("done. time:", time.time() - t0)

학습 실행:

python 04_train_gpt_from_scratch.py


⸻

✅ 6) 05_generate.py (학습된 모델로 생성)

# 05_generate.py
import os
import torch
from tokenizers import Tokenizer

# 반드시 학습 코드의 설정과 동일해야 함
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.1

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = torch.nn.Linear(n_embd, 3 * n_embd)
        self.proj = torch.nn.Linear(n_embd, n_embd)

        self.attn_drop = torch.nn.Dropout(dropout)
        self.resid_drop = torch.nn.Dropout(dropout)

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        import math
        import torch.nn.functional as F

        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(torch.nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = torch.nn.Linear(n_embd, 4 * n_embd)
        self.proj = torch.nn.Linear(4 * n_embd, n_embd)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        import torch.nn.functional as F
        return self.drop(self.proj(F.gelu(self.fc(x))))

class Block(torch.nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = torch.nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(torch.nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_emb = torch.nn.Embedding(block_size, n_embd)
        self.drop = torch.nn.Dropout(dropout)

        self.blocks = torch.nn.ModuleList([
            Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)
        ])
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for b in self.blocks:
            x = b(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        import torch.nn.functional as F

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print("device =", device)

tokenizer = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

model = GPT(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_embd=N_EMBD,
    dropout=DROPOUT,
).to(device)

ckpt = torch.load("checkpoints/ckpt.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

prompt = """[QUESTION]
필리핀에서 한국인이 경찰서를 가야 하는 대표적인 상황과 절차를 상세히 정리해줘.
[/QUESTION]

[ANSWER]
요약:
-"""

x = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long, device=device)[None, :]
y = model.generate(x, max_new_tokens=400, temperature=0.9, top_k=50)

print(tokenizer.decode(y[0].tolist())[:4000])

실행:

python 05_generate.py


⸻

✅ “한 파일에 줄줄이 섞인 raw.txt”에서 결과를 더 좋게 만드는 팁 (중요)

1) 학습 품질이 확 좋아지는 핵심: “샘플 수 늘리기”

01_prepare_samples.py에서 chunk를 많이 만들수록 좋아요.
	•	3,000개: “되는 느낌”
	•	10,000개: “형식이 안정”
	•	30,000개: “요약/정리 습관 강해짐”

2) “연락처”는 삭제하지 말고 포맷 고정(이게 학습에 유리)

TEL:, ADDR:, WEB: 같은 라벨이 계속 반복되면
모델이 자동으로 “연락처 섹션”을 만들기 시작합니다.

3) 모델이 너무 헛소리하면
	•	temperature를 0.7로 낮추기
	•	top_k를 20으로 줄이기
→ 더 보수적으로 생성합니다.

⸻

✅ 지금 바로 실행 순서 (정리)

# 1) 학습 샘플 생성
python 01_prepare_samples.py

# 2) 토크나이저 학습
python 02_train_tokenizer.py

# 3) 바이너리 데이터셋 생성
python 03_build_bin_dataset.py

# 4) LLM 학습
python 04_train_gpt_from_scratch.py

# 5) 생성 테스트
python 05_generate.py


