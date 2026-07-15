# pescli — getpes.com 파이 인공지능(FAI) 로컬 학습 CLI 패키지
#
# 원칙: 서버(getpes.com)는 인증·DB 입출력만 한다.
#       토크나이저 학습, GPT 트레이닝, 병합, ONNX 빌드는 전부 이 컴퓨터에서 수행하고
#       결과만 API 로 업로드한다.
#
# 명령: fai login / pull / train / push / build / merge
