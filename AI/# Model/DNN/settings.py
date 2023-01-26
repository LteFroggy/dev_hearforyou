import os
import wandb
from pathlib import Path

wandb.init(project="AI_DNN")

wandb.config = {
  "learning_rate": 0.005,
  "batch_size": 64,
  "epochs": 200,
}


# 기본 경로
dataPath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent.parent, "data")

# 음원파일을 불러올 때의 Sample Rate
SAMPLE_RATE = 22050

# 음성파일을 자를 시간의 단위(아직 변경을 지원하지 않음)
CUT_SEC = 1

# 음성파일에서 추출할 MFCC의 갯수
N_MFCCS = 50

# 라벨 종류 -------------------------------------------------------------------------
main_label = {
    0 : "개 짖는 소리",
    1 : "사이렌",
    2 : "차량 경적 소리",
    3 : "총소리",
    4 : "박수",
    5 : "남성",
    6 : "여성",
    7 : "아이",
    8 : "비명"
}

test_Labels = {
    10 : "t1"
}

sample_label = {
    0 : "개",
    1 : "차량사이렌",
    2 : "차량경적"
}
# 라벨 종류 -------------------------------------------------------------------------

# 모델의 구분을 위해 붙일 값
# 예시처럼 220910이면 결과 모델은 model_N Epochs_221101.pt 의 형태로 나옴
model_label = "DNN_230124"

# AI용 학습 가중치
LEARNING_RATE = 0.005
BATCH_SIZE = 64
EPOCHS = 200