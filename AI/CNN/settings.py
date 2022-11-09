import os
import wandb
from pathlib import Path


wandb.init(project="AI_CNN")

LEARNING_RATE = 0.005
BATCH_SIZE = 32
EPOCHS = 200

wandb.config = {
  "learning_rate": 0.005,
  "batch_size": 32,
  "epochs": 200,
}

# 음원파일을 불러올 때의 Sample Rate
SAMPLE_RATE = 22050

# 음성파일을 자를 시간의 단위(아직 변경을 지원하지 않음)
CUT_SEC = 1

# 라벨 종류
UrbanSounds_labels = {
    0 : "개 짖는 소리",
    1 : "사이렌",
    2 : "차량 경적 소리",
    3 : "총소리"
}

test_Labels = {
    0 : "TestFold"
}

TS_Labels = {
    0 : "14.개",
    1 : "2.차량사이렌",
    2 : "3.차량경적"
}

dataPath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "data")