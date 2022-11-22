import os
# import wandb
from pathlib import Path

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ wandb @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# wandb.init(project="AI_CNN")

# wandb.config = {
#   "learning_rate": 0.001,
#   "batch_size": 256,
#   "epochs": 200,
# }

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 200

# 음원파일을 불러올 때의 Sample Rate
SAMPLE_RATE = 22050

# 이미지파일을 자를 가로, 세로 픽셀의 길이
CUT_SIZE = 32

# 라벨 종류
main_label = {
    0 : "개 짖는 소리",
    1 : "사이렌",
    2 : "차량 경적 소리",
    3 : "총소리",
    4 : "Fire alarm",
    5 : "Screaming",
    6 : "Skidding"
}

model_label = "CNN_221117_resized"

dataPath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent.parent, "data")
dataPath