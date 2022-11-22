# 음원파일을 불러올 때의 Sample Rate
SAMPLE_RATE = 22050

# 음성파일을 자를 시간의 단위(아직 변경을 지원하지 않음)
CUT_SEC = 1

# 음성파일에서 추출할 MFCC의 갯수
N_MFCCS = 50

# 라벨 종류
labels = {
    0 : "개 짖는 소리",
    1 : "사이렌",
    2 : "차량 경적 소리",
    3 : "총소리"
}

LEARNING_RATE = 0.005
BATCH_SIZE = 32
EPOCHS = 10