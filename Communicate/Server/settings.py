import os
from pathlib import Path

# 음원파일을 불러올 때의 Sample Rate
SAMPLE_RATE = 22050

# 음성파일에서 추출할 MFCC의 갯수
N_MFCCS = 50

label = {
    0 : "개 짖는 소리",
    1 : "차량 사이렌 소리",
    2 : "차량 경적 소리",
    3 : "총소리",
    4 : "박수소리",
    5 : "남성 대화",
    6 : "여성 대화",
    7 : "아이 대화",
    8 : "비명소리",
}
