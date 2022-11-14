import os
import shutil
import librosa
import soundfile
import numpy as np
import settings as set
from pathlib import Path

dataPath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "data")

def getSTD(wav_loaded) :
    db_detected = librosa.amplitude_to_db(wav_loaded)
    maxVal = abs(db_detected.min()) if abs(db_detected.min()) > abs(db_detected.min()) else abs(db_detected.min())
    return 2200 / maxVal

def loadWAV(filePath) :
    return librosa.load(filePath, sr = set.SAMPLE_RATE)[0]
    
def removeSilence(wav_loaded) :
    wav_splited = librosa.effects.split(wav_loaded, top_db = getSTD(wav_loaded))
    output = []
    for section in wav_splited :
        #print(section)
        output.extend(wav_loaded[section[0] : section[1]])

    return np.array(output)

def saveFile(savePath, wavFile) :
    soundfile.write(file = savePath, data = wavFile, samplerate = set.SAMPLE_RATE, format = 'wav')

    
def regulateFile(wav_loaded) :
    cuttedFiles = []
    # 들어온 파일이 1초 미만이면 앞을 잘라서 1초가 되게 이어붙인 이후 저장
    if len(wav_loaded) < set.CUT_SEC * set.SAMPLE_RATE :

        # For Error Checking
        if len(wav_loaded) < (set.CUT_SEC / 2) * set.SAMPLE_RATE :
            raise Exception()

        lackLength = (1 * set.SAMPLE_RATE) - len(wav_loaded)
        wav_loaded = np.append(wav_loaded, wav_loaded[: lackLength])
        cuttedFiles.append(wav_loaded)
        
        return cuttedFiles

    # 1초 초과라면? 먼저 몇 개의 파일이 될 지 확인하고, 1초씩 잘라서 append
    elif len(wav_loaded) > 1 * set.SAMPLE_RATE :
        saveFileCount = int(len(wav_loaded) / (set.CUT_SEC * set.SAMPLE_RATE))
        for i in range(saveFileCount) :
            cuttedFiles.append(wav_loaded[set.SAMPLE_RATE * set.CUT_SEC * i : set.SAMPLE_RATE * set.CUT_SEC * (i + 1)])

        # 1초가 안되는 남은 부분은, 0.5초 이하라면 버리고 이상이라면 이어붙여서 append
        leftover = wav_loaded[set.SAMPLE_RATE * (saveFileCount) : ]
        if len(leftover) < (set.CUT_SEC / 2) * set.SAMPLE_RATE :
            None
        else :
             lackLength = (set.CUT_SEC * set.SAMPLE_RATE) - len(leftover)
             leftover = np.append(leftover, leftover[ : lackLength])
             cuttedFiles.append(leftover)

        return cuttedFiles
    # 딱 1초라면 그냥 리턴
    else :
        cuttedFiles.append(wav_loaded)
        return cuttedFiles


# CNN에서 추가됨

def get_melspectrogram_db(wavFile) :
    spectrum = librosa.feature.melspectrogram(y = wavFile, sr = 22050)
    spectrum_db = librosa.power_to_db(spectrum)
    return spectrum_db

def spec_to_image(spec, eps=1e-6):
    # 들어온 spec의 평균값 계산
    mean = spec.mean()

    # spec의 표준편차 계산
    std = spec.std()

    # spec_norm = spec - 평균 / 표준편차 + eps
    spec_norm = (spec - mean) / (std + eps)

    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def removeUsedFolder(soundPath) :
    print(f"{soundPath}폴더 삭제 중")
    shutil.rmtree(soundPath)
    print(f"{soundPath}폴더 삭제 완료")