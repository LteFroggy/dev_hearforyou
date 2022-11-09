import librosa
import os
import numpy as np
import soundfile
import csv
import settings as set
from pathlib import Path
from matplotlib import pyplot as plt

dataPath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "data")

def getLabelData(csvPath) :
    label = [[]]
    
    with open(csvPath, 'r') as file :
        reader = csv.reader(file)
        for line in reader :
            label.append(line)

    return label

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

def removeSilence_forDrawing(wav_loaded) :
    wav_splited = librosa.effects.split(wav_loaded, top_db = getSTD(wav_loaded))
    output = np.zeros(len(wav_loaded))
    count = 0
    for section in wav_splited :
        count += len(wav_loaded[section[0] : section[1]])
        output[section[0] : section[1]] = wav_loaded[section[0] : section[1]]
    
    return np.array(output), (True if count >= 0.5 * set.SAMPLE_RATE else False)

def getInfo(wav_loaded) :
    dbs = librosa.amplitude_to_db(wav_loaded)
    print("Shape :", wav_loaded.shape)
    print("Context :", wav_loaded)
    print("Sum :", wav_loaded.sum())
    print("Caculated DB :", abs(dbs.min()), "DB")
    print("Average :",wav_loaded.mean())
    print("Maximum : ", wav_loaded.max())
    print("Minmum :", wav_loaded.min())

def trim_audio_data(filePath) :
    y = librosa.load(filePath, sr = set.SAMPLE_RATE)[0]

def saveAsPhoto(wav_loaded, savePath, fileName) :
    removed, flag = removeSilence_forDrawing(wav_loaded)
    if flag :
        time = np.linspace(0, len(wav_loaded)/set.SAMPLE_RATE, len(wav_loaded))
        plt.plot(time, wav_loaded, color = "blue")
        plt.plot(time, removed, color = "blue")
        plt.title(fileName + " [removed Under" + str(getSTD(wav_loaded)) + " DB]")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.savefig(savePath)
    else :
        None

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
        for i in range(1, saveFileCount) :
            cuttedFiles.append(wav_loaded[set.SAMPLE_RATE * set.CUT_SEC * i : set.SAMPLE_RATE * set.CUT_SEC * (i + 1)])

        # 1초가 안되는 남은 부분은, 0.5초 이하라면 버리고 이상이라면 이어붙여서 append
        leftover = wav_loaded[set.SAMPLE_RATE * saveFileCount : ]
        if len(leftover) < (set.CUT_SEC / 2) * set.SAMPLE_RATE :
            None
        else :
             lackLength = (set.CUT_SEC * set.SAMPLE_RATE) - len(leftover)
             leftover = np.append(leftover, leftover[ : lackLength])
             cuttedFiles.append(leftover)

        return cuttedFiles
    # 딱 1초라면 그냥 리턴
    else :
        return wav_loaded

def getFolderName(file) :
    return os.path.dirname(os.path.realpath(file))


def getMFCC(wav_file) :
    return np.mean(librosa.feature.mfcc(y=wav_file, sr=set.SAMPLE_RATE, n_mfcc=set.N_MFCCS).T, axis = 0)