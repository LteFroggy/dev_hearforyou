import os
import librosa
import numpy as np


filePath = os.path.dirname(os.path.realpath(__file__))
filePath = os.path.join(filePath, "data")
filePath = os.path.join(filePath, "test")
filePath = os.path.join(filePath, "1. FilteredSound")
filePath = os.path.join(filePath, "TestFold")

fileList = os.listdir(filePath)

for file in fileList :
    try :
        loadedFile = librosa.load(os.path.join(filePath, file))[0]
    except :
        continue
    mfcc = librosa.feature.mfcc(y = loadedFile, sr = 22050, n_mfcc = 100)
    print(f"FileName : {file}, MFCC's shape = {mfcc.shape}")




# return np.mean(librosa.feature.mfcc(y=wav_file, sr=set.SAMPLE_RATE, n_mfcc=set.N_MFCCS).T, axis = 0)