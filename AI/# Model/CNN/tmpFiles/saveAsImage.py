import os
import librosa
import numpy as np
import settings as set
import librosa.display as display
import wav_functions as func
from tqdm import tqdm
from matplotlib import pyplot as plt


def saveAsImage(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    soundPath = os.path.join(basePath, "2. RegulatedSound")
    savePath = os.path.join(basePath, "2-1. RegulatedPhoto")

    if not(os.path.isdir(savePath)) :
        os.mkdir(savePath)

    for folderCount, folderName in enumerate(os.listdir(soundPath)) : 
        soundFolderPath = os.path.join(soundPath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        if not (os.path.isdir(saveFolderPath)) :
            os.mkdir(saveFolderPath)
        
        try :
            soundFiles = os.listdir(soundFolderPath)
        except :
            continue

        for fileName in tqdm(soundFiles, desc = f"{folderName}폴더 [{folderCount} / {len(os.listdir(soundPath))}] 이미지화 진행 중") :
            soundFilePath = os.path.join(soundFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, (fileName[:-4] + ".png"))

            try :
                wavFile = librosa.load(soundFilePath, sr = 22050)[0]
            except :
                continue
            
            plt.cla()
            wavSpec = func.get_melspectrogram_db(wavFile)
            wavImg = func.spec_to_image(wavSpec)
            display.specshow(wavImg)
            plt.savefig(saveFilePath, bbox_inches="tight", pad_inches = 0)