import os
import pickle
import librosa
import warnings
import soundfile
from tqdm import tqdm
from pathlib import Path

def main(soundName) :
    print("파일 커팅 시작")
    warnings.filterwarnings('ignore')

    basePath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "data")
    basePath = os.path.join(basePath, soundName)
    dataPath = os.path.join(basePath, "downloaded")
    savePath = os.path.join(basePath, "cutted")

    if not (os.path.isdir(savePath)) :
        os.mkdir(savePath)

    labelPath = os.path.join(basePath, "name_label.pkl")

    with open(labelPath, "rb") as file :
        labels = pickle.load(file)

    dataFiles = os.listdir(dataPath)

    for fileName in tqdm(dataFiles, desc = f"받은 파일을 자르는 중") :
        filePath = os.path.join(dataPath, fileName)
        saveFilePath = os.path.join(savePath, fileName)
        
        
        try :
            loadedFile, sr = librosa.load(filePath)
            times = labels[os.path.splitext(fileName)[0]]
            cuttedFile = loadedFile[int(sr * float(times[0])) : int(sr * float(times[1]))]
            soundfile.write(file = saveFilePath, data = cuttedFile, samplerate = sr, format = 'wav')
        except Exception as e:
            print(f"Error : {e}")
            continue