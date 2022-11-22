import os
from pathlib import Path
from tqdm import tqdm

def main(soundName) :
    print("파일명 변경 시작")
    basePath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "data")
    basePath = os.path.join(basePath, soundName)
    dataPath = os.path.join(basePath, "downloaded")

    files = os.listdir(dataPath)

    for file in tqdm(files, "파일명 변경 중") :
        fileName_noext = os.path.splitext(file)[0]
        os.rename(os.path.join(dataPath, file), os.path.join(dataPath, fileName_noext) + ".mp3")
