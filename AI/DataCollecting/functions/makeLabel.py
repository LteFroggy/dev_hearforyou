import os
import pickle
from tqdm import tqdm
from pytube import YouTube
from pathlib import Path

def main(soundName) :
    basePath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "data")
    basePath = os.path.join(basePath, soundName)
    savePath = os.path.join(basePath, "downloaded")
    pklPath = os.path.join(basePath, "summary.pkl")

    if not (os.path.isdir(savePath)) :
        os.mkdir(savePath)

    with open(pklPath, "rb") as file : 
        labels = pickle.load(file)

    new_labels = {}
    basicLink = "https://www.youtube.com/watch?v="

    for line in tqdm(labels, desc = f"라벨 변경 중") :
        link = basicLink + str(line[0])
        yt = YouTube(link)

        try :
            new_labels[yt.title] = [line[1], line[2]]
        except Exception as e :
            continue

    