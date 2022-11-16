import os
import csv
import pickle
from tqdm import tqdm
from pathlib import Path

def getMid(soundName) :
    path = os.path.join(os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "csv"), "class_labels_indices.csv")
    soundName = soundName

    with open(path, "r") as file :
        reader = csv.reader(file)
        
        for line in reader :
            if line[2] == soundName :
                return line[1]
    

def main(soundName) :
    # 3개의 csv파일을 읽어서 원하는 라벨을 가진 파일을 모두 꺼내 pickle로 저장한다.
    basePath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "data")
    labelPath = os.path.join(basePath, soundName)

    if not (os.path.isdir(labelPath)) :
        os.mkdir(labelPath)

    labelPath = os.path.join(labelPath, "summary.pkl")
    csvPath = os.path.join(basePath, "csv")
    fileList = ["balanced_train_segments.csv", "eval_segments.csv", "unbalanced_train_segments.csv"]
    
    mid = getMid()

    # fileName = "class_labels_indices.csv"
    # with open(os.path.join(basePath, fileName), "r") as file :
    #     readedFile = csv.reader(file)
    #     iterater = iter(readedFile)
    #     value = next(iterater)
    #     print(value)
    #     print(len(value))
    #     for i in range(len(value)) :
    #         print(value[i].strip(" \""))

    values = []

    for num, fileName in enumerate(fileList, start = 1) :
        csvPath = os.path.join(basePath, fileName)
        with open(csvPath, 'r') as file :
            reader = csv.reader(file)
            for line in tqdm(reader, desc = f"{fileName} 파일 라벨 처리 중 [{num} / {len(fileList)}]") :
                for i in range(len(line)) :
                    line[i] = line[i].strip(" \"")

                num = -1
                for i in range(3, len(line)) :
                    if line[i] == mid :
                        num = i

                if num != -1 :
                    values.append([line[0], line[1], line[2], num])

    print(f" 총 길이는 {len(values)} 입니다.")

    with open(labelPath, "wb") as file :
        pickle.dump(values, file)