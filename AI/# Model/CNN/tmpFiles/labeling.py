import os
import pickle
import shutil
import unicodedata
import settings as set
from tqdm import tqdm


def labeling(target, labels) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    dataPath = os.path.join(basePath, "2-2. CuttedPhoto")
    savePath = os.path.join(basePath, "2-3. ModelData")

    if not(os.path.isdir(savePath)) :
        os.mkdir(savePath)

    dataFolderList = os.listdir(dataPath)
    totalFolderCount = len(dataFolderList)
    label_train = []
    label_test = []

    for folderCount, folderName in enumerate(dataFolderList) :
        dataFolderPath = os.path.join(dataPath, folderName)
        trainFolderPath = os.path.join(savePath, "trainData")
        testFolderPath = os.path.join(savePath, "testData")

        if not(os.path.isdir(trainFolderPath)) :
            os.mkdir(trainFolderPath)

        if not(os.path.isdir(testFolderPath)) :
            os.mkdir(testFolderPath)

        try :
            dataFileList = os.listdir(dataFolderPath)
        except :
            continue

        folderLabel = -1
        for key in labels.keys() :
            # 값을 정규화해주지 않으면 길이가 달라 같은 문자열로 인식하지 못하는 오류 발생 -> 수정
            forderName_normalized = unicodedata.normalize("NFC", folderName)
            value_normalized = unicodedata.normalize("NFC", labels[key])

            if forderName_normalized == value_normalized :
                folderLabel = key
                print(f"{folderName}의 Label은 {folderLabel}입니다")
            
        if (folderLabel == -1) :
            print(f"{folderName}은(는) 적절한 폴더가 아니어서 넘어갔습니다")
            continue

        for count, fileName in enumerate(tqdm(dataFileList, desc = f"{folderName} 폴더 [{folderCount} / {totalFolderCount}] 라벨링 진행 중")) :
            dataFilePath = os.path.join(dataFolderPath, fileName)
            if (fileName == ".DS_Store") :
                continue
            
            # Test로 갈 것
            if count % 10 == 0 :
                shutil.copy(dataFilePath, os.path.join(testFolderPath, fileName))
                label_test.append([fileName, folderLabel])
            # Train으로 갈 것
            else :
                shutil.copy(dataFilePath, os.path.join(trainFolderPath, fileName))
                label_train.append([fileName, folderLabel])
            
        # pickle을 이용해 만들어진 LabelFile을 저장
        trainingLabelPath = os.path.join(trainFolderPath, "labels.pkl")
        testingLabelPath = os.path.join(testFolderPath, "labels.pkl")

        # 트레이닝용 라벨 저장
        with open(trainingLabelPath, "wb") as file :
            pickle.dump(label_train, file)

        # 테스팅용 라벨 저장
        with open(testingLabelPath, "wb") as file :
            pickle.dump(label_test, file)