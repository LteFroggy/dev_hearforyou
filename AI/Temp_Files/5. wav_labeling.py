import os
import shutil
import pickle
import settings as set

def main(target) :
    # 베이스 경로 지정
    baseFolderPath = os.path.dirname(os.path.realpath(__file__))
    baseFolderPath = os.path.join(baseFolderPath, target)
    basePath = os.path.join(baseFolderPath, "data")
    dataPath = os.path.join(basePath, "4. MFCCs")
    dataPath = os.path.join(dataPath, str(set.N_MFCCS))

    # 저장할 기본 경로 지정
    saveBasePath = os.path.join(basePath, "5. ModelData")

    # 저장할 경로에 폴더가 없다면 생성
    if not(os.path.isdir(saveBasePath)) :
        os.mkdir(saveBasePath)

    # MFCC별로 다른 곳에 저장해야 하니 MFCC도 적용해서 경로 재설정
    savePath = os.path.join(saveBasePath, str(set.N_MFCCS))

    # 저장할 경로에 폴더가 없다면 생성
    if not(os.path.isdir(savePath)) :
        os.mkdir(savePath)

    dataFolders = os.listdir(dataPath)

    # 테스팅데이터와 트레이닝 데이터를 나눠 저장할 예정.
    trainingSavePath = os.path.join(savePath, "trainData")
    testingSavePath = os.path.join(savePath, "testData")

    # 각각의 폴더가 필요하다면 만들기
    if not(os.path.isdir(trainingSavePath)) :
        os.mkdir(trainingSavePath)

    if not(os.path.isdir(testingSavePath)) :
        os.mkdir(testingSavePath)

    trainingLabelFile = []
    testLabelFile = []

    for dataFolderName in dataFolders :
        # 폴더별로 라벨링을 위해 라벨넘버 먼저 labels 참고해서 정해두기
        labelNum = -1
        for i in range(len(set.labels)) :
            if (set.labels[i] == dataFolderName) :
                labelNum = i
                print(f"{dataFolderName}은(는) {labelNum}의 라벨 번호를 가집니다")
                break
        if (labelNum == -1) :
            print(f"{dataFolderName}은(는) 적절한(라벨에서 설정된) 폴더가 아니어서 넘어갔습니다")
            continue
        
        # 데이터가 들어있는 폴더별 경로 지정
        dataFolderPath = os.path.join(dataPath, dataFolderName)

        # 폴더 내부의 파일들을 리스팅
        dataFiles = os.listdir(dataFolderPath)
        totalFileCount = len(dataFiles)

        # 각 파일별 라벨 제작 및 복사를 위한 for문
        for num, dataFileName in enumerate(dataFiles) :
            dataFilePath = os.path.join(dataFolderPath, dataFileName)

            # 진행도 출력부
            if num % 50 == 0 :
                print(f"{dataFolderName} 폴더 test, train 데이터 분리 중 [{num} / {totalFileCount}]")

            # 테스트 데이터와 트레이닝 데이터의 비율을 1:9로 맞추기 위해 10개중 하나씩만 테스트데이터 폴더로 넣기
            if num % 10 == 0 :
                shutil.copy(dataFilePath, testingSavePath)
                testLabelFile.append([dataFileName, labelNum])
            else :
                # 각 파일들은 저장경로에 그냥 복사
                shutil.copy(dataFilePath, trainingSavePath)

                # 파일별 이름, 그리고 labelNum은 labelFile에 저장해두기
                trainingLabelFile.append([dataFileName, labelNum])

    # pickle을 이용해 만들어진 LabelFile을 저장
    trainingLabelPath = os.path.join(trainingSavePath, "labels.pickle")
    testingLabelPath = os.path.join(testingSavePath, "labels.pickle")

    # 트레이닝용 라벨 저장
    with open(trainingLabelPath, "wb") as file :
        pickle.dump(trainingLabelFile, file)

    # 테스팅용 라벨 저장
    with open(testingLabelPath, "wb") as file :
        pickle.dump(testLabelFile, file)