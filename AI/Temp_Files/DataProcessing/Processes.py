import os
import shutil
import pickle
import settings as set
import wav_functions as func

def removeSilence(target) :
    basePath = func.dataPath
    basePath = os.path.join(basePath, target)
    soundPath = os.path.join(basePath, "1. FilteredSound")
    savePath = os.path.join(basePath, "2. RemovedSound")

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)    

    soundFolders = os.listdir(soundPath)
    totalFolderCount = len(soundFolders)
    for folderCount, folderName in enumerate(soundFolders, start = 1) :
        soundPath_folder = os.path.join(soundPath, folderName)
        try :   
            soundFiles = os.listdir(soundPath_folder)
            totalFileCount = len(soundFiles)
            print(folderName + " 폴더 무음부분 제거 시작")
        except :
            print(folderName + " 는 폴더가 아닙니다")
            continue

        # 소음이 제거된 파일들을 모아둘 폴더 경로 지정
        savePath_folder = os.path.join(savePath, folderName)
        if (not os.path.isdir(savePath_folder)) :
            os.mkdir(savePath_folder)
            # print(folderName + " 생성")

        for count, fileName in enumerate(soundFiles) :
            # 파일을 librosa를 이용하여 읽어오기
            soundPath_file = os.path.join(soundPath_folder, fileName)
            try :   
                wav_loaded = func.loadWAV(soundPath_file)
            except :
                continue
            # func.drawPlot(wav_loaded, "Raw Data")

            # 진행도 출력부
            if(count % 50 == 0) :
                print(f"{folderName} 폴더[{folderCount} / {totalFolderCount}] 무음부분 제거 진행 중... [{count} / {totalFileCount}]")

            # 파일에서 아무 소리 없는 부분을 제거한다
            wav_removed = func.removeSilence(wav_loaded)
            
            # 0.5초가 넘어가는 파일만 저장
            if len(wav_removed) >= 0.5 * set.SAMPLE_RATE :
                func.saveFile(os.path.join(savePath_folder, fileName), wav_removed)
    print("무음부분 제거 완료")


def lengthRegulate(target) :
    basePath = func.dataPath
    basePath = os.path.join(basePath, target)
    sourcePath = os.path.join(basePath, "2. RemovedSound")
    savePath = os.path.join(basePath, "3. RegulatedSound")

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)
        
    # 먼저 불러올 데이터들이 들어있는 폴더를 읽는다
    sourceFolders = os.listdir(sourcePath)
    totalFolderCount = len(sourceFolders)
    
    # 모든 폴더에 대해 같은 동작을 수행한다
    for folderCount, folderName in enumerate(sourceFolders, start = 1) :

        sourceFolderPath = os.path.join(sourcePath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        #먼저 폴더 내의 데이터를 처리하기 위해 각각의 데이터를 읽음
        try :
            sourceFiles = os.listdir(sourceFolderPath)
            totalFileCount = len(sourceFiles)
            print(folderName, "폴더 길이 균일화 시작")
        except :
            print(folderName + " 는 폴더가 아닙니다")
            continue

        # 이후 길이 균일화 진행 전 해당 폴더 내부의 데이터를 처리하고 저장하기 위한 폴더를 만듦
        if not os.path.isdir(saveFolderPath) :
            os.mkdir(saveFolderPath)
            # print(folderName, "폴더 생성")

        # 파일 하나씩 처리
        for count, fileName in enumerate(sourceFiles) :
            # 먼저 처리를 위해 데이터를 읽어오기
            sourceFilePath = os.path.join(sourceFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)

            # 진행도 출력부
            if(count % 50 == 0) :
                print(f"{folderName} 폴더[{folderCount} / {totalFolderCount}] 길이 균일화 진행 중... [{count} / {totalFileCount}]")

            try :
                wav_loaded = func.loadWAV(sourceFilePath)
            except :
                continue
            
            # 함수를 수행하고 파일을 1초씩 자른 wav_result 받기
            wav_result = func.regulateFile(wav_loaded)
            
            saveFilePath = saveFilePath[ : len(saveFilePath) - 4]
            for num in range(len(wav_result)) :
                func.saveFile(saveFilePath + "_" + str(num) + ".wav", wav_result[num])

    print("길이 균일화 완료")


def getMFCC(target) :
    basePath = func.dataPath
    basePath = os.path.join(basePath, target)
    sourcePath = os.path.join(basePath, "3. RegulatedSound")
    savePath = os.path.join(basePath, "4. MFCCs")

    # 필요하다면 먼저 MFCC폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)    

    # MFCC의 갯수별로 폴더를 따로 생성하기 위해 폴더 추가 생성
    savePath = os.path.join(savePath, str(set.N_MFCCS))

    # 먼저 폴더의 내용을 받아옴
    sourceFolders = os.listdir(sourcePath)
    totalFolderCount = len(sourceFolders)

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)

    # N_MFCCS 값에 따른 변화를 알아내기 위해 N_MFCC 값별로 폴더를 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)
        print(f"MFCC = {set.N_MFCCS} 로 진행")

    for folderCount, folderName in enumerate(sourceFolders, start = 1) :
        sourceFolderPath = os.path.join(sourcePath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        # 폴더 내의 파일들을 불러옴
        try :
            sourceFiles = os.listdir(sourceFolderPath)
            totalFileCount = len(sourceFiles)
            print(folderName, "폴더 MFCC 구하기 시작")
        except :
            print(folderName, "내부의 파일을 읽어올 수 없었습니다")
            continue
        
        # 필요하다면 저장할 폴더도 생성함
        if not os.path.isdir(saveFolderPath) :
            os.mkdir(saveFolderPath)
            # print(folderName, "생성됨")

        # 각각의 파일들에 대하여 같은 동작 수행
        for count, fileName in enumerate(sourceFiles) :
            sourceFilePath = os.path.join(sourceFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)[ : -4] + ".pkl"
            # wav 파일을 읽어오는 부분
            try :
                wav_loaded = func.loadWAV(sourceFilePath)
            except :
                continue

            # 진행도 출력부
            if count % 50 == 0 :
                print(f"{folderName} 폴더[{folderCount} / {totalFolderCount}] MFCC 계산 진행 중... [{count} / {totalFileCount}]")
                
            # MFCC를 함수를 이용해서 구하기 
            mfcc = func.getMFCC(wav_loaded)
            
            # pickle을 이용하여 저장
            with open(saveFilePath, "wb") as f :
                pickle.dump(mfcc, f)


def labeling(target, labels) :
    # 베이스 경로 지정
    basePath = func.dataPath
    basePath = os.path.join(basePath, target)
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
    totalFolderCount = len(dataFolders)

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

    for folderCount, dataFolderName in enumerate(dataFolders, start = 1) :
        # 폴더별로 라벨링을 위해 라벨넘버 먼저 labels 참고해서 정해두기
        labelNum = -1
        for value in labels.keys() :
            if (labels[value] == dataFolderName) :
                labelNum = value
                print(f"{dataFolderName}은(는) {labelNum}의 라벨 번호를 가집니다")
                break
        if (labelNum == -1) :
            print(f"{dataFolderName}은(는) 적절한 폴더가 아니어서 넘어갔습니다")
            continue
        
        # 데이터가 들어있는 폴더별 경로 지정
        dataFolderPath = os.path.join(dataPath, dataFolderName)

        # 폴더 내부의 파일들을 리스팅
        dataFiles = os.listdir(dataFolderPath)
        totalFileCount = len(dataFiles)

        # 각 파일별 라벨 제작 및 복사를 위한 for문
        for num, dataFileName in enumerate(dataFiles) :
            if (dataFileName == ".DS_Store") :
                continue

            dataFilePath = os.path.join(dataFolderPath, dataFileName)

            # 진행도 출력부
            if num % 50 == 0 :
                print(f"{dataFolderName} 폴더[{folderCount} / {totalFolderCount}] test, train 데이터 분리 중 [{num} / {totalFileCount}]")            

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
    trainingLabelPath = os.path.join(trainingSavePath, "labels.pkl")
    testingLabelPath = os.path.join(testingSavePath, "labels.pkl")

    # 트레이닝용 라벨 저장
    with open(trainingLabelPath, "wb") as file :
        pickle.dump(trainingLabelFile, file)

    # 테스팅용 라벨 저장
    with open(testingLabelPath, "wb") as file :
        pickle.dump(testLabelFile, file)