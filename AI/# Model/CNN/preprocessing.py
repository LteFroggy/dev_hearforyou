import os
import torch
import pickle
import shutil
import librosa
import unicodedata
import numpy as np
import settings as set
import wav_functions as func
import PIL.Image as ImgLoader
import librosa.display as display
import torchvision.transforms as transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

deleteFlag = [
    True, # False면 전체 미삭제, True면 아래의 값에 따라 삭제 여부 결정
    False, # RemovedSilence 폴더 삭제 여부
    False, # RegulatedSound 폴더 삭제 여부
    True, # RegulatedPhoto 폴더 삭제 여부
    False, # CuttedPhoto 폴더 삭제 여부
    False, # Resized 폴더 삭제 여부
]

def removeSilence(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    soundPath = os.path.join(basePath, "source")
    savePath = os.path.join(basePath, "1. RemovedSound")

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)    

    soundFolders = os.listdir(soundPath)
    totalFolderCount = len(soundFolders)
    for folderCount, folderName in enumerate(soundFolders, start = 1) :
        soundPath_folder = os.path.join(soundPath, folderName)
        try :   
            soundFiles = os.listdir(soundPath_folder)
        except :
            print(folderName + " 는 폴더가 아닙니다")
            continue

        # 소음이 제거된 파일들을 모아둘 폴더 경로 지정
        savePath_folder = os.path.join(savePath, folderName)
        if (not os.path.isdir(savePath_folder)) :
            os.mkdir(savePath_folder)
            # print(folderName + " 생성")

        for fileName in tqdm(soundFiles, desc = f"{folderName} 폴더[{folderCount} / {totalFolderCount}] 무음부분 제거 진행 중") :
            # 파일을 librosa를 이용하여 읽어오기
            soundPath_file = os.path.join(soundPath_folder, fileName)
            try :   
                wav_loaded = func.loadWAV(soundPath_file)
            except :
                continue
            # func.drawPlot(wav_loaded, "Raw Data")

            # 파일에서 아무 소리 없는 부분을 제거한다
            wav_removed = func.removeSilence(wav_loaded)
            
            # 0.5초가 넘어가는 파일만 저장
            if len(wav_removed) >= 0.5 * set.SAMPLE_RATE :
                func.saveFile(os.path.join(savePath_folder, fileName), wav_removed)

    print(f"무음 제거 완료")



def lengthRegulate(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    sourcePath = os.path.join(basePath, "1. RemovedSound")
    savePath = os.path.join(basePath, "2. RegulatedSound")

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
        except :
            print(folderName + " 는 폴더가 아닙니다")
            continue

        # 이후 길이 균일화 진행 전 해당 폴더 내부의 데이터를 처리하고 저장하기 위한 폴더를 만듦
        if not os.path.isdir(saveFolderPath) :
            os.mkdir(saveFolderPath)
            # print(folderName, "폴더 생성")

        # 파일 하나씩 처리
        for fileName in tqdm(sourceFiles, desc = f"{folderName} 폴더[{folderCount} / {totalFolderCount}] 길이 균일화 진행 중") :
            # 먼저 처리를 위해 데이터를 읽어오기
            sourceFilePath = os.path.join(sourceFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)

            try :
                wav_loaded = func.loadWAV(sourceFilePath)
            except :
                continue
            
            # 함수를 수행하고 파일을 1초씩 자른 wav_result 받기
            wav_result = func.regulateFile(wav_loaded)
            
            saveFilePath = saveFilePath[ : len(saveFilePath) - 4]
            for num in range(len(wav_result)) :
                func.saveFile(saveFilePath + "_" + str(num) + ".wav", wav_result[num])

    print(f"정규화 완료")
    if deleteFlag[0] and deleteFlag[1] :
        func.removeUsedFolder(sourcePath)

def saveAsImage(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    soundPath = os.path.join(basePath, "2. RegulatedSound")
    savePath = os.path.join(basePath, "2-1. RegulatedPhoto")

    if not(os.path.isdir(savePath)) :
        os.mkdir(savePath)

    for folderCount, folderName in enumerate(os.listdir(soundPath), start = 1) : 
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

    print(f"이미지화 완료")
    if deleteFlag[0] and deleteFlag[2] :
        func.removeUsedFolder(soundPath)

cutColumn = 4
cutRow = 5

def cutImage(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    soundPath = os.path.join(basePath, "2-1. RegulatedPhoto")
    savePath = os.path.join(basePath, "2-2. CuttedPhoto")

    if not (os.path.isdir(savePath)) :
        os.mkdir(savePath)

    soundFolderList = os.listdir(soundPath)
    totalFolderCount = len(soundFolderList)

    for folderCount, folderName in enumerate(soundFolderList, start = 1) :
        soundFolderPath = os.path.join(soundPath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        if not (os.path.isdir(saveFolderPath)) :
            os.mkdir(saveFolderPath)

        # listdir 중 폴더가 아니면 에러가 날 수 있음
        try :
            soundFileList = os.listdir(soundFolderPath)
        except :
            continue
        
        for fileName in tqdm(soundFileList, desc = f"{folderName} 폴더 [{folderCount} / {totalFolderCount}] 이미지 커팅 진행 중") :
            soundFilePath = os.path.join(soundFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)

            # 이미지파일 불러오기
            imgFile = ImgLoader.open(soundFilePath)

            # 이미지파일 텐서화하기
            img_tensor = transforms.ToTensor()(imgFile)

            # 원본 크기 출력
            # print(f"원본 Shape : \t{img_tensor.shape}")

            # 앞쪽 열 10개, 뒤쪽 행 열개 자르기
            cuttedImg = torch.Tensor(len(img_tensor), len(img_tensor[0]) - (cutColumn), len(img_tensor[0][0]) - cutRow)
            for i in range(len(img_tensor)) :
                cuttedImg[i] = img_tensor[i][ : -cutColumn, cutRow : ]
            # print(f"값 : {cuttedImg[0]}")
            # print(f"길이 1 : {len(cuttedImg)}")
            # print(f"길이 2 : {len(cuttedImg[0])}")
            # print(f"길이 3 : {len(cuttedImg[0][0])}")
            imgFile = transforms.ToPILImage()(cuttedImg)
            imgFile.save(saveFilePath)

    print(f"이미지 커팅 완료")
    if deleteFlag[0] and deleteFlag[3] :
        func.removeUsedFolder(soundPath)

def labeling(target, labels) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    dataPath = os.path.join(basePath, "2-3. Resized")
    savePath = os.path.join(basePath, "2-4. ModelData")

    if not(os.path.isdir(savePath)) :
        os.mkdir(savePath)

    dataFolderList = os.listdir(dataPath)
    totalFolderCount = len(dataFolderList)
    label_train = []
    label_test = []

    for folderCount, folderName in enumerate(dataFolderList, start = 1) :
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

    print(f"무음 제거 완료")
    if deleteFlag[0] and deleteFlag[5] :
        func.removeUsedFolder(dataPath)

def imgResizing(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    dataPath = os.path.join(basePath, "2-2. CuttedPhoto")
    savePath = os.path.join(basePath, "2-3. Resized")

    if not (os.path.isdir(savePath)) :
        os.mkdir(savePath)

    folders = os.listdir(dataPath)
    totalFolderCount = len(folders)
    
    for folderCount, folderName in enumerate(folders, start = 1) :
        dataFolderPath = os.path.join(dataPath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        if not (os.path.isdir(saveFolderPath)) :
            os.mkdir(saveFolderPath)

        try :  
            dataFiles = os.listdir(dataFolderPath)
        except :
            continue
        
        for fileName in tqdm(dataFiles, desc = f"{folderName}폴더 [{folderCount} / {totalFolderCount}] 리사이징 진행 중") :
            dataFilePath = os.path.join(dataFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)
            
            try :
                img = ImgLoader.open(dataFilePath)
                img_resize = img.resize((set.CUT_SIZE, set.CUT_SIZE))
                img_resize.save(saveFilePath, 'png')
            except Exception as e:
                print(e)
                continue

    print(f"리사이징 완료")
    if deleteFlag[0] and deleteFlag[4] :
        func.removeUsedFolder(dataPath)