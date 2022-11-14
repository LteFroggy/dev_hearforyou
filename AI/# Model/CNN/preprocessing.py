import os
import shutil
from tqdm import tqdm
import wav_functions as func
import settings as set

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

    print("길이 균일화 완료")
    print(f"{sourcePath}폴더 삭제 중")
    shutil.rmtree(sourcePath)
    print(f"{sourcePath}폴더 삭제 완료")