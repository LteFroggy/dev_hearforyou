# %%
import wav_functions as func
import pickle
import os

def main(target) :
    basePath = func.dataPath
    basePath = os.path.join(basePath, target)
    sourcePath = os.path.join(basePath, "3. RegulatedSound")
    savePath = os.path.join(basePath, "4. MFCCs")

    # 필요하다면 먼저 MFCC폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)    

    # MFCC의 갯수별로 폴더를 따로 생성하기 위해 폴더 추가 생성
    savePath = os.path.join(savePath, str(func.N_MFCCS))

    # 먼저 폴더의 내용을 받아옴
    sourceFolders = os.listdir(sourcePath)

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)

    # N_MFCCS 값에 따른 변화를 알아내기 위해 N_MFCC 값별로 폴더를 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)
        print(savePath, "폴더 생성")

    for folderName in sourceFolders :
        sourceFolderPath = os.path.join(sourcePath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        # 폴더 내의 파일들을 불러옴
        try :
            sourceFiles = os.listdir(sourceFolderPath)
            totalFileCount = len(sourceFiles)
            print(sourceFolderPath, "폴더 MFCC 구하기 시작")
        except :
            print(sourceFolderPath, "내부의 파일을 읽어올 수 없었습니다")
            continue
        
        # 필요하다면 저장할 폴더도 생성함
        if not os.path.isdir(saveFolderPath) :
            os.mkdir(saveFolderPath)
            print(saveFolderPath, "생성됨")

        # 각각의 파일들에 대하여 같은 동작 수행
        for count, fileName in enumerate(sourceFiles) :
            sourceFilePath = os.path.join(sourceFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)[ : -4] + ".pickle"
            # wav 파일을 읽어오는 부분
            try :
                wav_loaded = func.loadWAV(sourceFilePath)
            except :
                continue
            if count % 50 == 0 :
                print(f"{folderName} 폴더 MFCC 계산 진행 중... [{count} / {totalFileCount}]")
                
            # MFCC를 함수를 이용해서 구하기 
            mfcc = func.getMFCC(wav_loaded)
            
            # pickle을 이용하여 저장
            with open(saveFilePath, "wb") as f :
                pickle.dump(mfcc, f)