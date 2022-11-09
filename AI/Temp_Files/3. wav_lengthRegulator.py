
import wav_functions as func
import os

def main(target) :
    basePath = func.dataPath
    basePath = os.path.join(basePath, target)
    sourcePath = os.path.join(basePath, "2. RemovedSound")
    savePath = os.path.join(basePath, "3. RegulatedSound")

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)
        
    # 먼저 불러올 데이터들이 들어있는 폴더를 읽는다
    sourceFolders = os.listdir(sourcePath)
    
    # 모든 폴더에 대해 같은 동작을 수행한다
    for folderName in sourceFolders :

        sourceFolderPath = os.path.join(sourcePath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        #먼저 폴더 내의 데이터를 처리하기 위해 각각의 데이터를 읽음
        try :
            sourceFiles = os.listdir(sourceFolderPath)
            totalFileCount = len(sourceFiles)
            print(sourceFolderPath, "폴더 길이 균일화 시작")
        except :
            continue

        # 이후 길이 균일화 진행 전 해당 폴더 내부의 데이터를 처리하고 저장하기 위한 폴더를 만듦
        if not os.path.isdir(saveFolderPath) :
            os.mkdir(saveFolderPath)
            print(saveFolderPath, "폴더 생성")

        # 파일 하나씩 처리
        for count, fileName in enumerate(sourceFiles) :
            # 먼저 처리를 위해 데이터를 읽어오기
            sourceFilePath = os.path.join(sourceFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)

            if(count % 50 == 0) :
                print(f"{folderName} 폴더 길이 균일화 진행 중... [{count} / {totalFileCount}]")

            try :
                wav_loaded = func.loadWAV(sourceFilePath)
            except :
                continue
            
            # 함수를 수행하고 파일을 1초씩 자른 wav_result 받기
            wav_result = func.regulateFile(wav_loaded)
            
            saveFilePath = saveFilePath[ : len(saveFilePath) - 4]
            for num in range(len(wav_result)) :
                func.saveFile(saveFilePath + "_" + str(num) + ".wav", wav_result[num])

        print(sourceFolderPath, "진행 완료")