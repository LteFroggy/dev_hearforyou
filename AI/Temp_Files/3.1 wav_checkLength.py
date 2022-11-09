import wav_functions as func
import os

def main() :
    basePath = func.dataPath
    sourcePath = os.path.join(basePath, "regulatedSound")

    # 먼저 불러올 데이터들이 들어있는 폴더를 읽는다``
    sourceFolders = os.listdir(sourcePath)
    
    # 모든 폴더에 대해 같은 동작을 수행한다
    for folderName in sourceFolders :

        sourceFolderPath = os.path.join(sourcePath, folderName)

        #먼저 폴더 내의 데이터를 불러옴
        try :
            sourceFiles = os.listdir(sourceFolderPath)
            print(sourceFolderPath, "진행 중")
        except :
            continue

        # 파일 하나씩 처리
        for fileName in sourceFiles :
            sourceFilePath = os.path.join(sourceFolderPath, fileName)

            try :
                wav_loaded = func.loadWAV(sourceFilePath)
            except :
                continue

            if len(wav_loaded) != func.CUT_SEC * func.SAMPLE_RATE :
                raise Exception("길이 이상한 녀석 있음. " + sourceFilePath + ", 길이는 " + str(len(wav_loaded)))
            else :
                print(sourceFilePath, ":", len(wav_loaded))
                
        
        print(sourceFolderPath, "완료")