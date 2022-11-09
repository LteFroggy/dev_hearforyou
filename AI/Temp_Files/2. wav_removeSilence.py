'''
    음성 파일의 원할한 분석을 위해 2초 단위로 wav파일을 자름
    파일이 해당 크기보다 작다면 수행하지 않음. 그냥 삭제해야 할 듯?

    librosa.load함수는 음성 파일을 sample rate * sec 갯수의 ndarray로 반환한다
    따라서 ndarray.shape / (sample rate * 2)가 총 나올 파일의 갯수가 됨
'''
import os
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
    for folderName in soundFolders :
        soundPath_folder = os.path.join(soundPath, folderName)
        try :   
            soundFiles = os.listdir(soundPath_folder)
            totalFileCount = len(soundFiles)
            print(soundPath_folder + " 폴더 분류 시작")
        except :
            print(soundPath_folder + " 는 폴더가 아닙니다")
            continue

        # 소음이 제거된 파일들을 모아둘 폴더 경로 지정
        savePath_folder = os.path.join(savePath, folderName)
        if (not os.path.isdir(savePath_folder)) :
            os.mkdir(savePath_folder)
            print(savePath_folder + " 생성")

        for count, fileName in enumerate(soundFiles) :
            # 파일을 librosa를 이용하여 읽어오기
            soundPath_file = os.path.join(soundPath_folder, fileName)
            wav_loaded = func.loadWAV(soundPath_file)
            # func.drawPlot(wav_loaded, "Raw Data")

            if(count % 30 == 0) :
                print(f"{folderName} 폴더 무음부분 제거 진행 중... [{count} / {totalFileCount}]")

            # 파일에서 아무 소리 없는 부분을 제거한다
            wav_removed = func.removeSilence(wav_loaded)
            
            # 0.5초가 넘어가는 파일만 저장
            if len(wav_removed) >= 0.5 * set.SAMPLE_RATE :
                func.saveFile(os.path.join(savePath_folder, fileName), wav_removed)
    print("무음부분 제거 완료")