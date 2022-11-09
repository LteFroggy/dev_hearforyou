import os
import wav_functions as func

if __name__== "__main__" :
    basePath = func.dataPath
    basePath = os.path.join(basePath, "UrbanSound")
    soundPath = os.path.join(basePath, "3. regulatedSound")
    savePath = os.path.join(os.path.join(basePath, "3. regulatedSound"), "photos")

    # 소음 제거 전의 소리가 들어있는 폴더들을 지정
    soundFolders = os.listdir(soundPath) # filteredSound

    if not (os.path.isdir(savePath)) :
        os.mkdir(savePath)

    for folderName in soundFolders :
        # 폴더의 경로 지정
        soundPath_folder = os.path.join(soundPath, folderName) # filteredSound/개 짖는 소리

        # 지정된 폴더에서 목록을 가져옴. 폴더가 아닌 경우에는 에러 날 수 있으니 try-except 사용하여 처리
        try :   
            soundFiles = os.listdir(soundPath_folder)
            print(soundPath_folder + " 진행 중")
        except :
            print(soundPath_folder + " 는 폴더가 아닙니다")
            continue

        # 소음 제거 상태를 확인할 사진 저장 경로 지정하고, 없다면 생성하기
        savePath_folder = os.path.join(savePath, folderName) # removedSound/photos/개 짖는 소리
        if (not os.path.isdir(savePath_folder)) :
            os.mkdir(savePath_folder)
            print(savePath_folder + " 생성")

        for fileName in soundFiles :
            # soundFile의 경로를 재지정한다
            soundPath_file = os.path.join(soundPath_folder, fileName) # filteredSound/개 짖는 소리/***.wav
            
            # 소리파일의 경로를 지정하고, 로드한다. 이 때 로드가 불가능한 파일이면 에러가 날 수 있으니 try-except 활용
            try :
                wav_loaded = func.loadWAV(soundPath_file)
                print(soundPath_file + " 로드 완료")
            except :
                print(soundPath_file + " 로드 실패")
                continue
            
            # 로드가 성공했다면, 제거한 후의 사진을 저장할 경로 지정
            savePath_file = os.path.join(savePath_folder, fileName[:len(fileName) - 4] + '.png')
            func.saveAsPhoto(wav_loaded, savePath_file, fileName)
    print(soundPath_folder + " 종료")
