import pandas as pd
import os
import shutil
import wav_functions as func


def main() :
    basePath = func.dataPath
    csvPath = os.path.join(basePath, "UrbanSound8K.csv")
    soundPath = os.path.join(basePath, "soundData")
    savePath = os.path.join(basePath, "1. filteredSound")
    soundFolders = os.listdir(soundPath)
    csvContext = func.getLabelData(csvPath)

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)
    
    soundList = {
        '1' : "차량 경적 소리",
        '3' : "개 짖는 소리",
        '8' : "사이렌",
        '6' : "총소리"
    }

    # 읽어온 csvContext를 읽기 편한 형태로 변환
    df = pd.DataFrame(csvContext)
    df.columns = csvContext[1]
    df = df.drop([0, 1], axis = 0)
    df = df.reset_index(drop = True)
    # %%
    
    # soundData내의 폴더 하나하나에 대한 작업
    for folderName in soundFolders :
        # 먼저 폴더 경로를 지정한다
        soundPath_folder = os.path.join(soundPath, folderName)        

        # 폴더 내의 파일들을 리스팅하는데, 폴더가 아니라면 에러가 날 수 있으므로 try-except 활용
        try : 
            files = os.listdir(soundPath_folder)
            print(soundPath_folder, "수행 중")
        except : 
            print(soundPath_folder + "는 폴더가 아닙니다")
            continue


        # 리스팅된 폴더 내의 파일 하나하나에 대한 작업
        for fileName in files :
            # typeNum은 읽어진 파일의 이름에 대한 csv파일 내의 classID
            typeNum = df.loc[df['slice_file_name'] == fileName]['classID']
            
            # 먼저 읽어진 fileName으로 soundPath_file 지정
            soundPath_file = os.path.join(soundPath_folder, fileName)

            # classID를 잘 읽었다면, 해당 classID가 우리가 필요한 정보인지(soundList내에 존재하는지)확인한다
            # 존재하지 않으면 savePath_folder 지정하는 과정에서 에러가 날 수 있으므로 try-except 활용
            try :
                savePath_folder = os.path.join(savePath, soundList[typeNum.iloc[0]])
                savePath_file = os.path.join(savePath_folder, fileName)
                # savePath_folder가 없다면 새로 생성하고 복사함
                if (not os.path.isdir(savePath_folder)) :
                    os.mkdir(savePath_folder)
                    shutil.copy(soundPath_file, savePath_file)

                # 벌써 있는 폴더라면 그냥 복사함
                else :
                    shutil.copy(soundPath_file, savePath_file)

            except Exception as e :
                continue
        print(folderName, "수행 완료")