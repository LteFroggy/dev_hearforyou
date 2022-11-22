import librosa
import torch
import numpy as np
import settings as set

# WAV 파일을 로드해주는 함수
def loadWAV(filePath) :
    return librosa.load(filePath, sr = set.SAMPLE_RATE)[0]

# setting.py 파일에서 설정한 대로 MFCC를 구해서 반환해주는 함수.
def getMFCC(wav_file) :
    return np.mean(librosa.feature.mfcc(y=wav_file, sr=set.SAMPLE_RATE, n_mfcc=set.N_MFCCS).T, axis = 0)

# 파일을 받아 끝의 1초만 잘라 반환해주는 함수
# 파일이 0.5초 미만이라면 -1을 리턴하여 에러로 활용한다
# 0.5초 이상, 1초 미만이라면 필요한 만큼 잘라서 리턴
def trimData(file) :
    sr = set.SAMPLE_RATE
    
    # 파일이 0.5초 미만인 경우, -1 반환
    if (len(file) < 0.5 * sr) :
        return -1

    # 파일이 1초 이상이라면, 잘라주기
    elif (len(file) >= sr) :
        return file[-(sr * 1):]
    
    # 파일이 0.5초 이상 1초 미만이라면, 부족한 만큼 잘라붙이기
    else :
        arr = np.ndarray.tolist(file)
        print(f"length of arr : {len(arr)}")
        arr.extend(arr[:sr - len(arr)])
        print(f"Cutted arr Len : {len(arr)}")
        return np.array(arr)

# 음성파일의 경로를 받아 뒤의 1초만 자르고 MFCC 분석 이후 되돌려주는 함수
def wavToMFCC(filePath) :

    # 음원파일 불러오기
    loadedData = loadWAV(filePath)
    # 음원파일 확인 출력
    # print(f"Shape of loadedData : {loadedData.shape}")
    # print(f"Type of loadedData : {type(loadedData)}")

    # 음원파일의 길이를 1초로 자르기
    trimedData = trimData(loadedData)

    # trimedData의 결과가 -1인 경우에는 커팅이 잘 안 된 것. 에러를 발생시킨다
    try : 
        if (trimedData == -1) : 
            raise Exception("원본 음원 파일의 길이가 0.5초 미만입니다")
    except : None
    # 커팅 확인 출력
    # print(f"Shape of trimedData : {trimedData.shape}")
    # print(f"Type of trimedData : {type(trimedData)}")

    # 잘린 음원파일의 MFCC를 구한다.
    MFCCs = getMFCC(trimedData)
    # MFCC 확인 출력
    # print(f"Shape of MFCCs : {MFCCs.shape}")
    # print(f"Type of MFCCs : {type(MFCCs)}")
    
    # 구해진 MFCC의 텐서화
    x_data = torch.Tensor(MFCCs)
    # 텐서 확인 출력
    # print(f"Shape of x_data : {x_data.shape}")
    # print(f"Type of x_data : {type(x_data)}")

    return x_data