import librosa
import torch
import numpy as np
import soundfile
import torch.nn.functional as F
import settings as set

# 모델을 load 해서 사용하려 한다면 꼭 모델을 import해줘야 함!
from dnn_model import NeuralNetwork

# 파일 경로와 모델 경로를 적어주면 모든 과정을 수행해서 결과를 반환해주는 함수
def all_in_one(filePath, modelPath) :
    loaded_file = loadWAV(filePath)
    print("음원 불러오기 완료")
    earned_mfcc = torch.Tensor(getMFCC(loaded_file))
    print("mfcc 구하기 완료")
    model = loadModel(modelPath)
    print("모델 불러오기 완료")
    return getPrediction(model, earned_mfcc)

# filePath를 입력받아 wav(mp3, m4a 등의 다른 형식도 가능)파일을 리턴해주는 함수
def loadWAV(filePath) :
    return librosa.load(filePath, sr = set.SAMPLE_RATE)[0]

# 저장할 경로와 wav파일을 입력받아 저장해주는 함수. 확장자는 wav여야 함. ex) savePath = /assets/sounds/sample.wav
def saveFile(savePath, wavFile) :
    soundfile.write(file = savePath, data = wavFile, samplerate = set.SAMPLE_RATE, format = 'wav')

# wav파일을 입력받아 MFCC를 반환해주는 함수
def getMFCC(wav_file) :
    mfcc = np.mean(librosa.feature.mfcc(y=wav_file, sr=set.SAMPLE_RATE, n_mfcc=set.N_MFCCS).T, axis = 0)
    return torch.Tensor(mfcc)

# 모델의 경로를 입력받아 모델을 반환해주는 함수
def loadModel(modelPath) :
    model = NeuralNetwork(len(set.label))
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    return model

# 모델과 input을 입력받아 판단 결과를 반환해주는 함수
def getPrediction(model, values) :
    pred = model(values)
    softmax = F.softmax(pred, dim = 0)
    print(f"Softmax의 최댓값은 {softmax.max()}")
    print(f"판단 결과는 {set.label[softmax.argmax().item()]}")
    if softmax.argmax() >= 0.7 :
        if softmax.argmax().item() >= 5 and softmax.argmax().item() <= 7 :
            return "unknown"
        return set.label[softmax.argmax().item()]
    else :
        return "unknown"

