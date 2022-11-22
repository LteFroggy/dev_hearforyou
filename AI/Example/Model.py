'''
    1. 음원을 받아 뒤의 1초만 자르기
    2. 잘라진 음원의 MFCC 구하기
    3. 최적화된 모델에 넣고 값 돌려주기
'''
import os
import torch
import settings as set
import functions as func
import torch.nn.functional as F
from NeuralNetwork import NeuralNetwork

# 모델을 로드하여 분석을 수행하고 결과를 리턴해주는 함수
def Analyze(fileName, ereaseFlag = False) :
    path = os.path.dirname(os.path.realpath(__file__))
    modelPath = os.path.join(path, "ModelFile")
    filePath = os.path.join(path, "SoundFile")
    filePath = os.path.join(filePath, fileName)
    model = NeuralNetwork()
    model = torch.load(os.path.join(modelPath, "model.pt"))
    model.load_state_dict(torch.load(os.path.join(modelPath, "state_dict.pt")))
    model.eval()
    
    x_data = func.wavToMFCC(filePath)

    # 분석한 파일을 지우기
    if (ereaseFlag) :
        os.remove(filePath)

    with torch.no_grad() :
        pred = model(x_data)
        # print(f"Type of pred : {type(pred)}")
        # print(f"Shape of pred : {pred.shape}")
        # print(f"Value of pred : {pred}")

        # Softmax의 최댓값 구하기. 0.6 이하면 못알아들었다고 해도 될듯??
        pred_normalized = F.softmax(pred, dim = 0)
        # print(f"Type of pred_normalized : {type(pred_normalized)}")
        # print(f"Shape of pred_normalized : {pred_normalized.shape}")
        # print(f"Value of pred_normalized : {pred_normalized}")

    if (pred_normalized.max() > 0.9) :
        return pred_normalized.argmax().item()
    
    else :
        return -1











FILENAME = "135528-6-3-0.wav"

if __name__ == "__main__" :
    label = set.labels
    result = Analyze(FILENAME, ereaseFlag = False)
    
    if result != -1 :
        print(f"{FILENAME}은 {label[result]} 입니다.")
    else :
        print(f"{FILENAME}이 무엇인지 확인하지 못했습니다")