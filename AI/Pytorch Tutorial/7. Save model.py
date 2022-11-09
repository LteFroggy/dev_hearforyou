import os
import torch
import torchvision.models as models


# 모델 가중치 저장하고 불러오기
# 가중치를 불러오기 위해서는 동일한 모델의 인스턴스를 생성하고 load_state_dict()를 사용한다
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), "model_weight.pth")

model = models.vgg16() # 기존 가중치가 필요 없으므로 Pretrained = True를 지정하지 않는다
model.load_state_dict(torch.load('model_weight.pth'))
# 추론 이전에 model.eval() 을 호출하여 dropout과 배치 정규화(batch normalization) 을 평가모드로 설정해야 한다.
model.eval()

# 모델의 형태를 포함하여 저장하고 불러오기
torch.save(model, 'model.pth')
print("저장 완료!")

model = torch.load('model.pth')