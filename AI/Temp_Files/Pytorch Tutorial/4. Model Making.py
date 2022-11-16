import os
import torch
import torch.cuda
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 학습을 위한 장치 얻기
# 가능하다면 학습은 GPU등의 하드웨어 가속기에서 수행하자. torch.cuda가 True인지 확인하고, 그렇지 않으면 CPU를 사용하는 코드
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# 클래스 정의하기
# 신경망 모델을 nn.Module의 하위클래스로 정의하고, __init__에서 신경망 계층의 초기화를 수행한다.
# nn.Module을 상속받은 모든 클래스는 forward 메소드에 입력 데이터에 대한 연산을 구현해야 함.
class NeuralNetwork(nn.Module) :
    def __init__(self) :
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x) :
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# 이후 모델의 사용을 위해 입력 데이터를 전달한다.
# 필요한 백그라운드 연산과 함께 model의 forward 함수를 실행하니 직접 호출할 필요 없음
# 모델에 입력을 호출하면, 각 분류(class)에 대한 원시(raw)예측값이 존재하는 10차원 텐서가 반환된다.
# 반환된 원시 예측값을 nn.Softmax 모듈의 instance에 통과시켜 예측 확률을 얻는다.(확률로 반환하기 위함)
X = torch.rand(1, 28, 28, device = device)
logits = model(X)
print("모델에 들어갔다 나온 값 : {}".format(logits))
pred_probab = nn.Softmax(dim = 1)(logits)
print("Softmax함수를 적용한 결과 : {}".format(pred_probab))
y_pred = pred_probab.argmax(1)
print("Predicted class : {}".format(y_pred))

# 모델 계층
# FashionMNIST 모델의 계층들을 살펴보자. 이를 설명하기 위해 28 * 28 크기의 이미지 3개로 구성된 미니배치를 이용하여 신경망을 통과시켜볼 예정
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten
# 계층을 초기화하고, 28x28의 2D 이미지를 784 픽셀 값을 가지는 연속된 배열로 변환한다.
# Flatten에서 dim = 0에 존재하는 미니배치 차원은 유지된다.
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear
# 선형 계층으로, 저장된 가중치(weight)와 편향(bias)을 사용하여 입력에 선형 변환을 적용하는 모듈이다.
layer1 = nn.Linear(in_features=28*28, out_features = 20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU
# 비선형 활성화함수를 이용하여 입, 출력 값 사이를 mapping하여 비선형성을 도입하고, 다양한 현상의 학습이 가능하도록 돕는다
# 여기서는 nn.ReLU를 사용하지만, 모델 생성 시에는 비선형성을 가진 다른 활성화함수를 사용하는 것도 가능하다.
print("Before ReLU : {} \n\n".format(hidden1))
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU : {hidden1} \n\n")

# nn.Sequential
# 순서를 가지는 모듈의 컨테이너이며, 데이터는 정의된 것과 같은 순서로 모든 모듈을 통해 전달된다.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax
# 위에서 나온 최종 결과값은 raw값인 logits를 반환한다. 이를 합이 1인 확률값처럼 사용하기 위해 softMax를 사용한다.
# dim = 1이라는 것은 값의 총 합을 1로 만든다는 것
softmax = nn.Softmax(dim = 1)
pred_probab = softmax(logits)

# 모델 매개변수
# 신경망 내부의 많은 계층들은 모두 매개변수화(Parameterize)된다. 즉, 학습 중에 최적화되는 가중치와 편향에 영향을 받는다는 것.
# nn.Module을 상속하면 모델 객체 내부의 모든 필드가 자동으로 추적(Tracking)되며 parameters()및 named_parameters()메소드로 모든 매개변수에 접근이 가능해진다.
# 모든 매개변수를 순회하고(iterate), 매개변수의 크기와 값을 출력한다.
print(f"Model structrue : {model} \n\n")
for name, param in model.named_parameters() :
    print(f"Layer : {name} \n Size : {param.size()} \n Values : {param[:2]} \n\n")