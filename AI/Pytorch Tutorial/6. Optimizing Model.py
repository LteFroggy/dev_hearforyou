'''
    모델과 데이터를 모두 준비하였으니, 데이터로 매개변수를 최적화하여 모델의 학습, 검증, 테스트를 수행해보자.
    모델 학습은 반복적으로 수행되며, 각각의 반복을 Epoch라 한다.
    각 에폭에서 모델은 출력을 추측하고, 추측과 정답 사이의 오류를 계산하고, 매개변수에 대한 도함수를 수집한 뒤 경사하강법을 이용하여 최적화(optimize)한다.
'''
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 기본 코드
# Dataset과 DataLoader를 다시 이용한다
training_data = datasets.FashionMNIST(
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"),
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"),
    train = False,
    download = True,
    transform = ToTensor()
)

# Dataset을 가진 DataLoader 생성
train_dataloader = DataLoader(training_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

# 모델의 클래스 생성. init에서 필요한 기능을 정의하고, forward에서는 그때그때 수행할 일만 정의한다. forward는 알아서 호출됨.
class NeuralNetwork(nn.Module) :
    def __init__(self) :
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x) :
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 모델 생성
model = NeuralNetwork()

# Hyperparameter
# 하이퍼파라미터는 모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수이다.
# 서로 다른 하이퍼파라미터 값은 모델 학습과 수렴율(Conergence Rate)에 영향을 미칠 수 있다.
# Epoch, Batch Size, Learning Rate등이 여기에 포함된다.
learning_rate = 1e-3
batch_size = 64
epoch = 5

# 최적화 단계
# 하이퍼파라미터 설정 이후에는 최적화 단계를 통해 모델을 학습시킨다. 각 반복은 에폭이라고 함.
# 에폭은 학습 단계, 검증/테스트 단계로 구성되며 학습 단게에서는 학습 데이터셋을 순회(iterate)하고 매개변수 값을 수정하며, 테스트 단계에서는 테스트 데이터셋을 순회(iterate)한다. 

# Loss Function
# 학습용 데이터만 제공한다면 학습되지 않은 신경망은 정답을 제공하지 않는다. 따라서 손실 함수를 이용하여 Loss를 최소화시켜야 한다.
# 일반적인 손실함수에는 회귀 문제(Regression Task)에 사용하는 nn.NLLLose와 nn.CrossEntropyLoss등이 존재한다.
# 모델의 출력 로짓을 nn.CrossEntropyLoss에 전달하여 값을 정규화하고 예측 오류를 계산해보자
loss_fn = nn.CrossEntropyLoss()

# Optimizer
# 최적화는 각 학습 단게에서 모델의 오류를 줄이기 위해 모델의 매개변수를 조정하는 방법이다.
# 최적화 알고리즘은 이 과정이 수행되는 방식(여기서는 SGD(;Stochastic Gradient Decent)를 사용)을 정의한다.
# 모든 최적화 절차는 optimizer 객체에 캡슐화된다.
# 학습하려는 모델의 매개변수, 학습률 등의 Hyperparameter를 등록하여 초기화하자
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 학습 단게에서 최적화는 세 단계로 이루어진다.
# optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 재설정한다. 변화도는 더해지기 때문에 중복 계산을 막기 위해 반복시마다 0으로 설정한다.
# loss.backwards()를 호출하여 예측 손실(prediction loss)를 Back-Propagation한다. Pytorch에서는 각 매개변수에 대한 손실의 변화도를 저장한다
# 변화도 계산 이후에는 optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정한다.
# 다시 말의 위의 3개의 함수를 순서대로 사용하기만 하면 값이 개선된다!

# 구현
# 학습에는 한번에 batch_size갯수 만큼의 값이 들어가고, 교정도 한번에 수행된다. 
# 즉 batch_size가 작으먼 작을수록 적은 값들만 가지고 학습하고, 크면 한번에 많은 값들을 넣고 학습한다.
def train_loop(dataloader, model, loss_fn, optimizer) :
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader) :
        # 예측(Prediction)과 손실(Loss) 계산
        print(f"Shape of X : {X.size()}")
        print(f"Shape of y : {y.size()}")
        print(f"Value of y : {y}")
        pred = model(X)
        print(f"Size of pred : {pred.size()}")
        print(f"Value of pred : {pred}")
        loss = loss_fn(pred, y)
        # print(f"Loss : {loss}")

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 100개의 batch마다 loss값과 현재 진행된 갯수를 반환
        if batch % 100 == 0 :
            loss, current = loss.item(), batch * len(X)
            print(f"Loss : {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn) :
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Test단계에서는 값 추적할 필요 없으니 생략, 
    with torch.no_grad() :
        for X, y in dataloader :
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error : \n Accuracy : {(100 * correct) :>0.1f}%, \n Avg Loss : {test_loss:>8f} \n")

# 구현된 함수를 이용하여 손실 함수와 옵티마이저를 초기화하고 train_loop와 test_loop에 전달한다.
# 모델의 성능 향상을 눈으로 보기 위해 epoch의 수를 증가시켜 볼 수도 있다.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

epochs = 10
for t in range(epochs) :
    print(f"Epoch {t+1} \n ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")