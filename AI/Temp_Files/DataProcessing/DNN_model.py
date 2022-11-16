import os
import torch
import pickle
import wav_functions as func
import settings as set
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 사용자 지정 데이터셋
class SoundDataset(Dataset) :
    # init함수는 Dataset 객체가 생성될 때 한번만 실행된다. 여기서는 이미지, 주석파일이 포함된 디렉토리와 두가지 변형의 초기화를 진행
    def __init__(self, data_dir, transform = torch.tensor, target_transform = torch.tensor) :
        self.data_dir = data_dir
        # 피클 상태인 라벨 읽어오기
        with open(os.path.join(data_dir, "labels.pickle"), "rb") as file :
            self.labels = pickle.load(file)
        self.transform = transform
        self.target_transform = target_transform

    # len함수는 데이터셋의 샘플 갯수를 반환한다``
    def __len__(self) :
        return len(self.labels)

    # getitem함수는 주어진 인덱스에 해당하는 샘플을 데이터셋에서 불러오고 반환해준다. 인덱스 기반으로 디스크에서 데이터의 위치 식별 및 데이터->텐서 변환, 라벨 가져오기 등을 수행
    def __getitem__(self, idx) :
        # idx 번쨰 값의 데이터를 불러오기 위해 먼저 path 지정
        data_path = os.path.join(self.data_dir, self.labels[idx][0])

        # 데이터에 해당하는 피클 data에 가져오기
        with open(data_path, "rb") as data:
            data = pickle.load(data)

        # print(f"가져온 데이터명 : {self.labels[idx][0]}")
        # print(f"가져온 데이터의 Type : {type(data)}")
        # print(f"가져온 데이터의 모양 : {data.shape}")
        # print(f"가져온 데이터의 값 : {data}")

        # 라벨은 labels에서 가져오기
        label = self.labels[idx][1]

        # Transform 수행
        data = self.transform(data)
        label = self.target_transform(label)

        # 변환된 값 반환
        return data, label

# 사용할 모델
class NeuralNetwork(nn.Module) :
    def __init__(self) :
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 4)
        )
    
    def forward(self, x) :
        logits = self.linear_relu_stack(x)
        return logits
        
def train_loop(dataloader, model, loss_fn, optimizer) :
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader) :
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 :
            loss, current = loss.item(), batch * len(X)
            print(f"Loss : {loss:>5f} {current:>5d} / {size:>5d}")

def test_loop(dataloader, model, loss_fn) :
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad() :
        model.eval()
        for X, y in dataloader :
            pred = model(X)
            
            # Softmax의 최댓값 구하기. 0.6 이하면 못알아들었다고 해도 될듯??
            # for singleResult in pred :
            #     highest_softmax = F.softmax(singleResult, dim = 0)
            #     print(f"Highest softmax : {highest_softmax.max():>3f}")

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 경로부터 지정
basePath = func.dataPath
basePath = os.path.join(basePath, "UrbanSound")
basePath = os.path.join(basePath, "5. modelData")
basePath = os.path.join(basePath, str(set.N_MFCCS))
trainPath = os.path.join(basePath, "trainData")
testPath = os.path.join(basePath, "testData")

train_dataset = SoundDataset(trainPath)
test_dataset = SoundDataset(testPath)

train_dataLoader = DataLoader(train_dataset, batch_size = set.BATCH_SIZE, shuffle = True)
test_dataLoader = DataLoader(test_dataset, batch_size = set.BATCH_SIZE, shuffle = True)

train_datas, train_labels = next(iter(train_dataLoader))

print(f"Train data's Shape : {train_datas.size()}")
print(f"Train data's Value : {train_datas}")
print(f"Train label's Shape : {train_labels.size()}")
print(f"Train label's Value : {train_labels}")

model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = set.LEARNING_RATE)

for t in range(set.EPOCHS) :
    print(f"Epoch {t+1} \n ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    train_loop(train_dataLoader, model, loss_fn, optimizer)
    test_loop(test_dataLoader, model, loss_fn)

print("Finished!")

# 완성된 모델 저장하기
savePath = os.path.dirname(os.path.realpath(__file__))
torch.save(model, os.path.join(savePath, "model_128.pt"))
torch.save(model.state_dict(), os.path.join(savePath, "state_dict_128.pt"))