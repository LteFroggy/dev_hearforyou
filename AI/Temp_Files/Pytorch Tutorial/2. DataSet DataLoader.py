'''
    데이터 샘플 처리용 코드는 지저분하고 유지보수가 어렵다. 
    그렇기 때문에 더 나은 가독성(Readability)과 모듈성(Modularity)을 위해 데이터셋 코드를 모델 학습 코드와 분리해야 한다
    Pytorch는 DataLoader와 Dataset을 이용하여 이를 실현한다
    Dataset은 샘플(data)과 정답(label)을 저장하고, DataLoader는 Dataset이 샘플에 쉽게 접근 가능하도록 순회 가능한 객체(iterable)로 감싸준다.

    TorchVision에서 Fashion-MNIST 데이터셋을 불러와서 사용할 예정
    Fashion_MNIST는 기사 이미지 데이터셋으로, 60,000개의 학습 예제와 10,000개의 테스트 예제로 이루어져 있다.
    각 예제는 흑백의 28x28 이미지와 10개의 분류 중 하나인 정답으로 구성됨

    root는 학습/테스트 데이터가 저장되는 경로
    train은 학습용 또는 테스트용 데이터셋의 여부
    download=True는 root에 데이터가 없는 경우 다운로드한다는 의미
    transform과 target_transform은 특징(feature)과 정답(label) 변형을 지정한다.
'''
import os
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

# 데이터 경로 지정
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, "data")

# 트레이닝 데이터 불러오기
training_data = datasets.FashionMNIST(
    root = path,
    train = True,
    download = True,
    transform = ToTensor()
)

# 테스트 데이터 불러오기
test_data = datasets.FashionMNIST(
    root = path,
    train = False,
    download = True,
    transform = ToTensor()
)

# 데이터셋을 순회하고 시각화하기
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize = (8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1) :
    sample_idx = torch.randint(len(training_data), size = (1, 1)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap = "gray")

plt.show()

# DataLoader로 학습용 데이터 준비하기
# Dataset은 Dataset의 특징(feature)을 가져오고 하나의 샘플에 정답(label)을 지정한다.
# 모델 학습 시에는 일반적으로 샘플을 미니 배치(mini batch)로 전달하고, 매 에폭(epoch)마다 데이터를 다시 섞어 overfit을 막으며 
# python의 multiprocessing을 사용하여 데이터의 검색 속도를 높인다.
# DataLoader는 간단한 API로 위의 과정들을 추상화한 순회 가능한 객체(iterable)이다.
train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

# DataLoader를 통해 순회하기
# 데이터셋을 불러왔다면, 필요에 따라 순회(iteration)할 수 있다. 아래의 각 iteration은 각각 위에서 지정한 batch size에 맞는 train_data와 test_data의 묶음(batch)를 반환한다.
# Shuffle = True로 지정했으므로, 모든 배치를 순회했다면 데이터가 섞인다.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape : {train_features.size()}")
print(train_features[0])
print(f"Labels batch shape : {train_labels.size()}")


# 하나 뽑힌 데이터 보기
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap = "gray")
plt.show()
print(f"Label : {label} \n")