import os
import torch
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

print(training_data)