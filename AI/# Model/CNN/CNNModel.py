import os
import wandb
import pickle
import torch
import torch.nn
import settings as set
import PIL.Image as ImgLoader
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageDataSet(Dataset) :
    def __init__(self, data_dir, transform = ToTensor(), target_transform = torch.tensor) :
        self.data_dir = data_dir
        # 피클 상태인 라벨 읽어오기
        with open(os.path.join(data_dir, "labels.pkl"), "rb") as file :
            self.labels = pickle.load(file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) :
        return len(self.labels)
    
    def __getitem__(self, index) :
        image_path = os.path.join(self.data_dir, self.labels[index][0])

        x_image = ImgLoader.open(image_path)
        y_image = self.labels[index][1]

        x_image = self.transform(x_image)
        y_image = self.target_transform(y_image)

        return x_image, y_image

class CNN(torch.nn.Module) :
    def __init__(self) :
        super(CNN, self).__init__()
        self.Conv2d_1 = torch.nn.Conv2d(4, 8, 3)
        self.Conv2d_2 = torch.nn.Conv2d(8, 12, 3)
        self.MaxPool = torch.nn.MaxPool2d(3, 3)
        self.flatten = torch.nn.Flatten()
        self.Linear_1 = torch.nn.Linear(25920, 512)
        self.Linear_2 = torch.nn.Linear(512, 4)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, input) :
        print(f"Input Shape :\t{input.shape}")
        input = F.relu(self.Conv2d_1(input))
        print(f"After Conv2d_1 :\t{input.shape}")
        input = F.relu(self.MaxPool(input))
        print(f"After MaxPool :\t{input.shape}")
        input = F.relu(self.Conv2d_2(input))
        print(f"After Conv2d_2 :\t{input.shape}")
        input = F.relu(self.MaxPool(input))
        print(f"After MaxPool :\t{input.shape}")
        input = self.flatten(self.dropout(input))
        print(f"After Flattening :\t{input.shape}")
        input = F.relu(self.Linear_1(self.dropout(input)))
        print(f"After Linear_1 :\t{input.shape}")
        input = self.Linear_2(self.dropout(input))
        print(f"After Linear_2 :\t{input.shape}")
        return input

def train_loop(dataloader, model, loss_fn, optimizer, epoch) :
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader) :
        pred = model(X)
        # print(f"y의 길이 : {len(y)}")
        # print(f"pred의 길이 : {len(y)}")
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 :
            loss, current = loss.item(), batch * len(X)
            print(f"Loss : {loss:>5f} {current:>5d} / {size:>5d}")
    wandb.log({
        "train_loss" : loss / len(X)
        }, step = epoch)

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


def learnModel(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    basePath = os.path.join(basePath, "2-3. ModelData")
    trainPath = os.path.join(basePath, "trainData")
    testPath = os.path.join(basePath, "testData")

    train_dataset = ImageDataSet(trainPath)
    test_dataset = ImageDataSet(testPath)

    train_dataloader = DataLoader(train_dataset, batch_size = set.BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = set.BATCH_SIZE, shuffle = True)

    model = CNN()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = set.LEARNING_RATE)

    for i in range(set.EPOCHS) :
        print(f"Epoch {i+1} \n ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
        train_loop(train_dataloader, model, loss_fn, optimizer, i)
        test_loop(test_dataloader, model, loss_fn)