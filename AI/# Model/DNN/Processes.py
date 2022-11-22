import os
import torch
import wandb
import shutil
import pickle
import numpy as np
import settings as set
import wav_functions as func
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
deleteFlag = [
    False, # False면 전체 미삭제, True면 아래의 값에 따라 삭제 여부 결정
    False, # RemovedSilence 폴더 삭제 여부
    False, # RegulatedSound 폴더 삭제 여부
    False, # RegulatedPhoto 폴더 삭제 여부
    False, # CuttedPhoto 폴더 삭제 여부
]

def removeSilence(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    soundPath = os.path.join(basePath, "source")
    savePath = os.path.join(basePath, "1. RemovedSound")

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)    

    soundFolders = os.listdir(soundPath)
    totalFolderCount = len(soundFolders)
    for folderCount, folderName in enumerate(soundFolders, start = 1) :
        soundPath_folder = os.path.join(soundPath, folderName)
        try :   
            soundFiles = os.listdir(soundPath_folder)
        except :
            print(folderName + " 는 폴더가 아닙니다")
            continue

        # 소음이 제거된 파일들을 모아둘 폴더 경로 지정
        savePath_folder = os.path.join(savePath, folderName)
        if (not os.path.isdir(savePath_folder)) :
            os.mkdir(savePath_folder)
            # print(folderName + " 생성")

        for fileName in tqdm(soundFiles, desc = f"{folderName} 폴더[{folderCount} / {totalFolderCount}] 무음부분 제거 진행 중") :
            # 파일을 librosa를 이용하여 읽어오기
            soundPath_file = os.path.join(soundPath_folder, fileName)
            try :   
                wav_loaded = func.loadWAV(soundPath_file)\
                    
                # 파일에서 아무 소리 없는 부분을 제거한다
                wav_removed = func.removeSilence(wav_loaded)
            except :
                continue
            # func.drawPlot(wav_loaded, "Raw Data")


            
            # 0.5초가 넘어가는 파일만 저장
            if len(wav_removed) >= 0.5 * set.SAMPLE_RATE :
                func.saveFile(os.path.join(savePath_folder, fileName), wav_removed)

    print(f"무음 제거 완료")



def lengthRegulate(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    sourcePath = os.path.join(basePath, "1. RemovedSound")
    savePath = os.path.join(basePath, "2. RegulatedSound")

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)
        
    # 먼저 불러올 데이터들이 들어있는 폴더를 읽는다
    sourceFolders = os.listdir(sourcePath)
    totalFolderCount = len(sourceFolders)
    
    # 모든 폴더에 대해 같은 동작을 수행한다
    for folderCount, folderName in enumerate(sourceFolders, start = 1) :

        sourceFolderPath = os.path.join(sourcePath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        #먼저 폴더 내의 데이터를 처리하기 위해 각각의 데이터를 읽음
        try :
            sourceFiles = os.listdir(sourceFolderPath)
        except :
            print(folderName + " 는 폴더가 아닙니다")
            continue

        # 이후 길이 균일화 진행 전 해당 폴더 내부의 데이터를 처리하고 저장하기 위한 폴더를 만듦
        if not os.path.isdir(saveFolderPath) :
            os.mkdir(saveFolderPath)
            # print(folderName, "폴더 생성")

        # 파일 하나씩 처리
        for fileName in tqdm(sourceFiles, desc = f"{folderName} 폴더[{folderCount} / {totalFolderCount}] 길이 균일화 진행 중") :
            # 먼저 처리를 위해 데이터를 읽어오기
            sourceFilePath = os.path.join(sourceFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)

            try :
                wav_loaded = func.loadWAV(sourceFilePath)
            except :
                continue
            
            # 함수를 수행하고 파일을 1초씩 자른 wav_result 받기
            wav_result = func.regulateFile(wav_loaded)
            
            saveFilePath = saveFilePath[ : len(saveFilePath) - 4]
            for num in range(len(wav_result)) :
                func.saveFile(saveFilePath + "_" + str(num) + ".wav", wav_result[num])

    print(f"정규화 완료")
    if deleteFlag[0] and deleteFlag[1] :
        func.removeUsedFolder(sourcePath)


def getMFCC(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    sourcePath = os.path.join(basePath, "2. RegulatedSound")
    savePath = os.path.join(basePath, "3. MFCCs")

    # 필요하다면 먼저 MFCC폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)    

    # MFCC의 갯수별로 폴더를 따로 생성하기 위해 폴더 추가 생성
    savePath = os.path.join(savePath, str(set.N_MFCCS))

    # 먼저 폴더의 내용을 받아옴
    sourceFolders = os.listdir(sourcePath)
    totalFolderCount = len(sourceFolders)

    # 필요하다면 폴더 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)

    # N_MFCCS 값에 따른 변화를 알아내기 위해 N_MFCC 값별로 폴더를 생성
    if not os.path.isdir(savePath) :
        os.mkdir(savePath)
        print(f"MFCC = {set.N_MFCCS} 로 진행")

    for folderCount, folderName in enumerate(sourceFolders, start = 1) :
        sourceFolderPath = os.path.join(sourcePath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        # 폴더 내의 파일들을 불러옴
        try :
            sourceFiles = os.listdir(sourceFolderPath)
            print(folderName, "폴더 MFCC 구하기 시작")
        except :
            print(folderName, "내부의 파일을 읽어올 수 없었습니다")
            continue
        
        # 필요하다면 저장할 폴더도 생성함
        if not os.path.isdir(saveFolderPath) :
            os.mkdir(saveFolderPath)
            # print(folderName, "생성됨")

        # 각각의 파일들에 대하여 같은 동작 수행
        for fileName in tqdm(sourceFiles, desc = f"{folderName} 폴더[{folderCount} / {totalFolderCount}] MFCC 구하는 중)") :
            sourceFilePath = os.path.join(sourceFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)[ : -4] + ".pkl"
            # wav 파일을 읽어오는 부분
            try :
                wav_loaded = func.loadWAV(sourceFilePath)
            except :
                continue
                
            # MFCC를 함수를 이용해서 구하기 
            mfcc = func.getMFCC(wav_loaded)
            
            # pickle을 이용하여 저장
            with open(saveFilePath, "wb") as f :
                pickle.dump(mfcc, f)

    print("MFCC 구하기 완료")
    print(f"{sourcePath}폴더 삭제 중")
    #shutil.rmtree(sourcePath)
    print(f"{sourcePath}폴더 삭제 완료")


def labeling(target, labels) :
    # 베이스 경로 지정
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    dataPath = os.path.join(basePath, "3. MFCCs")
    dataPath = os.path.join(dataPath, str(set.N_MFCCS))

    # 저장할 기본 경로 지정
    saveBasePath = os.path.join(basePath, "4. ModelData")

    # 저장할 경로에 폴더가 없다면 생성
    if not(os.path.isdir(saveBasePath)) :
        os.mkdir(saveBasePath)

    # MFCC별로 다른 곳에 저장해야 하니 MFCC도 적용해서 경로 재설정
    savePath = os.path.join(saveBasePath, str(set.N_MFCCS))

    # 저장할 경로에 폴더가 없다면 생성
    if not(os.path.isdir(savePath)) :
        os.mkdir(savePath)

    dataFolders = os.listdir(dataPath)
    totalFolderCount = len(dataFolders)

    # 테스팅데이터와 트레이닝 데이터를 나눠 저장할 예정.
    trainingSavePath = os.path.join(savePath, "trainData")
    testingSavePath = os.path.join(savePath, "testData")

    # 각각의 폴더가 필요하다면 만들기
    if not(os.path.isdir(trainingSavePath)) :
        os.mkdir(trainingSavePath)

    if not(os.path.isdir(testingSavePath)) :
        os.mkdir(testingSavePath)

    trainingLabelFile = []
    testLabelFile = []

    for folderCount, dataFolderName in enumerate(dataFolders, start = 1) :
        # 폴더별로 라벨링을 위해 라벨넘버 먼저 labels 참고해서 정해두기
        labelNum = -1
        for value in labels.keys() :
            if (labels[value] == dataFolderName) :
                labelNum = value
                print(f"{dataFolderName}은(는) {labelNum}의 라벨 번호를 가집니다")
                break
        if (labelNum == -1) :
            print(f"{dataFolderName}은(는) 적절한 폴더가 아니어서 넘어갔습니다")
            continue
        
        # 데이터가 들어있는 폴더별 경로 지정
        dataFolderPath = os.path.join(dataPath, dataFolderName)

        # 폴더 내부의 파일들을 리스팅
        dataFiles = os.listdir(dataFolderPath)

        # 각 파일별 라벨 제작 및 복사를 위한 for문
        for num, dataFileName in enumerate(tqdm(dataFiles, desc = f"{dataFolderName} 폴더[{folderCount} / {totalFolderCount}] test, train 데이터 분리 중")) :
            if (dataFileName == ".DS_Store") :
                continue

            dataFilePath = os.path.join(dataFolderPath, dataFileName)      

            # 테스트 데이터와 트레이닝 데이터의 비율을 1:9로 맞추기 위해 10개중 하나씩만 테스트데이터 폴더로 넣기
            if num % 10 == 0 :
                shutil.copy(dataFilePath, testingSavePath)
                testLabelFile.append([dataFileName, labelNum])
            else :
                # 각 파일들은 저장경로에 그냥 복사
                shutil.copy(dataFilePath, trainingSavePath)

                # 파일별 이름, 그리고 labelNum은 labelFile에 저장해두기
                trainingLabelFile.append([dataFileName, labelNum])

    # pickle을 이용해 만들어진 LabelFile을 저장
    trainingLabelPath = os.path.join(trainingSavePath, "labels.pkl")
    testingLabelPath = os.path.join(testingSavePath, "labels.pkl")

    # 트레이닝용 라벨 저장
    with open(trainingLabelPath, "wb") as file :
        pickle.dump(trainingLabelFile, file)

    # 테스팅용 라벨 저장
    with open(testingLabelPath, "wb") as file :
        pickle.dump(testLabelFile, file)


    print("라벨링 완료")
    print(f"{dataPath}폴더 삭제 중")
    #shutil.rmtree(dataPath)
    print(f"{dataPath}폴더 삭제 완료")


# 사용자 지정 데이터셋
class SoundDataset(Dataset) :
    # init함수는 Dataset 객체가 생성될 때 한번만 실행된다. 여기서는 이미지, 주석파일이 포함된 디렉토리와 두가지 변형의 초기화를 진행
    def __init__(self, data_dir, transform = torch.tensor, target_transform = torch.tensor) :
        self.data_dir = data_dir
        # 피클 상태인 라벨 읽어오기
        with open(os.path.join(data_dir, "labels.pkl"), "rb") as file :
            self.labels = pickle.load(file)
        self.transform = transform
        self.target_transform = target_transform

    # len함수는 데이터셋의 샘플 갯수를 반환한다
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
    def __init__(self, label_len) :
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, label_len)
        )
    
    def forward(self, x) :
        logits = self.linear_relu_stack(x)
        return logits
        
def train_loop(dataloader, model, loss_fn, optimizer) :
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader) :
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 150 == 0 :
            loss, current = loss.item(), batch * len(X)
            print(f"Loss : {loss:>5f} {current:>5d} / {size:>5d}")

def test_loop(dataloader, model, loss_fn, epoch) :
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad() :
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

    wandb.log({
        "train_loss" : test_loss / len(X)
        }, step = epoch)
    return test_loss, 100 * correct

def Optimizing(path, label) :
    # 경로부터 지정
    basePath = set.dataPath
    basePath = os.path.join(basePath, path)
    basePath = os.path.join(basePath, "4. ModelData")
    basePath = os.path.join(basePath, str(set.N_MFCCS))

    trainPath = os.path.join(basePath, "trainData")
    testPath = os.path.join(basePath, "testData")

    train_dataset = SoundDataset(trainPath)
    test_dataset = SoundDataset(testPath)

    train_dataLoader = DataLoader(train_dataset, batch_size = set.BATCH_SIZE, shuffle = True)
    test_dataLoader = DataLoader(test_dataset, batch_size = set.BATCH_SIZE, shuffle = True)

    # train_datas, train_labels = next(iter(train_dataLoader))

    # print(f"Train data's Shape : {train_datas.size()}")
    # print(f"Train data's Value : {train_datas}")
    # print(f"Train label's Shape : {train_labels.size()}")
    # print(f"Train label's Value : {train_labels}")

    model = NeuralNetwork(len(label))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = set.LEARNING_RATE)

    for t in range(set.EPOCHS) :
        print(f"Epoch {t+1} \n ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
        train_loop(train_dataLoader, model, loss_fn, optimizer)
        test_loss, accuracy = test_loop(test_dataLoader, model, loss_fn, t)

        if (t + 1) % 10 == 0 :
            saveModel(model, t, test_loss, accuracy)
            

    print("Finished!")

    # 완성된 모델 저장하기
    savePath = os.path.dirname(os.path.realpath(__file__))
    torch.save(model, os.path.join(savePath, "model.pt"))
    torch.save(model.state_dict(), os.path.join(savePath, "state_dict.pt"))

# 일정 에포크 수마다 모델을 저장하고, 진행 경과를 텍스트파일을 만들어서 저장
def saveModel(model, epoch, test_loss, accuracy) :
    savePath = os.path.dirname(os.path.realpath(__file__))
    savePath = os.path.join(savePath, "folder_models")
    savePath = os.path.join(savePath, str(set.model_label))
    
    if not (os.path.isdir(savePath)) :
        os.mkdir(savePath)
    

    if (epoch + 1) % 50 == 0 :
        torch.save(model, os.path.join(savePath, "model_" + str(epoch + 1) + " Epochs_" + str(set.model_label) + ".pt"))
        torch.save(model.state_dict(), os.path.join(savePath, "state_dict_" + str(epoch + 1) + " Epochs_" + str(set.model_label) + ".pt"))
    with open(os.path.join(savePath, "Model_Process_" + str(set.model_label) + ".txt"), "a") as file :
        context = []
        context.append(f"Epoch {epoch + 1} \n")
        context.append(f"Accuracy : {accuracy:>0.1f}% Loss : {test_loss:>8f}\n\n")
        file.writelines(context)