import os
import torch
import shutil
import numpy as np
import settings as set
import PIL.Image as ImgLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path

cutColumn = 4
cutRow = 5

def cutImage(target) :
    basePath = set.dataPath
    basePath = os.path.join(basePath, target)
    soundPath = os.path.join(basePath, "2-1. RegulatedPhoto")
    savePath = os.path.join(basePath, "2-2. CuttedPhoto")

    if not (os.path.isdir(savePath)) :
        os.mkdir(savePath)

    soundFolderList = os.listdir(soundPath)
    totalFolderCount = len(soundFolderList)

    for folderCount, folderName in enumerate(soundFolderList, start = 1) :
        soundFolderPath = os.path.join(soundPath, folderName)
        saveFolderPath = os.path.join(savePath, folderName)

        if not (os.path.isdir(saveFolderPath)) :
            os.mkdir(saveFolderPath)

        # listdir 중 폴더가 아니면 에러가 날 수 있음
        try :
            soundFileList = os.listdir(soundFolderPath)
        except :
            continue
        
        for fileName in tqdm(soundFileList, desc = f"{folderName} 폴더 [{folderCount} / {totalFolderCount}] 이미지 커팅 진행 중") :
            soundFilePath = os.path.join(soundFolderPath, fileName)
            saveFilePath = os.path.join(saveFolderPath, fileName)

            # 이미지파일 불러오기
            imgFile = ImgLoader.open(soundFilePath)

            # 이미지파일 텐서화하기
            img_tensor = transforms.ToTensor()(imgFile)

            # 원본 크기 출력
            # print(f"원본 Shape : \t{img_tensor.shape}")

            # 앞쪽 열 10개, 뒤쪽 행 열개 자르기
            cuttedImg = torch.Tensor(len(img_tensor), len(img_tensor[0]) - (cutColumn), len(img_tensor[0][0]) - cutRow)
            for i in range(len(img_tensor)) :
                cuttedImg[i] = img_tensor[i][ : -cutColumn, cutRow : ]
            # print(f"값 : {cuttedImg[0]}")
            # print(f"길이 1 : {len(cuttedImg)}")
            # print(f"길이 2 : {len(cuttedImg[0])}")
            # print(f"길이 3 : {len(cuttedImg[0][0])}")
            imgFile = transforms.ToPILImage()(cuttedImg)
            imgFile.save(saveFilePath)

    # print(f"{soundPath}폴더 삭제 중")
    # shutil.rmtree(soundPath)
    # print(f"{soundPath}폴더 삭제 완료")