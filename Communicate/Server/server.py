import os
import time
import shutil
import uvicorn
import wav_functions as func
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from dnn_model import NeuralNetwork
from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

@app.get("/")
async def testRoot() :
    message = []
    message.append("Hello, World!")
    message.append("Welcome to my server")
    return {
        "message" : message
    }

@app.post("/uploadFile")
async def fileUpload(file : UploadFile = Form(), userName : str = Form()) :
    message = []
    try :
        # 받은 음원 파일을 저장하는 부분
        dirName = os.path.dirname(os.path.realpath(__file__))
        dirName = os.path.join(dirName, "received")
        dirName = os.path.join(dirName, userName)

        # 유저별 폴더가 없다면 새로 만들기
        if not (os.path.isdir(dirName)) : 
            os.mkdir(dirName)
         
        # 저장할 파일은 유저별 폴더 만들고 분류결과_시간으로 저장
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H시%M분%S초')
        fileName =  "_" + timestamp + ".wav"
        savePath = os.path.join(dirName, fileName)

        with open(savePath, "wb") as buffer :
            shutil.copyfileobj(file.file, buffer)

        # 받은 음원 파일을 저장하는 부분
        modelPath = os.path.dirname(os.path.realpath(__file__))
        modelPath = os.path.join(modelPath, "modelFile")
        modelPath = os.path.join(modelPath, "state_dict.pt")

        result = func.all_in_one(savePath, modelPath)
        
        newFileName = result + fileName
        newSavePath = os.path.join(dirName, newFileName)

        os.rename(savePath, newSavePath)

        message.append(f"{newFileName} has saved.")
        message.append(f"Thanks! {userName}")
        
        print("정상적으로 반환됨")
        return {
            'message' : message,
            'prediction' : result
        }
    except Exception as e: 
        print(e)
        print("서버 에러 발생 반환됨")
        message.append(f"서버에서 에러 발생 : {e}")
        return {
            "message" : message
        }


if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
