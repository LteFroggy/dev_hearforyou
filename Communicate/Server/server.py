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
         
        # 저장할 파일명은 유저명_시간.wav 로 일단
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        fileName = userName + "_" + timestamp + ".wav"
        savePath = os.path.join(dirName, fileName)

        with open(savePath, "wb") as buffer :
            shutil.copyfileobj(file.file, buffer)

        # 받은 음원 파일을 저장하는 부분
        modelPath = os.path.dirname(os.path.realpath(__file__))
        modelPath = os.path.join(modelPath, "modelFile")
        modelPath = os.path.join(modelPath, "state_dict.pt")

        result = func.all_in_one(savePath, modelPath)

        message.append(f"{fileName} has saved.")
        message.append(f"Thanks! {userName}")
        return {
            'message' : message,
            'prediction' : result
        }
    except Exception as e: 
        print(e)
        message.append(f"서버에서 에러 발생 : {e}")
        return {
            "message" : message
        }


if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)