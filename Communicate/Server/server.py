import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile

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
async def fileUpload(file : UploadFile = File(...)) :
    try :
        dirName = os.path.dirname(os.path.realpath(__file__))
        dirName = os.path.join(dirName, "received")
        savePath = os.path.join(dirName, file.filename)

        with open(savePath, "wb") as buffer :
            shutil.copyfileobj(file.file, buffer)

        return {
            "message" : f"{file.filename} has saved. "
        }
    except Exception as e: 
        print(e)
        return {
            "message" : "Save Failed"
        }


if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 5000
    )