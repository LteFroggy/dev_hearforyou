from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def testRoot() :
    message = []
    message.append("Hello, World!")
    message.append("Welcome to my server")
    return {
        "message" : message
    }
