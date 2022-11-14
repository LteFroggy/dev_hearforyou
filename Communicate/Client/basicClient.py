import os
import requests
from pathlib import Path

address = "http://127.0.0.1:8000/"
target = "uploadFile"

path_url = address + target
dirName = os.path.dirname(os.path.realpath(__file__))
fileName = "7.동물_8979_1.wav"
path_file = os.path.join(dirName, fileName)

f = open(path_file, "rb")
print(type(f))

datas = {
    "file" : f
}

result = requests.post(path_url, data = datas)
returns = result.json()

if result.status_code != 200 :
    print(f"Error : Status Code : {result.status_code}")

for i, key in enumerate(returns.keys()) :
    print(f"{key} : {returns[key]}")
    5
f.close()