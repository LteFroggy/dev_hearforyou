import os
import torch
import wav_functions as func


if __name__ == "__main__" :
    path = "/Users/hsjack/Documents/GitHub/dev_hearforyou/Communicate/Server/received/input_정호종.wav"
    modelPath = "/Users/hsjack/Documents/GitHub/dev_hearforyou/Communicate/Server/modelFile/state_dict.pt"

    file = func.loadWAV(path)
    print(f"길이 : {len(file)}")
    print(f"모양 : {file.shape}")
    print(f"타입 : {type(file)}")
    print(file)

    mfcc = func.getMFCC(file)
    print(f"길이 : {len(mfcc)}")
    print(f"모양 : {mfcc.shape}")
    print(f"타입 : {type(mfcc)}")
    print(mfcc)


    result = func.all_in_one(path, modelPath)
    print(result)