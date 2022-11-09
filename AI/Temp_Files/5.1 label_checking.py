import pickle
import os
import settings as set


path = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"), "5. trainingData"), str(set.N_MFCCS))
fileName = "labels.pickle"

filePath = os.path.join(path, fileName)
 
with open(filePath, "rb") as f :
    labels = pickle.load(f)
    for i in range(len(labels)) :
        if i % 50 == 0 :
            print(f"FileName : {labels[i][0]}, Label : {labels[i][1]}")