import os


# Read wav File as TXT

dirName = os.path.dirname(os.path.realpath(__file__))
filePath = os.path.join(dirName, "7.동물_8979_1.wav")
newFilePath = os.path.join(dirName, "그냥만들어봄.wav")

with open(filePath, "rb") as f :
    result = f.readlines()

print(len(result))

with open(newFilePath, "wb") as f :
    f.writelines(result)