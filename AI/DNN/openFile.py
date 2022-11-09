import os
import librosa
from matplotlib import pyplot as plt


path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, "data")
path = os.path.join(path, "Urban")
path1 = os.path.join(path, "2. RemovedSound")
path2 = os.path.join(path, "3. RegulatedSound")

path1_folder = os.path.join(path1, os.listdir(path1)[0])
path2_folder = os.path.join(path2, os.listdir(path2)[0])

path1_file = os.path.join(path1_folder, "196561-3-0-9.wav")


fileName1 = os.listdir(path1_folder)[0]
fileName2 = os.listdir(path2_folder)[0]

path1_wav = librosa.load(path1_file)[0]


plt.plot(path1_wav)
plt.show()

plt.cla()

i = 0

while True :
    try :
        tmp = "196561-3-0-9_" + str(i) + ".wav"
        path2_file = os.path.join(path2_folder, tmp)
        path2_wav = librosa.load(path2_file)[0]
        plt.subplot(1, 5, (2 * i) + 1)
        plt.plot(path2_wav)
        plt.title(str(i+1) + "st File")
        i += 1
    except :
        print(f"{i}까지 구하고 끝남")
        break

plt.show()