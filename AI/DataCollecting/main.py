import functions.getFile as getFile
import functions.cutFile as cutFile
import functions.read_csv as read_csv
import functions.changeExt as changeExt

if __name__ == "__main__" :
    soundName = ["Female speech, woman speaking", "Child speech, kid speaking", "Clapping"]

    for i in range(len(soundName)) :
        # # 1
        # read_csv.main(soundName[i])

        # # 2
        # getFile.main(soundName[i])

        # # 3
        # changeExt.main(soundName[i])

        # 4
        cutFile.main(soundName[i])
        