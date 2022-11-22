import functions.getFile as getFile
import functions.cutFile as cutFile
import functions.read_csv as read_csv
import functions.changeExt as changeExt

if __name__ == "__main__" :
    soundName = "Screaming"

    # 1
    read_csv.main(soundName)

    # 2
    getFile.main(soundName)

    # 3
    changeExt.main(soundName)

    # 4
    cutFile.main(soundName)