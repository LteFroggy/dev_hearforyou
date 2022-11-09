import Processes
import settings

if __name__ == "__main__" :
    pathName = "test"
    # 1.
    Processes.removeSilence(pathName)

    # 2.
    Processes.lengthRegulate(pathName)

    # 3.
    #Processes.getMFCC(pathName)

    # 4.
    #Processes.labeling(pathName, settings.TS_Labels)