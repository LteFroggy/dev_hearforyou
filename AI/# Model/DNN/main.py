import Processes
import settings

if __name__ == "__main__" :
    pathName = "maindata"
    label = settings.main_label
    # # 1.
    # Processes.removeSilence(pathName)

    # # 2.
    # Processes.lengthRegulate(pathName)

    # # 3.
    # Processes.getMFCC(pathName)

    # # 4.
    # Processes.labeling(pathName, label)

    # 5.
    Processes.Optimizing(pathName, label)