import cutImage as cut
import labeling as lab
import settings as set
import CNNModel as model
import saveAsImage as save
import preprocessing as pre

if __name__ == "__main__" :
    targetName = "TestData"
    labelName = set.test_Labels
    
    # # 1
    # pre.removeSilence(targetName)

    # # 2
    # pre.lengthRegulate(targetName)

    # # 3
    # save.saveAsImage(targetName)

    # # 4
    # cut.cutImage(targetName)

    # 5
    lab.labeling(targetName, labelName)

    # 6
    model.learnModel(targetName)