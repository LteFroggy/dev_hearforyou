import settings as set
import CNNModel as model
import preprocessing as pre

if __name__ == "__main__" :
    targetName = "TestData"
    labelName = set.test_Labels
    
    # 1
    pre.removeSilence(targetName)

    # 2
    pre.lengthRegulate(targetName)

    # 3
    pre.saveAsImage(targetName)

    # 4
    pre.cutImage(targetName)

    # 5
    pre.labeling(targetName, labelName)

    # 6
    model.learnModel(targetName)