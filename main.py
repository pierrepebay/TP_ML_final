import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

import Data

def main():
    data = Data.data("data/xTrain.csv", "data/yTrain.csv", "data/xEval.csv", trainSize=0.95)
    data.removeErrorLines()
    data.setSplits()
    s, ns = data.computeImbalance()
    print(s, ns)
    data.classifyRandomForest(n_e=10, v=3)

    confusion_matrix = data.getConfusionMatrix()
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    print("True Negatives: {}, False Negatives: {}, True Positives: {}, False Positives: {}".format(confusion_matrix[0,0], confusion_matrix[1,0], confusion_matrix[1,1], confusion_matrix[0,1],))

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print("TPR: ", TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    print("TNR: ", TNR) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print("PPV: ", PPV)
    # Negative predictive value
    NPV = TN/(TN+FN)
    print("NPV: ", NPV)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print("FPR: ", FPR)
    # False negative rate
    FNR = FN/(TP+FN)
    print("FNR: ", FNR)
    # False discovery rate
    FDR = FP/(TP+FP)
    print("FDR: ", FDR)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print("ACC: ", ACC)
    
    # data.classifyAdaBoost(n_est=50)
    # print("AdaBoost Accuracy:",data.getModelAccuracy())
    # n_estimators = range(10, 200, 10)
    # x = []
    # accuracyY = []
    # for e in n_estimators:
    #     data.classifyRandomForest(n_e=e, v=3)
    #     a = data.getModelAccuracy()
    #     x.append(e)
    #     accuracyY.append(a)
    #     print("Random Forest Accuracy with {} trees:".format(e), a)
    # plt.plot(x, accuracyY)
    # plt.show()
    # data.classifyKNeighbors()
    # print("K Nearest Neighbors Accuracy:",data.getModelAccuracy())
    # data.classifyLinearSVC()
    # print("Linear SVC Accuracy:",data.getModelAccuracy())

if __name__ == main():
    main()