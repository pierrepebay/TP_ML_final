import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

import Data

def main():
    data = Data.data("data/xTrain.csv", "data/yTrain.csv", "data/xEval.csv", trainSize=0.8)

    data.classifyRandomForest(n_e=200, v=3)

    confusion_matrix = data.getConfusionMatrix()
    print(confusion_matrix)
    print("f1 score: ", data.getF1Score())
    metrics.RocCurveDisplay.from_predictions(data.yTest, data.yPred)
    plt.show()
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