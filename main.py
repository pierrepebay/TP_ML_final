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
    # data.classifyAdaBoost(n_est=50)
    # print("AdaBoost Accuracy:",data.getModelAccuracy())
    n_estimators = range(1, 200, 10)
    x = []
    accuracyY = []
    hl, = plt.plot([], [])
    for e in n_estimators:
        data.classifyRandomForest(n_e=e, v=3)
        a = data.getModelAccuracy()
        x.append(e)
        accuracyY.append(a)
        print("Random Forest Accuracy with {} trees:".format(e), a)
        print(hl.get_xdata(), hl.get_ydata())
        hl.set_xdata(np.append(hl.get_xdata(), e))
        hl.set_ydata(np.append(hl.get_ydata(), a))
        plt.draw()
        plt.pause(0.1)
    # data.classifyKNeighbors()
    # print("K Nearest Neighbors Accuracy:",data.getModelAccuracy())
    # data.classifyLinearSVC()
    # print("Linear SVC Accuracy:",data.getModelAccuracy())

if __name__ == main():
    main()