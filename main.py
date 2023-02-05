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
    data.classifyAdaBoost(n_est=50)
    print("AdaBoost Accuracy:",data.getModelAccuracy())
    data.classifyRandomForest()
    print("Random Forest Accuracy:",data.getModelAccuracy())
    data.classifyKNeighbors()
    print("K Nearest Neighbors Accuracy:",data.getModelAccuracy())
    data.classifyLinearSVC()
    print("Linear SVC Accuracy:",data.getModelAccuracy())

if __name__ == main():
    main()