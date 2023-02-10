import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics

class data:
    def __init__(self, xTrainPath, yTrainPath, xEvalPath, trainSize):
        self.xTrainDataframe = pd.read_csv(xTrainPath, delimiter=",")
        self.yTrainDataframe = pd.read_csv(yTrainPath, names=["y"])
        self.xEvalDataframe = pd.read_csv(xEvalPath, delimiter=",")

        self.allDataFrame = pd.concat([self.xTrainDataframe, self.yTrainDataframe], axis=1)

        self.m, self.n = self.xTrainDataframe.shape

        self.trainSize = trainSize

    def setSplits(self):
        x = self.getXTrainArray()
        y = self.getYTrainArray().astype('int')
        y=y.astype('int')
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(x, y, test_size=(1 - self.trainSize))

    def getXTrainArray(self):
        return self.allDataFrame.to_numpy()[:,range(1,74)]
    
    def getYTrainArray(self):
        return self.allDataFrame.to_numpy()[:,74]

    def removeErrorLines(self):
        self.allDataFrame.dropna(inplace=True)
    
    def getModelAccuracy(self):
        return metrics.accuracy_score(self.yTest, self.yPred)

    def classifyAdaBoost(self,n_est):
        abc = AdaBoostClassifier(n_estimators=n_est, learning_rate=1)

        # Train Adaboost Classifer
        model = abc.fit(self.xTrain, self.yTrain)

        # Predict the response for test dataset
        self.yPred = model.predict(self.xTest)
    
    def classifyRandomForest(self, n_e, v):
        clf = RandomForestClassifier(n_estimators = n_e, verbose=v, random_state=0)

        # Train Random Forest Classifer
        model = clf.fit(self.xTrain, self.yTrain)

        # Predict the response for test dataset
        self.yPred = model.predict(self.xTest)
    
    def classifyKNeighbors(self):
        neigh = KNeighborsClassifier()

        # Train Random Forest Classifer
        model = neigh.fit(self.xTrain, self.yTrain)

        # Predict the response for test dataset
        self.yPred = model.predict(self.xTest)
    
    def classifyLinearSVC(self):
        clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))

        # Train LinearSVC Classifier
        model = clf.fit(self.xTrain, self.yTrain)

        # Predict the response for test dataset
        self.yPred = model.predict(self.xTest)

    def toCsv(self):
        self.xTrainDataframe.to_csv("xTrainClean.csv",index=False)
        self.yTrainDataframe.to_csv("yTrainClean.csv",index=False)

        self.allDataFrame.to_csv("allData.csv", index=False)