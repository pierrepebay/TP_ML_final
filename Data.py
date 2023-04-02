import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, AllKNN

class data:
    def __init__(self, xTrainPath, yTrainPath, xEvalPath, trainSize, data_augmentation=True):
        self.xTrainDataframe = pd.read_csv(xTrainPath, delimiter=",")
        self.yTrainDataframe = pd.read_csv(yTrainPath, names=["y"])
        self.xEvalDataframe = pd.read_csv(xEvalPath, delimiter=",")

        self.allDataFrame = pd.concat([self.xTrainDataframe, self.yTrainDataframe], axis=1)
        self.allDataFrame.drop('idCow', axis=1, inplace=True)

        self.allDataFrame['data_hour'] = self.allDataFrame['data_hour'].apply(lambda date_str: pd.to_datetime(date_str).hour)

        self.m, self.n = self.xTrainDataframe.shape

        self.trainSize = trainSize

        print("Removing erroneous lines...")
        self.removeErrorLines()
        print("Normalizing time-location values...")
        self.normalizeTimeValues()
        self.allDataFrame.to_csv("normalizedXTrain.csv",index=False)
        print("Setting train/test splits...")
        self.setSplits()
        
        if type(data_augmentation) == bool: 
            if data_augmentation:
                self.dataAugmentation()
        else:
            self.dataAugmentation(data_augmentation)
    
    def dataAugmentation(self, imbalance_ratio="auto"):
        s, ns = np.count_nonzero(self.yTrain == 1), np.count_nonzero(self.yTrain == 0)
        print(f'{s} sick cows, {ns} non sick cows: BEFORE SMOTE')
        #imbalance = s // ns
        self.xTrain, self.yTrain = SMOTE(sampling_strategy=imbalance_ratio).fit_resample(self.xTrain, self.yTrain)
        # self.xTrain, self.yTrain = ClusterCentroids(sampling_strategy=0.2).fit_resample(self.xTrain, self.yTrain) # ca prend beaucoup de temps a tourner
        s, ns = np.count_nonzero(self.yTrain == 1), np.count_nonzero(self.yTrain == 0)
        print(f'{s} sick cows, {ns} non sick cows: AFTER SMOTE')

    def normalizeTimeValues(self):
        for index, row in self.allDataFrame.iterrows():
            for i in range(0,24):
                all_i = row[f"all{i}"]
                rest_i = row[f"rest{i}"]
                eat_i = row[f"eat{i}"]
                norm_sum = all_i + rest_i + eat_i
                new_all_i = all_i / norm_sum
                new_rest_i = rest_i / norm_sum
                new_eat_i = eat_i / norm_sum
                self.allDataFrame.at[index,f"all{i}"] = new_all_i
                self.allDataFrame.at[index,f"rest{i}"] = new_rest_i
                self.allDataFrame.at[index,f"eat{i}"] = new_eat_i

    def setSplits(self):
        x = self.getXTrainArray()
        y = self.getYTrainArray()
        y = y.astype('int')
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        n_components = 3  # Set the desired number of components (dimensions)
        pca = PCA(n_components=n_components)
        principal_components = pca.fit(x)
        X_pca = pca.transform(x)
        print(pca.explained_variance_ratio_)

        # scatter plot pca result
        principalDf = pd.DataFrame(data = X_pca, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
        print(principalDf)

        finalDf = pd.concat([principalDf, pd.DataFrame(data=y,columns=['y'])], axis = 1)

        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        
        # Creating plot
        ax.scatter3D(finalDf['principal component 1'].to_numpy(),finalDf['principal component 2'].to_numpy(),finalDf['principal component 3'].to_numpy(),c=finalDf['y'], cmap = 'prism')
        plt.title("simple 3D scatter plot")
        
        # show plot
        plt.show()
        
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(x, y, test_size=(1 - self.trainSize))

    def getXTrainArray(self):
        return self.allDataFrame.drop('y', axis=1).to_numpy()
    
    def getYTrainArray(self):
        return self.allDataFrame['y'].to_numpy()

    def removeErrorLines(self):
        self.allDataFrame.dropna(inplace=True)
        self.m, self.n = self.allDataFrame.shape
    
    def getModelAccuracy(self):
        return metrics.accuracy_score(self.yTest, self.yPred)
    
    def getModelBalancedAccuracy(self):
        return metrics.balanced_accuracy_score(self.yTest, self.yPred)

    def getConfusionMatrix(self):
        return metrics.confusion_matrix(self.yTest, self.yPred)
    
    def getF1Score(self):
        return metrics.f1_score(self.yTest, self.yPred)

    def classifyAdaBoost(self,n_est):
        abc = AdaBoostClassifier(n_estimators=n_est, learning_rate=1)

        # Train Adaboost Classifer
        model = abc.fit(self.xTrain, self.yTrain)

        # Predict the response for test dataset
        self.yPred = model.predict(self.xTest)
    
    def classifyRandomForest(self, n_e, v, max_d=None, min_smp_splt=2):
        print("*** Classifying with random forest...")
        clf = RandomForestClassifier(n_estimators = n_e, verbose=v, random_state=0, max_depth=max_d, min_samples_split=min_smp_splt)

        # Train Random Forest Classifer
        self.model = clf.fit(self.xTrain, self.yTrain)

        # Predict the response for test dataset
        self.yPred = self.model.predict(self.xTest)
    
    def classifyKNeighbors(self):
        def lorentzian_distance(x, y):
            return metrics.pairwise.distance.minkowski(x, y, p=1)

        neigh = KNeighborsClassifier(n_neighbors=3, metric=lorentzian_distance)

        # Train Random Forest Classifer
        self.model = neigh.fit(self.xTrain, self.yTrain)

        # Predict the response for test dataset
        self.yPred = self.model.predict(self.xTest)
    
    def classifyLinearSVC(self):
        clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))

        # Train LinearSVC Classifier
        model = clf.fit(self.xTrain, self.yTrain)

        # Predict the response for test dataset
        self.yPred = model.predict(self.xTest)
    
    def computeImbalance(self):
        self.sickData = 0
        self.nonSickData = 0
        s = 0
        for i in range(self.m):
            if self.allDataFrame.iloc[i]['y'] == 1.0:
                s += 1
                self.sickData += 1
            elif self.allDataFrame.iloc[i]['y'] == 0.0:
                s += 1
                self.nonSickData += 1
            else:
                raise ValueError("value not 1 or 0")
        self.sickData /= self.m
        self.nonSickData /= self.m
        return self.sickData, self.nonSickData
        

    def toCsv(self):
        self.xTrainDataframe.to_csv("xTrainClean.csv",index=False)
        self.yTrainDataframe.to_csv("yTrainClean.csv",index=False)

        self.allDataFrame.to_csv("allData.csv", index=False)
