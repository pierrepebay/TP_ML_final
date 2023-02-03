import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

xTrainDataframe = pd.read_csv("data/xTrain.csv", delimiter=",")
yTrainDataframe = pd.read_csv("data/yTrain.csv", names=["y"])
# xTrainDataframe.replace("", np.nan, inplace=True)
# xTrainDataframe.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
# xTrainDataframe.fillna(method='ffill', inplace=True)
# xTrainDataframe.fillna(method='bfill', inplace=True)
# xTrainDataframe.fillna(0, inplace=True)
print(yTrainDataframe.head())
yTrainDataframe = yTrainDataframe[yTrainDataframe["y"].astype(str).str.contains("\"\" ")==False]
yTrainDataframe.to_csv("test.csv", index=False)

test = []
print(len(np.unique(xTrainDataframe["idCow"])))