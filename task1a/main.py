import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from statistics import mean


def getRMSE(l, df):
    train_y = df.y
    train_X = df.drop(columns='y')
    regressor = Ridge(alpha=l)
    scores = cross_val_score(regressor, train_X, train_y, cv=10, scoring='neg_root_mean_squared_error')
    return abs(mean(scores))


df = pd.read_csv('train.csv')
lambdas = [0.1, 1, 10, 100, 200]
results = []

for l in lambdas:
    results.append(getRMSE(l, df))

data = np.array(results)
dfResult = pd.DataFrame(data=data)
dfResult.to_csv('result.csv', index=False, header=False)