import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('train.csv')
df.set_index('Id', inplace=True)

train_y = df.y
train_X = df.drop(columns='y')

regressor = LinearRegression(fit_intercept=False)
regressor.fit(train_X, train_y)

test_X = pd.read_csv('test.csv')
test_X.set_index('Id', inplace=True)

neededResult = test_X.sum(axis=1).div(10)

test_y = regressor.predict(test_X)
result = pd.DataFrame(data=test_y)
result.index += 10000
result.columns = ['y']
result.index.names = ['Id']

neededResult = test_X.sum(axis=1).div(10)

print(mean_squared_error(neededResult, result)**0.5)

result.to_csv('test_result.csv')