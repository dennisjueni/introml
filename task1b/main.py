import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


df_in = pd.read_csv('train.csv', index_col = 0)
train_y = df_in.y
train_x = df_in.drop(columns='y')

data = pd.concat([train_x, np.square(train_x), np.exp(train_x), np.cos(train_x)], axis=1)

df = pd.DataFrame(data=data)

df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20']

df['x21'] = pd.Series([1 for x in range(len(df.index))])
df['y'] = train_y

errors = []
coeffs = pd.DataFrame()
idx = 0

kf = KFold(n_splits=10)
for train, test in kf.split(df):
    X = df.loc[train].drop(columns='y')
    y = df.loc[train]['y']
    X_test = df.loc[test].drop(columns='y')
    y_test = df.loc[test]['y']

    regressor = KernelRidge(alpha=0.1)
    regressor.fit(X, y)
    sol = regressor.predict(X_test)
    errors.append(mean_squared_error(sol, y_test))
    #coeffs.append(np.dot(X.transpose(), regressor.dual_coef_))

    coeffs[str(idx)] = np.dot(X.transpose(), regressor.dual_coef_)
    idx += 1

print(errors)
#result = pd.DataFrame(coeffs[errors.index(min(errors))])

result = coeffs.mean(axis=1)

result.to_csv('result.csv', index=False, header=False)