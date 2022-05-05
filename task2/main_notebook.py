# Imports
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier


# Structure the data
def structureData(stri):
       # Get Dataframes to work with
       df = pd.read_csv(stri)
       features = df.columns
       avgs = pd.DataFrame(0, index=np.arange(1), columns=features)

       # Find averages
       avgs = df.median().copy()

       # Groupby PID
       dfsmall = df.groupby('pid', sort=False)
       dfgrouped = dfsmall.agg(np.mean).copy()
       # Replace nan with avgs overall
       dfgrouped.fillna(value=avgs, inplace=True)
       return dfgrouped


# Get the Datasets
X_train = structureData("train_features.csv")
X_test = structureData("test_features.csv")

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = pd.read_csv("train_labels.csv")
y_train_sub1 = y_train[['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate','LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct','LABEL_EtCO2']].copy()
y_train_sub2 = y_train[['LABEL_Sepsis']].copy()
y_train_sub3 = y_train[['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']].copy()


# Get feature names and such
features1 = y_train_sub1.columns
features2 = y_train_sub2.columns
features3 = y_train_sub3.columns
y_train_sub2 = np.array(y_train_sub2.copy())


### Subtask 1
# Fit model:
clf_sub1 = MLPClassifier(activation='tanh', early_stopping=True, random_state=420).fit(X_train, y_train_sub1)


# Predict Testset
sub1_data = clf_sub1.predict_proba(X_test)
sol_sub1 = pd.DataFrame(columns=['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate','LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct','LABEL_EtCO2'],data=sub1_data)


### Subtask 2
clf_sub2 = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=420).fit(X_train, np.ravel(y_train_sub2))


# Predict
sub2_data = 1 - clf_sub2.predict_proba(X_test)[:, 0]
sol_sub2 = pd.DataFrame(columns=['LABEL_Sepsis'],data=sub2_data)


### Subtask 3
MORegression = MultiOutputRegressor(GradientBoostingRegressor(random_state=420)).fit(X_train, y_train_sub3)


#Predict
sub3_data = MORegression.predict(X_test)
sol_sub3 = pd.DataFrame(columns=['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate'],data=sub3_data)


### Final Solution
def getPids(stri):
    df = pd.read_csv(stri)
    dfsmall = df['pid']
    return dfsmall

ps = getPids("test_features.csv")[::12]

final_sol = pd.DataFrame(columns=['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
       'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
       'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
       'LABEL_EtCO2', 'LABEL_Sepsis', 'LABEL_RRate', 'LABEL_ABPm',
       'LABEL_SpO2', 'LABEL_Heartrate'])

final_sol['pid'] = ps
final_sol['LABEL_BaseExcess'] = [vals for vals in sol_sub1['LABEL_BaseExcess']]
final_sol['LABEL_Fibrinogen'] = [vals for vals in sol_sub1['LABEL_Fibrinogen']]
final_sol['LABEL_AST'] = [vals for vals in sol_sub1['LABEL_AST']]
final_sol['LABEL_Alkalinephos'] = [vals for vals in sol_sub1['LABEL_Alkalinephos']]
final_sol['LABEL_Bilirubin_total'] = [vals for vals in sol_sub1['LABEL_Bilirubin_total']]
final_sol['LABEL_Lactate'] = [vals for vals in sol_sub1['LABEL_Lactate']]
final_sol['LABEL_TroponinI'] = [vals for vals in sol_sub1['LABEL_TroponinI']]
final_sol['LABEL_SaO2'] = [vals for vals in sol_sub1['LABEL_SaO2']]
final_sol['LABEL_Bilirubin_direct'] = [vals for vals in sol_sub1['LABEL_Bilirubin_direct']]
final_sol['LABEL_EtCO2'] = [vals for vals in sol_sub1['LABEL_EtCO2']]
final_sol['LABEL_Sepsis'] = [vals for vals in sol_sub2['LABEL_Sepsis']]
final_sol['LABEL_RRate'] = [vals for vals in sol_sub3['LABEL_RRate']]
final_sol['LABEL_ABPm'] = [vals for vals in sol_sub3['LABEL_ABPm']]
final_sol['LABEL_SpO2'] = [vals for vals in sol_sub3['LABEL_SpO2']]
final_sol['LABEL_Heartrate'] = [vals for vals in sol_sub3['LABEL_Heartrate']]


# Write to csv
toPd = pd.DataFrame(final_sol) 
toPd.to_csv("prediction.zip", index = False, header=True, float_format='%.3f', compression='zip')





