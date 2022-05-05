#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# In[2]:


ISTEST = False


# In[3]:


def format(stri):
    df = pd.read_csv(stri)
    
    majority = df.loc[df.Active == 0].copy()
    minority = df.loc[df.Active == 1].copy()
    
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority),random_state=42)
    
    df_upsampled = pd.concat([majority, minority_upsampled])
    #print(df_upsampled.Active.value_counts())
    return df_upsampled


# In[4]:


def format2(df):
    
    new = [char for char in df["Sequence"].str]

    newDF = pd.DataFrame()
    newDF["one"] = new[0]
    newDF["two"] = new[1]
    newDF["three"] = new[2]
    newDF["four"] = new[3]
    return newDF


# In[5]:


X_train = format2(pd.read_csv('train.csv'))
y_train = pd.read_csv('train.csv')['Active']

if ISTEST:
    X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X_train, y_train, train_size=0.8, random_state=42)
    
    X_train_80 = X_train_80.copy()
    y_train_80 = y_train_80.copy()
    X_test_20 = X_test_20.copy()
    y_test_20 = y_test_20.copy()

df = format('train.csv')
X_train = format2(df)
y_train = df["Active"]
X_test = format2(pd.read_csv("test.csv"))


# In[6]:


encoder = OneHotEncoder(sparse=False)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

#print(X_train.shape)
if ISTEST:
    X_test_20 = encoder.transform(X_test_20)
    X_train_80 = encoder.transform(X_train_80)


# In[7]:


if ISTEST:
    model = MLPClassifier(solver='adam', early_stopping=True, random_state=2).fit(X_train_80, y_train_80)
else:
    model = MLPClassifier(solver='adam', early_stopping=True, random_state=2).fit(X_train, y_train)


# In[8]:


if ISTEST:
    y_20_predicted = (model.predict_proba(X_test_20)[:,1] >= 0.49).astype(int)
    print(f1_score(y_test_20, y_20_predicted))


# In[9]:


sol = (model.predict(X_test))

pd.DataFrame(sol, columns=['sol']).to_csv('result.csv', index=False, header=False)


# In[ ]:




