#Decision Tree fraudcheck
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Fraud_data = pd.read_csv("C:/Users/cawasthi/Desktop/Data Science/R ML Code/Classification/Decision_tree_assignment/Fraud_check.csv")
Fraud_data.columns

Fraud_data['Fraud_Var'] = Fraud_data['Taxable.Income'] <= 30000
Fraud_data.dtypes
Fraud_data['Fraud_Var'].value_counts()

string_col = ['Undergrad','Marital.Status','Urban','Fraud_Var']

#to comvert string fields to numeric
from sklearn import preprocessing
for i in string_col:
    number = preprocessing.LabelEncoder()
    Fraud_data[i] = number.fit_transform(Fraud_data[i])
    
from sklearn.model_selection import train_test_split
train,test = train_test_split(Fraud_data,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,0:6],train['Fraud_Var'])

preds = model.predict(test.iloc[:,0:6])
pd.Series(preds).value_counts()
pd.crosstab(test['Fraud_Var'],preds)

# Accuracy = train
np.mean(train['Fraud_Var'] == model.predict(train.iloc[:,0:6])) # 1

# Accuracy = Test
np.mean(preds==test['Fraud_Var']) # 1


#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")

rf.fit(train.iloc[:,0:6],train['Fraud_Var'])
preds = rf.predict(test.iloc[:,0:6])

# Accuracy = train
np.mean(train['Fraud_Var'] == rf.predict(train.iloc[:,0:6])) # 1

# Accuracy = Test
np.mean(preds==test['Fraud_Var']) # 1
