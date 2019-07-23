#Decision Tree fraudcheck
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("C:/Users/cawasthi/Desktop/Data Science/R ML Code/Classification/Decision_tree_assignment/iris.csv")
iris.columns
iris['Species'].value_counts()

from sklearn.model_selection import train_test_split
train,test = train_test_split(iris,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,0:4],train['Species'])

preds = model.predict(test.iloc[:,0:4])
pd.Series(preds).value_counts()
pd.crosstab(test['Species'],preds)

# Accuracy = train
np.mean(train['Species'] == model.predict(train.iloc[:,0:4])) # 1

# Accuracy = Test
np.mean(preds==test['Species']) # 1

#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=10,criterion="entropy")

rf.fit(train.iloc[:,0:4],train['Species'])

np.mean(train['Species'] == rf.predict(train.iloc[:,0:4])) #0.99

np.mean(test['Species'] == rf.predict(test.iloc[:,0:4])) #0.93

