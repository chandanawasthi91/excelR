#Decision Tree fraudcheck
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

computer_sale = pd.read_csv("C:/Users/cawasthi/Desktop/Data Science/R ML Code/Classification/Decision_tree_assignment/Company_Data.csv")

computer_sale.columns
computer_sale.dtypes

#lets considersales are high if > 7.5, creating the variable to identify sales as high or low
computer_sale['Sale_Var'] = computer_sale['Sales'] >= 7.5
computer_sale.drop("Sales",axis=1,inplace=True)
computer_sale_str_columns = ['ShelveLoc','Urban','US','Sale_Var']

#convert the string columns from numeric
from sklearn import preprocessing
for i in computer_sale_str_columns:
    number = preprocessing.LabelEncoder()
    computer_sale[i] = number.fit_transform(computer_sale[i])
    
from sklearn.model_selection import train_test_split
train,test = train_test_split(computer_sale,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,1:11],train['Sale_Var'])

np.mean(train['Sale_Var'] == model.predict(train.iloc[:,1:11])) #1

np.mean(test['Sale_Var'] == model.predict(test.iloc[:,1:11])) #0.81

#test accuracy is very low oerfit model
#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")

rf.fit(train.iloc[:,1:11],train['Sale_Var'])

np.mean(train['Sale_Var'] == rf.predict(train.iloc[:,1:11])) #0.99

np.mean(test['Sale_Var'] == rf.predict(test.iloc[:,1:11])) #0.95
