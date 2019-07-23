# Importing Libraries 
import pandas as pd
import numpy as np

'''
Question 1
Implement a KNN model to classify the animals in to categorie
'''

animals = pd.read_csv("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\Classification\\Assignment\\Zoo.csv")
animals.type.value_counts()

#coverting the string values into numeric 
animals.drop(["animal name"],axis=1,inplace=True)

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(animals,test_size = 0.2) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)

# Fitting with training data 
neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16]) # 94 %

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16]) # 100%
test["pred"] = neigh.predict(test.iloc[:,0:16])
# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
    acc.append([train_acc,test_acc])

acc
'''
[[0.9625, 0.9523809523809523],
 [0.925, 0.9047619047619048],
 [0.9, 0.9047619047619048],
 [0.8125, 0.9047619047619048],
 [0.775, 0.8571428571428571],
 [0.75, 0.9047619047619048],
 [0.75, 0.9047619047619048],
 [0.75, 0.9047619047619048],
 [0.75, 0.9047619047619048],
 [0.75, 0.9047619047619048],
 [0.7375, 0.9047619047619048],
 [0.6625, 0.8571428571428571],
 [0.6625, 0.8571428571428571],
 [0.6625, 0.8571428571428571],
 [0.6625, 0.8571428571428571],
 [0.525, 0.7619047619047619],
 [0.525, 0.7619047619047619],
 [0.5, 0.7619047619047619],
 [0.4875, 0.7142857142857143],
 [0.4625, 0.6666666666666666],
 [0.5, 0.6666666666666666],
 [0.4625, 0.6190476190476191],
 [0.425, 0.5714285714285714],
 [0.425, 0.47619047619047616]]
'''

# k = 3 is giving the best acuracy
import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-");plt.legend(["train","test"])

'''
Question 2
Prepare a model for glass classification using KNN
'''
glass = pd.read_csv("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\Classification\\Assignment\\glass.csv")

# Training and Test data using 
from sklearn.model_selection import train_test_split
train_glass,test_glass = train_test_split(glass,test_size = 0.2) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh_glass = KNC(n_neighbors= 3)
neigh_glass.fit(glass.iloc[:,0:9],glass.iloc[:,9])

train_glass_acc = np.mean(neigh_glass.predict(train_glass.iloc[:,0:9]) == train_glass.iloc[:,9])

test_glass_acc = np.mean(neigh_glass.predict(test_glass.iloc[:,0:9]) == test_glass.iloc[:,9])

glass_pred = []

for i in range(3,50,2):
    neigh_glass = KNC(n_neighbors= i)
    neigh_glass.fit(glass.iloc[:,0:9],glass.iloc[:,9])
    train_glass_acc = np.mean(neigh_glass.predict(train_glass.iloc[:,0:9]) == train_glass.iloc[:,9])
    test_glass_acc = np.mean(neigh_glass.predict(test_glass.iloc[:,0:9]) == test_glass.iloc[:,9])
    glass_pred.append([train_glass_acc,test_glass_acc])
    
#k = 3 is giving the best accuracy
'''
[[0.8362573099415205, 0.8372093023255814],
 [0.7894736842105263, 0.6511627906976745],
 [0.7485380116959064, 0.6744186046511628],
 [0.7134502923976608, 0.6511627906976745],
 [0.695906432748538, 0.5581395348837209],
 [0.6783625730994152, 0.4883720930232558],
 [0.6783625730994152, 0.5348837209302325],
 [0.695906432748538, 0.5581395348837209],
 [0.6900584795321637, 0.5581395348837209],
 [0.6842105263157895, 0.5813953488372093],
 [0.6900584795321637, 0.6046511627906976],
 [0.6783625730994152, 0.5348837209302325],
 [0.6608187134502924, 0.5116279069767442],
 [0.6608187134502924, 0.5116279069767442],
 [0.6549707602339181, 0.5581395348837209],
 [0.6491228070175439, 0.5581395348837209],
 [0.6374269005847953, 0.5348837209302325],
 [0.6374269005847953, 0.5348837209302325],
 [0.6374269005847953, 0.5348837209302325],
 [0.6432748538011696, 0.5348837209302325],
 [0.6549707602339181, 0.5116279069767442],
 [0.6549707602339181, 0.5116279069767442],
 [0.6432748538011696, 0.5116279069767442],
 [0.6374269005847953, 0.5116279069767442]]
'''

# k = 3 is giving the best acuracy
import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in glass_pred],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in glass_pred],"ro-");plt.legend(["train","test"])
