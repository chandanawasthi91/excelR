#Hierarchical assigmnet
import pandas as pd
import matplotlib.pylab as plt 
'''
Perform clustering (Both hierarchical and K means clustering) for the airlines data to obtain optimum number of clusters. 
Draw the inferences from the clusters obtained.
'''

airline = pd.read_excel("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\Regression\\EastWestAirlines.xlsx",sheet_name="data")
airline.columns

#deleting the insignificant columns
airline_1 = airline.drop(["ID#"],axis=1)
airline = airline_1

# alternative normalization function 

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airline.iloc[:,:])

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

#p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(25, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=5,	linkage='complete',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)

airline['clust']=cluster_labels # creating a  new column and assigning it to new column 

airline = airline.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
airline['clust'].value_counts()
# getting aggregate mean of each cluster
airline.iloc[:,2:].groupby(airline['clust']).median()

#Kmeans
from sklearn.cluster import KMeans

K = range(1,15)
Sum_of_squared_distances = []
for i in K:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    kmeans.labels_
    Sum_of_squared_distances.append(kmeans.inertia_)

#elbow curve
#As k increases, the sum of squared distance tends to zero from K = 3 sharp bend
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')

airline['clust'].value_counts()


'''
Perform Clustering for the crime data and identify the number of clusters formed and draw inferences.
'''

crime = pd.read_csv("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\Regression\\crime_data.csv")
crime.columns

crime_data = crime.iloc[:,1:5]

# alternative normalization function 

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm_crime_data = norm_func(crime_data)

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm_crime_data, method="complete",metric="euclidean")

plt.figure(figsize=(25, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(df_norm_crime_data) 

crime['Clust']=pd.Series(h_complete.labels_)
crime['Clust'].value_counts()

# getting aggregate mean of each cluster
crime.groupby(crime['Clust']).median()


#Kmeans
from sklearn.cluster import KMeans

K = range(1,15)
Sum_of_squared_distances = []
for i in K:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm_crime_data)
    kmeans.labels_
    Sum_of_squared_distances.append(kmeans.inertia_)

#elbow curve
#As k increases, the sum of squared distance tends to zero from K = 3 sharp bend
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#4 is the optimal cluster as sharp bend in curve after 4