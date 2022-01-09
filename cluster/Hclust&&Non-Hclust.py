
######## hierarchical   &&  non_hierarchical clustering(k-mean)   ######

import pandas as pd
import matplotlib.pylab as plt

Univ2 = pd.read_excel("G:\\cluster\\hierarchial_clus\\University_Clustering.xlsx\\University_Clustering.xlsx")

Univ1 = Univ2 # copying

Univ1.describe()
Univ1.info()

Univ = Univ1.drop(["State"], axis=1)

###  Here the data set is not a mixed data set. Since all the informative datas are numeric applying standardisation scaling here 

### standardization scaling  ###

#Importing the Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

# define standard scaler
scaler = StandardScaler() # Standard Scaler or Standardization

# Standardized data frame (considering the numerical part of data)
df_norm = scaler.fit_transform(Univ.iloc[:, 1:]) #Fit to data, then transform it.
print("Standardized Scaler :\n",df_norm)

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

# convert the from array format to pandas series object
cluster_labels = pd.Series(h_complete.labels_)

Univ['clust'] = cluster_labels # creating a new column and assigning it to new column 

# convert into a dataframe format where first column indicate the cluster details
Univ1 = Univ.iloc[:, [7,0,1,2,3,4,5,6]]
Univ1.head()

# Aggregate mean of each cluster
Univ1.iloc[:, 2:].groupby(Univ1.clust).mean()

# creating a csv file 
Univ1.to_csv("University.csv", encoding = "utf-8")

# save the .csv file in location
import os
os.getcwd()



#####################  non_hierarchical K- means clustering  ###################
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# standardized data frame (considering the numerical part of data)
df_norm 

###### scree plot or elbow curve ############
TWSS = []    # initiate TWSS list
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)   ## appending the value of TWSS for each k_value
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Univ['clust'] = mb # creating a  new column and assigning it to new column 

Univ.head()
df_norm.head()

Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ.head()

# Aggregate mean of each cluster
Univ.iloc[:, 2:8].groupby(Univ.clust).mean()

Univ.to_csv("Kmeans_university.csv", encoding = "utf-8")

import os
os.getcwd()

