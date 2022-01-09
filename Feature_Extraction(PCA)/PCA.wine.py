
######## hierarchical   &&  non_hierarchical clustering(k-mean)   ######

import pandas as pd
import matplotlib.pylab as plt

wine2 = pd.read_csv("G:\\pca\\Datasets_PCA\\wine.csv")

wine1 = wine2
wine1.describe()
wine1.info()

###  Here the data set is not a mixed data set. Since all the informative datas are numeric applying standardisation scaling here 

### standardization scaling  ###

#Importing the Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

# define standard scaler
scaler = StandardScaler() # Standard Scaler or Standardization

# Standardized data frame (considering only the input variables columns of data)
df_norm = scaler.fit_transform(wine1.iloc[:, 1:]) #Fit to data, then transform it.
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

wine1['clust'] = cluster_labels # creating a new column and assigning it to new column 

# convert into a dataframe format where first column indicate the cluster details
wine = wine1.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.head()

# Aggregate mean of each cluster
wine1.iloc[:, 2:].groupby(wine1.clust).mean()

# creating a csv file 
wine1.to_csv("wine.csv", encoding = "utf-8")

# save the .csv file in location
import os
os.getcwd()


#####################  non_hierarchical K- means clustering  ###################
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

wine1= wine2

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# standardized data frame (considering the input variable column part of data)
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
wine1['clust'] = mb # creating a  new column and assigning it to new column 

wine1.head()

wine = wine1.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.head()

# Aggregate mean of each cluster
wine1.iloc[:, 2:8].groupby(wine1.clust).mean()

Univ.to_csv("Kmeans_university.csv", encoding = "utf-8")

import os
os.getcwd()

##################### K-mean clustering after applying PCA #####################

import pandas as pd 
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

wine1= wine2

# standardized data set 
df_norm

pca = PCA(n_components = 3)
pca_values = pca.fit_transform(df_norm)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2"
final = pd.concat([wine1.Type, pca_data.iloc[:, 0:3]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final.comp0, y = final.comp1)

## K-means clustering ##

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

wine1= wine2

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# standardized PCA data frame (considering the input variable column part of data)
final 

###### scree plot or elbow curve ############
TWSS = []    # initiate TWSS list
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final)
    TWSS.append(kmeans.inertia_)   ## appending the value of TWSS for each k_value
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(final)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
final['clust'] = mb # creating a  new column and assigning it to new column 

final.head()

final1 = final.iloc[:,[4,0,1,2,3]]
final1.head()

# Aggregate mean of each cluster
final.iloc[:, 1:4].groupby(final.clust).mean()

final1.to_csv("Kmeans_wine.csv", encoding = "utf-8")

import os
os.getcwd()