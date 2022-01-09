import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sn

# load the data set
zoo = pd.read_csv("C://Users//user//Downloads//knn//Zoo.csv")
zoo.shape
zoo1 = zoo.copy(deep=True)

###### Null value Treatment  ########
zoo1.isna().sum()   ## no null values

zoo1 = zoo1.iloc[:, 1:] # Excluding animal name column ==> less informative

###### Summary of the data set ####
zoo1.columns
zoo1.describe()

#######  Outlier Treatment  ########  
# Boxplots 
zoo_bx_input = zoo1.iloc[:, 0:16]  ## except output column
for i in  zoo_bx_input.columns:
  sns.boxplot(zoo_bx_input[i].dropna())
  plt.show()     ## combined boxplot not showig any outliers
       
############     Zero variance analysis     #############
zoo1.shape

##  importing  ###
from sklearn.feature_selection import VarianceThreshold

# Feature selector that removes all low-variance features that meets the variance threshold limit
var_thres = VarianceThreshold(threshold=0) # Threshold is subjective.
var_thres.fit(zoo1)     ###   fit the var_thres to data set 
# Generally we remove the columns with zero variance, but i took thresold value 0 (Near Zero Variance)

var_thres.get_support()     ### it  giving an array out, where zero variant column treat as False value. we already fit var_thres to data set. so it gives corresponding information on data set
zoo1.columns[var_thres.get_support()]    ##     non-zero variant column names 
constant_columns = [column for column in zoo1.columns if column not in zoo1.columns[var_thres.get_support()]]

print(len(constant_columns))   ### number of zero variant  variables
# since number of zero variant columns = 0  ==> none of the variables or column having zero variant property ; so directy going for further analysis without doing any zero variant treatment


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (except output)
wbcd_n = norm_func(zoo1.iloc[:, 0:16])
wbcd_n.describe()

# univariate & byvariate plot after normalisation
sn.pairplot(zoo1.iloc[:, 0:16])

# Differentiate predictors (input variables) & targeters(output variable)
X = np.array(wbcd_n.iloc[:,:]) # Predictors 
Y = np.array(zoo1['type']) # Target 

# train & test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)  #consider 5 nearest neighbors 
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
print(classification_report(Y_test, pred))
print(confusion_matrix(Y_test, pred))

# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
print(confusion_matrix(Y_train, pred_train))

# creating empty list variable 
acc = []
train_ac_value = []
test_ac_value = []
k_value = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])
    train_ac_value.append(train_acc) 
    test_ac_value.append(test_acc)
    k_value.append(i)

# accuracy table for different k value
accuracy_table = pd.DataFrame(columns=["k_value","train_score", 'test_score',"difference_square"])
difference=[]
error=[]
l = len(acc)
for i in range(l):
    difference = train_ac_value[i]- test_ac_value[i]
    error.append(difference*difference)  ## to get the minimum value of error in magnitude by eliminating -ve sign
    
accuracy_table.k_value = pd.Series(k_value)
accuracy_table.train_score = pd.Series(train_ac_value)
accuracy_table.test_score = pd.Series(test_ac_value) 
accuracy_table.difference_square = pd.Series(error)
accuracy_table.head(6) 
accuracy_table.describe()   ## k = 3 for least difference in train & test accuracy

# graphical representaion of train & test accuracy with different k value
import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")

# model for k = 3
knn = KNeighborsClassifier(n_neighbors = 3)  #consider 3 nearest neighbors 
knn.fit(X_train, Y_train)
pred = knn.predict(X_test)
print(classification_report(Y_test, pred))   ## test acc= .95
pred_train = knn.predict(X_train)   # evaluation on train data
print(classification_report(Y_train, pred_train))   ## test acc= .96 ==> least acc diiference

### checking accuracy while applying cross validation
from sklearn.model_selection import cross_val_score
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify= Y, test_size = 0.2)  ## proportion of output labels would be sames for each split
cross_val_score(KNeighborsClassifier(n_neighbors=(3)), X_train, Y_train, cv=4)                             
cross_val_score(KNeighborsClassifier(n_neighbors=(3)), X_train, Y_train,cv=15).mean()   # mean score                        

