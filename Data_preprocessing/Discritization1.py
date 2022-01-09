
########################    Discretization    ####################     

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer as si

### import data set "iris" as dataframe  ###
a_main = pd.DataFrame(iriscsv)
d = a_main.copy(deep=True)

## Treat species as output column. so not gonna do discretization technique  up on that perticular column

## sepal length
d.iloc[:,1].describe()
bins=[4,5,6,7,8]           ### [4,5],(5,6],(6,7],(7,8]
group_names= ['[4,5]','(5,6]','(6,7]','(7,8]']    ## (a,b] => a not included; but b included 
d.iloc[:,1]= pd.cut(d.iloc[:,1],bins, labels = group_names)

## sepal width
d.iloc[:,2].describe()
bins=[2,3,4,5]     ## [2,3],(3,4],(4,5]
group_names= ['[2,3]','(3,4]','(4,5]']     ## (a,b] => a not included; but b included 
d.iloc[:,2]= pd.cut(d.iloc[:,2],bins, labels = group_names)

## petal length
d.iloc[:,3].describe()
bins=[1,3,5,7]     ## [1,3],(3,5],(5,7]
group_names= ['[1,3]','(3,5]','(5,7]']     ## (a,b] => a not included; but b included 
d.iloc[:,3]= pd.cut(d.iloc[:,3],bins, labels = group_names)

## petal width
d.iloc[:,4].describe()
bins=[0,1,2,3]     ## [0,1],(1,2],(2,3]
group_names= ['[0,1]','(1,2]','(2,3]']     ## (a,b] => a not included; but b included 
d.iloc[:,4]= pd.cut(d.iloc[:,4],bins, labels = group_names)

## conclusion  ##

### discretized data set
print("discretized data set is:",d)
