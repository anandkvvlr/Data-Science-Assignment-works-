
####################  Dummy variable  ##################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer as si

## import data set animal_category as Data Frame   ####
df = pd.DataFrame(animal_categorycsv)

### creating copy and keep original data safe
d = df.copy(deep=True)
df1 = df.copy(deep=True)

df1.shape

### Here we considered "Type" column as output variable, so eliminate that perticular column from analysis
df1.drop(['Types'], axis = 1, inplace=True)

df1.dtypes   ## Data type of each column

### counts of each unique category
df1["Homly"].value_counts()
df1["Animals"].value_counts()
df1["Gender"].value_counts()

###  Here every independed variables "Homly","Animals & "Gender" are nominal datas. So there have not any order for them. Son instead of label encoding we are gonna apply "One Hot Encoding" here.

from sklearn.preprocessing import OneHotEncoder
#creating  OneHotEncoder 
OHE = OneHotEncoder(sparse = False, drop = 'first').fit(df1)    ## initializing OneHotEncoder as OHE ###  sparse = False=>Will return sparse matrix form output  ### drop = first=> drop first column each dummy variables after One Hot Encoding
df2 = OHE.fit_transform(df1[['Animals','Gender','Homly']])
df3=pd.DataFrame(df2,columns=[ 'Animals_Dog', 'Animals_Gout', 'Animals_Lion', 'Animals_Mouse', 'Gender_Male','Homly_Yes'])

### complete Data set
d4= pd.DataFrame(d[["Index","Types"]])
d_complete = pd.concat([d4,df3],axis=1)
