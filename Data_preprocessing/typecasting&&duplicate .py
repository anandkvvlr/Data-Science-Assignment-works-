################## Type casting  & duplicate treatment###############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer as si

### import data set OnlineRetail  ###
df = pd.DataFrame(OnlineRetailcsv)
d = df.copy(deep=True)
df1 = df.copy(deep=True)

df1.dtypes
### UNitPrice & CustomerID columns datas are float64 type. so will convert those to int64    

### remove the perticular rows having NA or missins values. Because type casting is not possible on NA datas
df1.isna()    ### missing values treat as True
df1.isna().sum()   ### total number of missing values  ## df1.notna()=>Treat missing values as False
df1.shape     ### shape of the dataframe
df2 = pd.DataFrame(df1.dropna(how='any'))  ### drop the row datas contain missing values  ;  how = any => if any of the row data is NA then complete row will get eleminate ; shape => showing shape
### for subset or column name mentioned operation => df5 = pd.DataFrame(df1.dropna(subset=["Description","CustomerID"],how='any'))
df2.shape

#type casting
# Now we will convert 'float64' into 'int64' type. 
df2.UnitPrice = df2.UnitPrice.astype('int64') 
df2.CustomerID = df2.CustomerID.astype('int64')
df2.dtypes

#Identify duplicates records in the data
duplicate = df2.duplicated()
duplicate
sum(duplicate)

#Removing Duplicates
df2_dup = df2.drop_duplicates() 
df2_dup

## conclusion
print("the data set without any duplicates:", df2_dup)

