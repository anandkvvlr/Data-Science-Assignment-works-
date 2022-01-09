import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer as si

###  import dataset Seeds_data as dataframe  ##
df = pd.DataFrame(Seeds_datacsv)
df1 = df.copy(deep=True)

###  Here the data set is not a mixed data set. Since all the datas are numeric for better visualisation normalization Scaling not suits here. Both normalization & standardization will work here. But since the output column "TYPE" is a ordinal data, thats even in the range 1 to 3 standardization scaling will gives visually better scaled output data set. So gonna apply standardisation scaling here 

### standardization scaling  ###

df.describe().T

#Importing the Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

# define standard scaler
scaler = StandardScaler() # Standard Scaler or Standardization

# Transform data  ## "TYPE" column is considered as ouput. so not gonna do any scaling there
df1.iloc[:,0:7] = scaler.fit_transform(df1.iloc[:,0:7]) #Fit to data, then transform it.
print("Standardized Scaler :\n",df1)
df1.describe().T   
### mean of each variable approximately = 0; stndered deviation of each variable approximately = 1. so satisfying standardization criteria































































































































