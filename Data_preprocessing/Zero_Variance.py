
############     Zero variance      #############
import pandas as pd

### import data set "Z_datasetcsv" as dataframe ##
df = pd.DataFrame(Z_datasetcsv)
df1 = df.copy(deep=True)      ### keeping original dataset by creating a copy df1
df1.columns   ## column name

#drop Id Column it doesn't gonna give any data information
df1.drop(['Id'], axis = 1, inplace=True)
df1.dtypes


### here the last column is a string data or character data. So for conducting zero variance checking applying label encoding to get corresponding numeric data
from sklearn.preprocessing import LabelEncoder
## creating instance of label encoder
labelencoder = LabelEncoder()

df1["colour"]= labelencoder.fit_transform(df1["colour"])

######  zero variance operation   ###
df1.shape

##  importing  ###
from sklearn.feature_selection import VarianceThreshold

# Feature selector that removes all low-variance features that meets the variance threshold limit
var_thres = VarianceThreshold(threshold=0.2) # Threshold is subjective.
var_thres.fit(df1)     ###   fit the var_thres to data set df1
# Generally we remove the columns with zero variance, but i took thresold value 0.2 (Near Zero Variance)

var_thres.get_support()     ### it  giving an array out, where zero variant column treat as False value. we already fit var_thres to df1. so it gives corresponding information on df1
df1.columns[var_thres.get_support()]    ##     non-zero variant column names 
constant_columns = [column for column in df1.columns if column not in df1.columns[var_thres.get_support()]]

print(len(constant_columns))   ### number of zero variant  variables

for feature in constant_columns:
    print(feature)               ### names of corresponding zero variant columns

df2 = df1.drop(constant_columns, axis = 1)    ### data set with non-zero variant variables or features
df2.colour = df.colour     ### since colour is not-zero variant column, replace back to its original format
df2["Id"] = df.Id     ### adding column back to data set

###  coclusion  ##
print("final data set with non_zero variance feature: ",df2)
    




