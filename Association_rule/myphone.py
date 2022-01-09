###############  myphonedata.csv  ##############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

# load the data set
df = pd.DataFrame(myphonedatacsv)
df.shape
df1 = df.iloc[:,3:]    ## numeric datas only

###### Null value Treatment  ########
df1.isna().sum()   ## no null values

###### Summary of the data set ####
df1.columns
df1.describe()

#######  Outlier Treatment  ########  
# Boxplots 

for i in df1.columns:
  sns.boxplot(df1[i].dropna())
  plt.show()     ##  no outliers because data set is in binary normalized format; & combined boxplot not showig any outliers

####### checking data is normalised format  or not #######
for i in df1.columns:
    for x in df1[i]:
        if(x!=0 | x!=1):
            standared = "false"
            row=x
            break
        else:
            standared = "true"
            
print(standared)    ## data is already in normalized form since every datas have either 1 or 0 values, so standarded = true
# since data is in normalised format directly going for further analysis

############     Zero variance analysis     #############
df1.shape

##  importing  ###
from sklearn.feature_selection import VarianceThreshold

# Feature selector that removes all low-variance features that meets the variance threshold limit
var_thres = VarianceThreshold(threshold=0) # Threshold is subjective.
var_thres.fit(df1)     ###   fit the var_thres to data set df1
# Generally we remove the columns with zero variance, but i took thresold value 0 (Near Zero Variance)

var_thres.get_support()     ### it  giving an array out, where zero variant column treat as False value. we already fit var_thres to df1. so it gives corresponding information on df1
df1.columns[var_thres.get_support()]    ##     non-zero variant column names 
constant_columns = [column for column in df1.columns if column not in df1.columns[var_thres.get_support()]]

print(len(constant_columns))   ### number of zero variant  variables
# since number of zero variant columns = 0  ==> none of the variables or column having zero variant property ; so directy going for further analysis without doing any zero variant treatment

######################## forming  association rules  ######################

# Implementing Apriori algorithm from mlxtend

# conda install mlxtend
# or
# pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# getting support of each combination using apriori 
frequent_itemsets =   apriori(df1, min_support = 0.075, max_len = 3, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=15)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)
