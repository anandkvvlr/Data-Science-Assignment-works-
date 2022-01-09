#################   groceries    ##################

## importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

####### creating groceries data set  ######

# Implementing Apriori algorithm from mlxtend
# conda install mlxtend
# or
# pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

groceries = []
with open("G:\\association rule\\groceries.csv") as f:
    groceries = f.read()

# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

all_groceries_list = [i for item in groceries_list for i in item]

from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_groceries_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

# Creating Data Frame for the transactions data
groceries_series = pd.DataFrame(pd.Series(groceries_list))

###### Null value Treatment  ########
groceries_series.isna().sum()   ## no null values

groceries_series = groceries_series.iloc[:200, :] # selecting only first 200 transactions 

groceries_series.columns = ["transactions"]
groceries_series.head(15)

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X1 = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

############### Data cleaning and Preprocessing begins  ###############

# load the data set
X1.shape
X = X1.copy(deep=True)

###### Null value Treatment  ########
X.isna().sum()   ## no null values

##### summary of the data set ######
X.columns
X.describe()

#######  Outlier Treatment  ########  
# Boxplots 

for i in X.columns:
  sns.boxplot(X[i].dropna())
  plt.show()     ## no outliers because data set is in binary normalized format; & combined boxplot not showig any outliers

#######  standardization//normalization  ####
#data set is already in normalized format so need to do standardization/normalization here  

############     Zero variance analysis     #############
X.shape

##  importing  ###
from sklearn.feature_selection import VarianceThreshold

# Feature selector that removes all low-variance features that meets the variance threshold limit
var_thres = VarianceThreshold(threshold=0) # Threshold is subjective.
var_thres.fit(X)     ###   fit the var_thres to data set X
# Generally we remove the columns with zero variance, but i took thresold value 0 (Near Zero Variance)

var_thres.get_support()     ### it  giving an array out, where zero variant column treat as False value. we already fit var_thres to X. so it gives corresponding information on X
X.columns[var_thres.get_support()]    ##     non-zero variant column names 
constant_columns = [column for column in X.columns if column not in X.columns[var_thres.get_support()]]

print(len(constant_columns))   ### number of zero variant  variables
# since number of zero variant columns = 0  ==> none of the variables or column having zero variant property ; so directy going for further analysis without doing any zero variant treatment


######################## forming  association rules  ######################

frequent_itemsets =   apriori(X, min_support = 0.0075, max_len = 3, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets.head(15)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=15)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)  # first 15 rules according to descending lift values
rules[["antecedents","consequents","support","lift"]].head(10)
