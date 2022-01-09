########### book ##############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

# load the data set
book = pd.DataFrame(bookcsv)
book.shape
book1 = book.copy(deep=True)

###### Null value Treatment  ########
book1.isna().sum()   ## no null values

###### Summary of the data set ####
book1.columns
book1.describe()

#######  Outlier Treatment  ########  
# Boxplots 

for i in book1.columns:
  sns.boxplot(book1[i].dropna())
  plt.show()     ## no outliers because data set is in binary normalized format; & combined boxplot not showig any outliers

####### checking data is normalisedformat  or not #######
###for i in book1.columns:
###    for x in book1[i]:   #################  Wrong
 #       if(x!=0 | x!=1):  
  #          standared = "false"
   #         row=x
    #        break      #################  Wrong
     #   else:
      #      standared = "true"
            
print(standared)    ## data is already in normalized form since every datas have either 1 or 0 values, so standarded = true
# since data is in normalised format directly going for further analysis

############     Zero variance analysis     #############
book1.shape

##  importing  ###
from sklearn.feature_selection import VarianceThreshold

# Feature selector that removes all low-variance features that meets the variance threshold limit
var_thres = VarianceThreshold(threshold=0) # Threshold is subjective.
var_thres.fit(book1)     ###   fit the var_thres to data set book1
# Generally we remove the columns with zero variance, but i took thresold value 0 (Near Zero Variance)

var_thres.get_support()     ### it  giving an array out, where zero variant column treat as False value. we already fit var_thres to book1. so it gives corresponding information on book1
book1.columns[var_thres.get_support()]    ##     non-zero variant column names 
constant_columns = [column for column in book1.columns if column not in book1.columns[var_thres.get_support()]]

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
frequent_itemsets =   apriori(book1, min_support = 0.08, max_len = 3, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=15)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(15)  # first 15 rules
rules.sort_values('lift', ascending = False).head(10)  # first 15 rules according to descending lift values
