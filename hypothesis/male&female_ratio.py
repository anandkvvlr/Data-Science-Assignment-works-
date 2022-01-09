#####  Business Problem Two find buyer ratios are similar across region or not  ###

import pandas as pd
import scipy
from scipy import stats

################ Chi-Square Test ################

sales = pd.read_csv("C:/Users/user/Downloads/hypothesis/Datasets_HT/BuyerRatio.csv",index_col=False)
sales.columns

###### Null value Treatment  ########
sales.isna().sum()   
sales.dropna(axis = 0, inplace = True)   ## drop na values

#summary
sales.describe()
sales.head()

############# Normality Test  ##########

#null hypothesis(h0):Data are normal alternate hypothesis(ha):data are not normal if 
#p-value is > 0.05 => Accept null hypothesis if p-value is < 0.05 =>Reject null hypothesis
# and consider alternative hypothesis

# stats.shapiro(sales.iloc[0,1:]) # Shapiro normality Test for East
# p-value = 0.42 > 0.05 so p high null fly => normal distribution
# stats.shapiro(sales.iloc[1,1:]) # Shapiro normality Test for West
# p-value = 0.86 > 0.05 so p high null fly => normal distribution


#Inputs are 4 discrete variables(east,west,north,south).
#Output is also discrete. We are trying to find out if proportions of male and female are 
#similar or not across the regions
# so proceed with chi-square test

################ Chi-Square Test ################
#Ho= Proportions of Male and Female are same
#Ha= Proportions of Male and Female are not same

# defining the table
array1 = sales.iloc[0,1:].values
array2 = sales.iloc[1,1:].values

count = [[array1], [array2]]

Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

#P-value=0.66 > 0.05.Hence we fail to reject Null.
# As per results we can say that there is proportion of male and female buying is similar
