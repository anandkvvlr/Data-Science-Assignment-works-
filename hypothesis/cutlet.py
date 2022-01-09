#### Business Problem Two check whether the diameter of two units are similar or not  ####

import pandas as pd
import scipy
from scipy import stats   # from scipy..stats import shapiro  

############ 2 sample T Test ##################

# Load the data
prom = pd.read_csv("C:/Users/user/Downloads/hypothesis/Datasets_HT/Cutlets.csv")
prom
prom.columns

###### Null value Treatment  ########
prom.isna().sum()   
prom.dropna(axis = 0, inplace = True)   ## drop na values

#summary
prom.describe()
prom.head()

x= prom['Unit A']
stats.shapiro(x)

##### Normality Test  #####

#null hypothesis(H0):Data are normal alternate hypothesis(Ha):data are not normal if p-value 
#is > 0.05 => Accept null hypothesis if p-value is < 0.05 =>Reject null hypothesisand go for Alternate hypothesis

help(stats.shapiro)
A= prom['Unit A']
stats.shapiro(A) # Shapiro normality Test
# p-value = 0.31 > 0.05 so p high null fly => normal distribution


B = prom['Unit B']
stats.shapiro(B)
# p-value = 0.52 > 0.05 so p high null fly => normal distribution

#### Variance test ####

#Variance Test H0: variance of unitA = variance of unitB Ha: variance of unitA NOT= variance of unitB
help(scipy.stats.levene)
scipy.stats.levene(prom['Unit A'], prom['Unit B'])
# p-value = 0.418 > 0.05 so p high null fly => Equal variances

#Assume Null hyposthesis as Ho: μ1 = μ2 (There is no difference in diameters of cutlets between two units)
# Thus Alternate hypothesis as Ha: μ1 ≠ μ2 (There is significant difference in diameters of cutlets
# between two units) 2 Sample 2 Tail test applicable

#(((As A and B are 2 Discrete variables and output variable diameter is a continuous,
#   we will go with 2-sample T test)))

# 2 Sample T test  
scipy.stats.ttest_ind(prom['Unit A'], prom['Unit B'])
help(scipy.stats.ttest_ind)

#You can see the detailed summary displayed.
## compare p_value with α = 0.05 (At 5% significance level)
#inference: As (p_value=0.4722) > (α = 0.05); Accept Null Hypothesis i.e. μ1 = μ2
#Inference is that there is no significant difference in the diameters of Unit A and Unit B
