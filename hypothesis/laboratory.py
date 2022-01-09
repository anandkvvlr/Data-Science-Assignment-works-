#### Business Problem Two check whether there is any difference in average TAT ####

import pandas as pd
import scipy
from scipy import stats
from scipy.stats import levene
import numpy as np

############ 2 sample T Test ##################

# Load the data
prom = pd.read_csv("C:/Users/user/Downloads/hypothesis/Datasets_HT/lab_tat_updated.csv")
prom
prom.columns

###### Null value Treatment  ########
prom.isna().sum()   
prom.dropna(axis = 0, inplace = True)   ## drop na values

#summary
prom.describe()
prom.head()

############# Normality Test  ##########

#null hypothesis(h0):Data are normal alternate hypothesis(ha):data are not normal if 
#p-value is > 0.05 => Accept null hypothesis if p-value is < 0.05 =>Reject null hypothesis
# and consider alternative hypothesis

stats.shapiro(prom.Laboratory_1) # Shapiro normality Test for Laboratory_1
# p-value = 0.42 > 0.05 so p high null fly => normal distribution
stats.shapiro(prom.Laboratory_2) # Shapiro normality Test for Laboratory_2
# p-value = 0.86 > 0.05 so p high null fly => normal distribution
stats.shapiro(prom.Laboratory_3) # Shapiro normality Test for Laboratory_3
# p-value = 0.07 > 0.05 so p high null fly => normal distribution
stats.shapiro(prom.Laboratory_4) # Shapiro normality Test for Laboratory_4
# p-value = 0.66 > 0.05 so p high null fly => normal distribution

######### Variance test  #########

#Variance Test H0: All variance are equal Ha: Atleast one variance is different
# All 4 variables are being checked for variances

scipy.stats.levene(prom.Laboratory_1,prom.Laboratory_2,prom.Laboratory_3,prom.Laboratory_4)
# p-value = 0.38 > 0.05 so p high null fly => all variables have Equal variances

#Inputs are 4 lab reports. So Input is Discrete in more than 2 categories.
#Output is continuous as we are trying to see the difference in average TAT.
#we proceed with ANOVA one-way test

############# One - Way Anova ################
from scipy.stats import f_oneway
# One - Way Anova
#Anova Test-One way H0:Average of all laboratory are same 
#Ha:Average of atleast 1 laboratory are different
F, p = stats.f_oneway(prom.Laboratory_1,prom.Laboratory_2,prom.Laboratory_3,prom.Laboratory_4)

# p value
p  # P low Null Go
#P-value is 2.1434e-58 < 0.05= Accept Ha, hence Average of atleast 1 laboratory are different
# As per results we can say that these are not equal i.e Average of atleast 1 laboratory are different
