#Business Problem To find proportion male vs female differ from weekdays or weekends are equal or not 

import pandas as pd
import scipy
from scipy import stats

#Inputs are 2 discrete variables & Output is Discrete as we are trying to find out if proportions of
# male and female walking in to the store is same or not
#We proceed with 2-proportion test

# Load the data
two_prop_test = pd.read_csv("C:/Users/user/Downloads/hypothesis/Datasets_HT/Fantaloons.csv")
two_prop_test.columns

######### 2-proportion test ###########
import numpy as np

from statsmodels.stats.proportion import proportions_ztest

two_prop_test['Weekdays'].value_counts()
two_prop_test['Weekend'].value_counts()

# creating a cross table of data
wk_day=[287,113]
wk_end_day=[233,167]
gender = ["female","male"]

crostab = pd.DataFrame(columns= ['gender','wk_day','wk_end_day'])

crostab['gender']=pd.Series(gender)
crostab['wk_day']=pd.Series(wk_day)
crostab['wk_end_day']=pd.Series(wk_end_day)

crostab

count = np.array([113,167])
nobs = np.array([400, 400])

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(pval) # Pvalue = 6.26e-05 < 0.05, hence we fail to reject Null.Hence proportions of 
#Male and Female are not same on week and week end days

#Now we will try to find out whose proportion is higher in accordance of days. We create another hypothesis
#Ho= Proportions of Male is same on week days and week end days
#Ha= Proportions of Male is for week days is lower than that of week end days

stats, pval = proportions_ztest(count, nobs, alternative = 'smaller')
print(pval)  # Pvalue 3.13e-05 < 0.05, Reject Null hypothesis(H0), 
## hence As per results we can say that Proportions of Male is for week days is lower than 
#that of week end days

