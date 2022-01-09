####### Business Problem To check whether the defective % varies by center or not ####

import pandas as pd
import scipy
from scipy import stats


################ Chi-Square Test ################

order = pd.read_csv("C:/Users/user/Downloads/hypothesis/Datasets_HT/CustomerOrderform.csv",index_col=False)
order.columns

###### Null value Treatment  ########
order.isna().sum()   
order.dropna(axis = 0, inplace = True)   ## drop na values

#summary
order.describe()
order.head()


# x is more than 2 discrete and y is discrete ; Here we will use Chi-square test Chi-Square Test 

################ Chi-Square Test ################
# H0:All are same Ha:atleast 1 are different

order.columns

order['Phillippines'].value_counts()
order['Indonesia'].value_counts()
order['Malta'].value_counts()
order['India'].value_counts()

# creating a cross tab of data
Philli=[271,29]
Indon=[27,33]
Malt=[29,31] 
Ind=[280,20]
Type=["error free","Defective"]

crostab = pd.DataFrame(columns= ['Type','Philli','Indon','Malt','Ind'])

crostab['Type']=pd.Series(Type)
crostab['Philli']=pd.Series(Philli)
crostab['Indon']=pd.Series(Indon)
crostab['Malt']=pd.Series(Malt)
crostab['Ind']=pd.Series(Ind)

crostab

count = crostab.iloc[:,1:]

Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

#P-value=1.75e-33 < 0.05.Hence reject Null.
# hence As per results we can say that all the centers defective percentage are not equal.
