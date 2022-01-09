 # Multilinear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sklearn

# loading the data
df = pd.read_csv("C://Users//user//Downloads//mlr//ToyotaCorolla.csv",encoding=('ISO-8859-1'),low_memory=False)

#changing column names
df.columns
df = df[['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight']]  # eliminate unneccesary column

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

###### Null value Treatment  ########
df.isna().sum()   
df.dropna(axis = 0, inplace = True)   ## drop na values

# summary
df.columns
df.info()
df.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

## EDA on Dataset -

#Histgram on Profit
sns.distplot(df['Price'],bins=5,kde=True)

# boxplot
#Check any outlier on features having numeric values
import matplotlib.pyplot as plt
df.columns
%matplotlib inline
for i in df.iloc[:,1:]:
    plt.boxplot(df[i],notch=True,patch_artist=True)
    plt.show()
df.iloc[:,0:5].describe()
df.iloc[:,5:10].describe()

# 'Age_08_04', KM ,HP, CC, Gears, Quarterly_Tax, Weight columns have outliers


#######  outlier treatment   ########

#outlier treatment for df.'Age_08_04'

q1 = df['Age_08_04'].quantile(0.25)
q1
q3 = df['Age_08_04'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['Age_08_04'] = np.where(df['Age_08_04']<q1, lower_limit, np.where(df['Age_08_04']>q3, upper_limit, df['Age_08_04']))


#outlier treatment for df.KM

q1 = df['KM'].quantile(0.25)
q1
q3 = df['KM'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['KM'] = np.where(df['KM']<q1, lower_limit, np.where(df['KM']>q3, upper_limit, df['KM']))


#outlier treatment for df.HP

q1 = df['HP'].quantile(0.25)
q1
q3 = df['HP'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['HP'] = np.where(df['HP']<q1, lower_limit, np.where(df['HP']>q3, upper_limit,df['HP']))

#outlier treatment for df.cc

q1 = df['cc'].quantile(0.25)
q1
q3 = df['cc'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['cc'] = np.where(df['cc']<q1, lower_limit, np.where(df['cc']>q3, upper_limit,df['cc']))

#outlier treatment for df.'Gears'

q1 = df['Gears'].quantile(0.25)
q1
q3 = df['Gears'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['Gears'] = np.where(df['Gears']<q1, lower_limit, np.where(df['Gears']>q3, upper_limit, df['Gears']))

#outlier treatment for df.'Quarterly_Tax'

q1 = df['Quarterly_Tax'].quantile(0.25)
q1
q3 = df['Quarterly_Tax'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['Quarterly_Tax'] = np.where(df['Quarterly_Tax']<q1, lower_limit, np.where(df['Quarterly_Tax']>q3, upper_limit, df['Quarterly_Tax']))

#outlier treatment for df.'Weight'

q1 = df['Weight'].quantile(0.25)
q1
q3 = df['Weight'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['Weight'] = np.where(df['Weight']<q1, lower_limit, np.where(df['Weight']>q3, upper_limit, df['Weight']))


# new boxplot after outlier treatment
for i in df.iloc[:,1:]:
    plt.boxplot(df[i],notch=True,patch_artist=True)
    plt.show()

# unique value count
df["Gears"].value_counts() 

# droping Gears column since it have zero variance
df.drop("Gears", axis=1, inplace = True)

######  zero variance operation   ###
df.shape

##  importing  ###
from sklearn.feature_selection import VarianceThreshold

# Feature selector that removes all low-variance features that meets the variance threshold limit
var_thres = VarianceThreshold(threshold=0.2) # Threshold is subjective.
var_thres.fit(df)     ###   fit the var_thres to data set df
# Generally we remove the columns with zero variance, but i took thresold value 0.2 (Near Zero Variance)

var_thres.get_support()     ### it  giving an array out, where zero variant column treat as False value. we already fit var_thres to df1. so it gives corresponding information on df1
df.columns[var_thres.get_support()]    ##     non-zero variant column names 
constant_columns = [column for column in df.columns if column not in df.columns[var_thres.get_support()]]

print(len(constant_columns))   ### number of zero variant  variables

for feature in constant_columns:
    print(feature)               ### names of corresponding zero variant columns : no zero variant columns
# no zero variant columns


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(df.iloc[:, :])
                              
# Correlation matrix 
df.corr()
sns.heatmap(df.corr(), annot=True)
# we see the collinearity between input variables are comparitively less
 
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Quarterly_Tax + Weight', data = df).fit() # regression model
# p_vaue of Doors column is > 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 109 is showing high influence so we can exclude that entire row

df_new = df.drop(df.index[[109]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Quarterly_Tax + Weight', data = df_new).fit() # regression model

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
#Split Dataset into X and y
X=df.drop(columns='Price')
y=df['Price']
pd.DataFrame({'Features':X.columns,'VIF':[ VIF(X.values,i) for i in range(len(X.columns))]}) 

# As Weight is having highest VIF value, we are going to drop this from the prediction model

# new model
ml_nw = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Quarterly_Tax', data = df).fit()
ml_nw .summary() 
# p >0.05 for cc column

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml_nw )
# Studentized Residuals = Residual/standard deviation of residuals
# index 109,111 is showing high influence so we can exclude that entire row

df_new = df.drop(df.index[[109,111]])

# Preparing model                  
ml_nw1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Quarterly_Tax', data = df_new).fit() # regression model

# Summary
ml_nw1.summary()
# p_value is still high for cc

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
#Split Dataset into X and y
X=df.drop(columns='Price')
X1= X.drop(columns='Weight')
y=df['Price']
pd.DataFrame({'Features':X1.columns,'VIF':[ VIF(X1.values,i) for i in range(len(X1.columns))]}) 

# As cc is having highest VIF value, we are going to drop this from the prediction model

# new model
final_ml = smf.ols('Price ~ Age_08_04 + KM + HP +  Doors + Quarterly_Tax', data = df).fit()

# Summary
final_ml.summary()
# p-values < 0.05 for all the features

# Prediction
pred = final_ml.predict(df)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
from scipy import stats
from matplotlib import pylab
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = df.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04 + KM + HP +  Doors + Quarterly_Tax", data = df_train).fit()

# prediction on test data set 
test_pred = model_train.predict(df_test)

# test residual values 
test_resid = test_pred - df_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(df_train)

# train residual values 
train_resid  = train_pred - df_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

