# Multilinear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sklearn
import statsmodels.api as sm

# loading the data
df = pd.read_csv("C://Users//user//Downloads//mlr//Computer_Data.csv")

#changing column names
df.columns
df = df.iloc[:,1:]  # eliminate unneccesary column

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


# Encoding categorical data
#converting into numerical
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['cd'] = lb.fit_transform(df['cd'])
df['multi'] = lb.fit_transform(df['multi'])
df['premium'] = lb.fit_transform(df['premium'])


#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

## EDA on Dataset -

#Histgram on Profit
sns.distplot(df['price'],bins=5,kde=True)

# boxplot
#Check any outlier on features having numeric values
import matplotlib.pyplot as plt
%matplotlib inline
X= df[['speed', 'hd', 'ram', 'screen','ads', 'trend']]
for i in X.iloc[:,0:]:
    plt.boxplot(df[i],notch=True,patch_artist=True)
    plt.show()
df.iloc[:,0:5].describe()
df.iloc[:,5:].describe()
# screen,ram,hd columns datas have outliers

#outlier treatment for df.hd

q1 = df['hd'].quantile(0.25)
q1
q3 = df['hd'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['hd'] = np.where(df['hd']<q1, lower_limit, np.where(df['hd']>q3, upper_limit, df['hd']))


#outlier treatment for df.ram

q1 = df['ram'].quantile(0.25)
q1
q3 = df['ram'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['ram'] = np.where(df['ram']<q1, lower_limit, np.where(df['ram']>q3, upper_limit, df['ram']))


#outlier treatment for df.screen

q1 = df['screen'].quantile(0.25)
q1
q3 = df['screen'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#Replacing by pulling outliers to lower and upper limit
df['screen'] = np.where(df['screen']<q1, lower_limit, np.where(df['screen']>q3, upper_limit,df['screen']))

# new boxplot after outlier treatment
for i in X.iloc[:,0:]:
    plt.boxplot(df[i],notch=True,patch_artist=True)
    plt.show()


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(df.iloc[:, :])
                              
# Correlation matrix 
df.corr()
sns.heatmap(df.corr(), annot=True)
# we see the collinearity between input variables are comparitively less

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
final_ml = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = df).fit() # regression model

# Summary
final_ml.summary()
# p-values < 0.05 for all the features

# Prediction
pred = final_ml.predict(df)

# Q-Q plot
import statsmodels.api as sm
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
from scipy import stats
from matplotlib import pylab
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = df.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend", data = df_train).fit()

# prediction on test data set 
test_pred = model_train.predict(df_test)

# test residual values 
test_resid = test_pred - df_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(df_train)

# train residual values 
train_resid  = train_pred - df_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

