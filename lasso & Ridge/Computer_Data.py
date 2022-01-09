
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

# loading the data
df = pd.read_csv("C:/Users/user/Downloads/lasso ridge/Computer_Data (1).csv")

#creating one copy
df1=df.copy(deep=True)

###### Null value Treatment  ########
df1.isna().sum()   
df1.dropna(axis = 0, inplace = True)   ## drop na values

df1.info()
df1.columns

#summary
df1.describe()

# converting ouput variable to numeric binary dummy variable format
lb = LabelEncoder()
df1['cd'] =lb.fit_transform(df1['cd'])
df1['multi'] =lb.fit_transform(df1['multi'])
df1['premium'] =lb.fit_transform(df1['premium'])    

# normalisation
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

# define standard scaler
scaler1 = MinMaxScaler()  # MinMax Scaler or Normalization

# Transform data ; except index& out column
df1.iloc[:,2:] = scaler1.fit_transform(df1.iloc[:,2:]) #Fit to data, then transform it.

# converting back to dataframe
df1 = pd.DataFrame(df1)

df.columns
df1.columns = 'index', 'price', 'speed', 'hd', 'ram', 'screen', 'cd', 'multi','premium', 'ads', 'trend'

# droping index column for easy operation
df1.drop('index', axis=1, inplace = True)
df1.head()    
df1.describe()

# Correlation matrix 
a = df1.corr()
a
# EDA
a1 = df1.describe()

# Scatter plot and histogram between variables
sns.pairplot(df1) # not having any multicolinearity issue between independant input variables

#train test split
from sklearn.model_selection import train_test_split

train, test = train_test_split(df1, test_size = 0.2, random_state=42)

# Preparing the model on train data 
model = smf.ols("price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend", data = train).fit()
model.summary()

#evaluation on test data

# Prediction 
pred_test = model.predict(test)
# Error
resid_test  = pred_test - test.price
# RMSE value for data 
rmse_test = np.sqrt(np.mean(resid_test * resid_test))
rmse_test   # 283.43 

#evaluation on train data

# Prediction 
pred_train = model.predict(train)
# Error
resid_train  = pred_train - train.price
# RMSE value for data 
rmse_train = np.sqrt(np.mean(resid_train * resid_train))
rmse_train  # 273.15


# To overcome the issues(reduce error value OR over fit problem), LASSO and RIDGE regression are used

######## LASSO REGRESSION MODEL ##########
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso()

parameters_l = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters_l, scoring = 'r2', cv = 5)
lasso_reg.fit(train.iloc[:, 1:], train.price)

lasso_reg.best_params_   # 0.01
lasso_reg.best_score_   #  0.78

lasso_pred_train = lasso_reg.predict(train.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(train.iloc[:, 1:], train.price)   # 0.78

# RMSE
np.sqrt(np.mean((lasso_pred_train - train.price)**2))  # 273.15

# lasso model for best alpha value
lasso = Lasso(alpha = 0.01, normalize = True)

# fit the train data with best alpha lasso model
lasso.fit(train.iloc[:, 1:], train.price) 

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(train.columns[1:]))

###LASSO Evaluation on Test Data###

# Prediction 
pred_lasso_test = lasso.predict(test.iloc[:, 1:])
# Error
resid_test  = pred_lasso_test - test.price

# RMSE value for data 
rmse_lasso_test = np.sqrt(np.mean(resid_test * resid_test))
rmse_lasso_test    # 283.32

# Test data score in adusted r_square term
lasso_reg.score(test.iloc[:, 1:], test.price)   # 0.75


####### RIDGE REGRESSION MODEL #######

from sklearn.linear_model import Ridge
help(Ridge)

ridge = Ridge()

# since in ridge have no problem of coeficient elimination we are going for a wide search of lambda value(tuning parameter)

p = []
x = 1e-320
for l in range(1,9,1):
    if(x > 1e-50):
        break
    else:
        p.append(x)
        x *= 1e+40
    l=l+1
for i in range(1, 23, 1):
    p.append(x)
    x = x * 100
    if(x > 1):
        break
    i = i+1
for c in range(1,50,4):
    p.append(c)
            
p        
parameters = {'alpha': p}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(train.iloc[:, 1:], train.price)

ridge_reg.best_params_   # 0.009999888671826829
ridge_reg.best_score_   #  0.78

ridge_pred_train = ridge_reg.predict(train.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(train.iloc[:, 1:], train.price)   # 0.78

# RMSE
np.sqrt(np.mean((ridge_pred_train - train.price)**2))  # 273.15

# ridge model for best alpha value
ridge = Ridge(alpha = 0.009999888671826829, normalize = True)

# fit the train data with best alpha ridge model
ridge.fit(train.iloc[:, 1:], train.price) 

# Coefficient values for all independent variables#
ridge.coef_
ridge.intercept_

plt.bar(height = pd.Series(ridge.coef_), x = pd.Series(train.columns[1:]))

###RIDGE Evaluation on Test Data###

# Prediction 
pred_ridge_test = ridge.predict(test.iloc[:, 1:])

# Error
resid_test  = pred_ridge_test - test.price

# RMSE value for data 
rmse_ridge_test = np.sqrt(np.mean(resid_test * resid_test))
rmse_ridge_test  # 283.38

# Test data score in adusted r_square term
ridge_reg.score(test.iloc[:, 1:], test.price)  #  0.75


########## ELASTIC NET REGRESSION MODEL ##########

from sklearn.linear_model import ElasticNet
help(ElasticNet)

enet = ElasticNet()

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(train.iloc[:, 1:], train.price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred_train = enet_reg.predict(train.iloc[:, 1:])

# Adjusted r-square#
enet_reg.score(train.iloc[:, 1:], train.price)

# RMSE
np.sqrt(np.mean((enet_pred_train - train.price)**2))

# ElasticNet model for best alpha value
enet = ElasticNet(alpha = 9.999888671826829e-07, normalize = True)

# fit the train data with best alpha ElasticNet model
enet.fit(train.iloc[:, 1:], train.price) 

# Coefficient values for all independent variables#
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(train.columns[1:]))

###ElasticNet Evaluation on Test Data###

# Prediction 
pred_enet_test = enet.predict(test.iloc[:, 1:])

# Error
resid_test  = pred_enet_test - test.price

# RMSE value for data 
rmse_enet_test = np.sqrt(np.mean(resid_test * resid_test))
rmse_enet_test  

# Test data score in adusted r_square term
enet_reg.score(test.iloc[:, 1:], test.price)




