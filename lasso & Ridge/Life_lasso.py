
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

# loading the data
df = pd.read_csv("C:/Users/user/Downloads/lasso ridge/Life_expectencey_LR.csv")

#creating one copy
df1=df.copy(deep=True)
df1.columns

# rearranging the columns
df1 = df1[['Life_expectancy','Country','Status', 'Year','Adult_Mortality',
       'infant_deaths', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B',
       'Measles', 'BMI', 'under_five_deaths', 'Polio', 'Total_expenditure',
       'Diphtheria', 'HIV_AIDS', 'GDP', 'Population', 'thinness',
       'thinness_yr', 'Income_composition', 'Schooling']]

###### Null value Treatment  ########
df1.isna().sum()   
df1.dropna(axis = 0, inplace = True)   ## drop na values

df1.info()
df1.columns

#summary
df1.describe()   

# converting ouput variable to numeric binary dummy variable format
lb = LabelEncoder()
df1['Country'] =lb.fit_transform(df1['Country'])
df1['Status'] =lb.fit_transform(df1['Status'])

### standardization scaling  ###

#Importing the Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

# define standard scaler
scaler = StandardScaler() # Standard Scaler or Standardization

# Transform data  ## "TYPE" column is considered as ouput. so not gonna do any scaling there
df1.iloc[:,1:] = scaler.fit_transform(df1.iloc[:,1:]) #Fit to data, then transform it.

df1.describe()
df1.columns
df1.head()    
  

# Correlation matrix 
a = df1.corr()
a

# Scatter plot and histogram between variables
sns.pairplot(df1) # having any multicolinearity issue between GDP & 'percentage_expenditure','thinness'&
 #      'thinness_yr' , 'infant_deaths'&'under_five_deaths'independant input variables 
 # so actually we have to treat and eliminate the corresponding problem ; 
 # but here we look up only on lasso & ridge method results

#train test split
from sklearn.model_selection import train_test_split

train, test = train_test_split(df1, test_size = 0.2, random_state=42)

# Preparing the model on train data 
model = smf.ols('Life_expectancy ~  Country + Status + Year + Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure +Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling', data = train).fit()
       
model.summary()

#evaluation on test data

# Prediction 
pred_test = model.predict(test)

# Error
resid_test  = pred_test - test.Life_expectancy

# RMSE value for data 
rmse_test = np.sqrt(np.mean(resid_test * resid_test))
rmse_test   # 3.61 

#evaluation on train data

# Prediction 
pred_train = model.predict(train)
# Error
resid_train  = pred_train - train.Life_expectancy
# RMSE value for data 
rmse_train = np.sqrt(np.mean(resid_train * resid_train))
rmse_train  # 3.52


# To overcome the issues(reduce error value OR over fit problem), LASSO and RIDGE regression are used

######## LASSO REGRESSION MODEL ##########
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso()

parameters_l = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters_l, scoring = 'r2', cv = 5)
lasso_reg.fit(train.iloc[:, 1:], train.Life_expectancy)

lasso_reg.best_params_   #  1e-08
lasso_reg.best_score_   #  0.83

lasso_pred_train = lasso_reg.predict(train.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(train.iloc[:, 1:], train.Life_expectancy)  #0.84

# RMSE
np.sqrt(np.mean((lasso_pred_train - train.Life_expectancy)**2)) # 3.524

# lasso model for best alpha value
lasso = Lasso(alpha = 49, normalize = True)

# fit the train data with best alpha lasso model
lasso.fit(train.iloc[:, 1:], train.Life_expectancy) 

# Coefficient values for all independent variables#
lasso.coef_    #  eliminated all independant input variable coefficient, hence its not a good approach. either go for another set of lambda value gor for ridge and find a better result
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(train.columns[1:]))

###LASSO Evaluation on Test Data###

# Prediction 
pred_lasso_test = lasso.predict(test.iloc[:, 1:])

# Error
resid_test  = pred_lasso_test - test.Life_expectancy

# RMSE value for data 
rmse_lasso_test = np.sqrt(np.mean(resid_test * resid_test))
rmse_lasso_test    #  8.46

# Test data score in adusted r_square term
lasso_reg.score(test.iloc[:, 1:], test.Life_expectancy)   # 0.82


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
ridge_reg.fit(train.iloc[:, 1:], train.Life_expectancy)

ridge_reg.best_params_   # 0.009999888671826829
ridge_reg.best_score_    # 0.83

ridge_pred_train = ridge_reg.predict(train.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(train.iloc[:, 1:], train.Life_expectancy)   # 0.84

# RMSE
np.sqrt(np.mean((ridge_pred_train - train.Life_expectancy)**2))   # 3.52

# ridge model for best alpha value
ridge = Ridge(alpha = 0.009999888671826829, normalize = True)

# fit the train data with best alpha ridge model
ridge.fit(train.iloc[:, 1:], train.Life_expectancy) 

# Coefficient values for all independent variables#
ridge.coef_
ridge.intercept_

plt.bar(height = pd.Series(ridge.coef_), x = pd.Series(train.columns[1:]))

###RIDGE Evaluation on Test Data###

# Prediction 
pred_ridge_test = ridge.predict(test.iloc[:, 1:])
# Error
resid_test  = pred_ridge_test - test.Life_expectancy

# RMSE value for data 
rmse_ridge_test = np.sqrt(np.mean(resid_test * resid_test))
rmse_ridge_test   # 3.64

# Test data score in adusted r_square term
ridge_reg.score(test.iloc[:, 1:], test.Life_expectancy)   # 0.82


########## ELASTIC NET REGRESSION MODEL ##########

from sklearn.linear_model import ElasticNet
help(ElasticNet)

enet = ElasticNet()

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(train.iloc[:, 1:], train.Life_expectancy)

enet_reg.best_params_     # 9.999888671826828e-05
enet_reg.best_score_      # -13.109

enet_pred_train = enet_reg.predict(train.iloc[:, 1:])

# Adjusted r-square#
enet_reg.score(train.iloc[:, 1:], train.Life_expectancy)  # -12.415

# RMSE
np.sqrt(np.mean((enet_pred_train - train.Life_expectancy)**2))   # 3.52

# ElasticNet model for best alpha value
enet = ElasticNet(alpha = 9.999888671826828e-05, normalize = True)

# fit the train data with best alpha ElasticNet model
enet.fit(train.iloc[:, 1:], train.Life_expectancy) 

# Coefficient values for all independent variables#
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(train.columns[1:]))

###ElasticNet Evaluation on Test Data###

# Prediction 
pred_enet_test = enet.predict(test.iloc[:, 1:])

# Error
resid_test  = pred_enet_test - test.Life_expectancy

# RMSE value for data 
rmse_enet_test = np.sqrt(np.mean(resid_test * resid_test))
rmse_enet_test  # 3.67

# Test data score in adusted r_square term
enet_reg.score(test.iloc[:, 1:], test.Life_expectancy)   # -13.02
