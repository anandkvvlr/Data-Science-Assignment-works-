# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("C:/Users/user/Downloads/New folder (3)/Datasets_SLR/delivery_time.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

#changing column names
df.rename({'Delivery Time':'dt' ,'Sorting Time':'st' }, axis=1, inplace =True)

#Data Cleaning

###### Null value Treatment  ########
df.isna().sum()   ## no null values

###### Summary of the data set ####
df.columns
df.describe()
  
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.figure(figsize= (12,3))
plt.subplot(1,3,1)
plt.bar(height = df['dt'], x = np.arange(1, 22, 1))
plt.title('bar plot')
plt.subplot(1,3,2)
plt.hist(df['dt'],bins = 8) #histogram
plt.title('histogram')
plt.subplot(1,3,3)
plt.boxplot(df['dt']) #boxplot
plt.title('boxplot')

plt.figure(figsize= (12,3))
plt.subplot(1,3,1)
plt.bar(height = df['st'], x = np.arange(1, 22, 1))
plt.title('bar plot')
plt.subplot(1,3,2)
plt.hist(df['st'],bins = 8) #histogram
plt.title('histogram')
plt.subplot(1,3,3)
plt.boxplot(df['st']) #boxplot
plt.title('boxplot')

# Scatter plot
plt.scatter(x = df['st'], y = df['dt'], color = 'green') 

# correlation
np.corrcoef(df['st'], df['dt']) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(df['st'], df['dt'])[0, 1]
cov_output

df.cov()

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('dt ~ st', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['st']))

# Regression Line
plt.scatter(df['st'], df['dt'])
plt.plot(df['st'], pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.title('linear model')
plt.show()

# Error calculation
res1 = df.dt - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(st); y = dt

plt.scatter(x = np.log(df['st']), y = df['dt'], color = 'brown')
np.corrcoef(np.log(df['st']), df['dt']) #correlation

model2 = smf.ols('dt ~ np.log(st)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['st']))

# Regression Line
plt.scatter(np.log(df['st']), df['dt'])
plt.plot(np.log(df['st']), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.title('logerithmic model')
plt.show()

# Error calculation
res2 = df['dt'] - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = st; y = log(dt)

plt.scatter(x = df['st'], y = np.log(df['dt']), color = 'orange')
np.corrcoef(df['st'], np.log(df['dt'])) #correlation

model3 = smf.ols('np.log(dt) ~ st', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['st']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(df['st'], np.log(df['dt']))
plt.plot(df['st'], pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.title("exponential model")
plt.show()

# Error calculation
res3 = df.dt - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = st; x^2 = st*st; y = log(dt)

model4 = smf.ols('np.log(dt) ~ st + I(st*st)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 1: ].values
X_poly = poly_reg.fit_transform(X)
# y = df.iloc[:, 1].values


plt.scatter(df['st'], np.log(df['dt']))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.title("plynomial model")
plt.show()


# Error calculation
res4 = df.dt - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model is polynomial model
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2, random_state = 42)

finalmodel = smf.ols('np.log(dt) ~ st + I(st*st)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_dt = np.exp(test_pred)
pred_test_dt

# Model Evaluation on Test data
test_res = test.dt - pred_test_dt
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_dt = np.exp(train_pred)
pred_train_dt

# Model Evaluation on train data
train_res = train.dt - pred_train_dt
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

