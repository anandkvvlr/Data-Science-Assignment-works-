# Multilinear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sklearn

# loading the data
df = pd.read_csv("C://Users//user//Downloads//mlr//50_Startups.csv")

###### Null value Treatment  ########
df.isna().sum()   
df.dropna(axis = 0, inplace = True)   ## drop na values

#changing column names
df.columns
df.rename({'R&D Spend':'RD' ,'Marketing Spend':'Marketing'}, axis=1, inplace =True)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

df.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

## EDA on Dataset -

#Histgram on Profit
sns.distplot(df['Profit'],bins=5,kde=True)

# boxplot
#Check any outlier on features having numeric values
import matplotlib.pyplot as plt
%matplotlib inline
for i in df.iloc[:,0:3]:
    plt.boxplot(df[i],notch=True,patch_artist=True)
    plt.show()

# profit split in State level - Looks Florida has the maximum Profit
sns.barplot(x='State',y='Profit',data=df, palette="Blues_d")
#sns.lineplot(x='State',y='Profit',data=dataset)

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(df.iloc[:, :])
   
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
df["State"] = labelencoder.fit_transform(df.iloc[:, 3])
                             
# Correlation matrix 
df.corr()
sns.heatmap(df.corr(), annot=True)

# we see there collinearity between input variables in comparitively high between
# [marketing & R&D] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
        
ml1 = smf.ols('Profit ~ RD + Administration + Marketing + State', data = df).fit() # regression model

# Summary
ml1.summary()
# p-values for Administration, Marketing & State are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 49 is showing high influence so we can exclude that entire row

df_new = df.drop(df.index[[49]])

# Preparing model                  
ml_new = smf.ols('Profit ~ RD + Administration + Marketing + State', data = df_new).fit() # regression model

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hp = smf.ols('RD ~ Administration + Marketing + State', data = df_new).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_wt = smf.ols('Administration  ~ RD +  Marketing + State', data = df_new).fit().rsquared  
vif_wt = 1/(1 - rsq_wt)

rsq_vol = smf.ols('Marketing  ~ RD + Administration + State', data = df_new).fit().rsquared  
vif_vol = 1/(1 - rsq_vol)

rsq_sp = smf.ols('State ~ RD + Administration + Marketing ', data = df_new).fit().rsquared  
vif_sp = 1/(1 - rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['RD', 'Administration', 'Marketing', 'State'], 'VIF':[vif_hp, vif_wt, vif_vol, vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# another simple way of VIF calculation for whole data
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
#Split Dataset into X and y
X=df.drop(columns='Profit')
y=df['Profit']
pd.DataFrame({'Features':X.columns,'VIF':[ VIF(X.values,i) for i in range(len(X.columns))]}) 

# As RD is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Profit ~ Administration + Marketing + State', data = df).fit()
final_ml.summary() 
# p >0.05 for state column

sm.graphics.influence_plot(final_ml)
# Studentized Residuals = Residual/standard deviation of residuals
# index 19 is showing high influence so we can exclude that entire row

df_new = df.drop(df.index[[19]])

# Final model
final_ml = smf.ols('Profit ~ Administration + Marketing + State', data = df_new).fit()
final_ml.summary() 
# p >0.05 for state column

# eliminate state column too
# Final model
final_ml = smf.ols('Profit ~ Administration + Marketing ', data = df).fit()
final_ml.summary() 

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
sns.residplot(x = pred, y = df.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Profit ~ Administration + Marketing", data = df_train).fit()

# prediction on test data set 
test_pred = model_train.predict(df_test)

# test residual values 
test_resid = test_pred - df_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(df_train)

# train residual values 
train_resid  = train_pred - df_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

