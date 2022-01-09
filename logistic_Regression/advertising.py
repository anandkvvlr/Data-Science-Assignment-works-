#### advertisement ####

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
df = pd.read_csv("C:/Users/user/Downloads/logistic reg/advertising.csv")

# create a copy
df1 =  df.copy(deep= True)

# droping less informative columns
df1.columns
df1.drop('Ad_Topic_Line', axis=1, inplace = True)
df1.drop('Timestamp', axis=1, inplace = True)

#changing column names
df1.rename({'Daily_Time_ Spent _on_Site':'DTSS' ,'Daily Internet Usage':'DIU' }, axis=1, inplace =True)

#removing CASENUM
c1 = df1
c1.head(11)
c1.describe()
c1.info()
c1.isna().sum()   # no null values

#converting into numerical
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
c1["City"] = lb.fit_transform(c1["City"])
c1["Country"] = lb.fit_transform(c1["Country"])

#######  Outlier Treatment  ########  
# Boxplots 

for i in c1.columns:
  sns.boxplot(c1[i].dropna())
  plt.show()     ##  no outliers ; combined boxplot not showig any outliers
 
  
### normalisation scaling  ###

#Importing the Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

# define standard scaler
scaler = MinMaxScaler() # Standard Scaler or Standardization

# scaling data  except output column
c1.iloc[:,0:7] = scaler.fit_transform(c1.iloc[:,0:7]) #Fit to data, then transform it.

c1.iloc[:,0:4].describe()

# Sctter plot and histogram between variables
sns.pairplot(c1) # sp-hp, wt-vol multicolinearity issue
          
#################### Train & Test  split  #######################################

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as smc1.columns
# Model building 
import statsmodels.formula.api as sm

logit_model = sm.logit('Clicked_on_Ad ~ DTSS + Age + Area_Income + DIU + City + Male + Country', data = train_data).fit()
             
#summary 
logit_model.summary2() # for AIC
logit_model.summary() # City, Male & Country column features showing P_value > 0.05. so actually we have to rectify multicollinearity problem between input variables. But we skip that step and directly going logistic operation 

pred = logit_model.predict(train_data.iloc[ :, 0:7])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(train_data.Clicked_on_Ad, pred)
optimal_idx = np.argmax(tpr - fpr)  
optimal_threshold = thresholds[optimal_idx]
optimal_threshold   #  0.7066  choose the threshold point giving max diff between tpr and fpr 

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]   
# fpr       tpr     1-fpr        tf  thresholds
#0.035088  0.963687  0.964912 -0.001225    0.437499
#threshold = 0.437499 ; choose the threshold value where tpr and tnr are high and with minimum difference (sensitivity and specificity are high and with minimum diff)
# almost equal and near to optimum_threshold getting from corresponding max tpr- fpr value

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])   # graph showing tpr value almost near to 0.96 for optimum threshold point

#ROC CURVE AND AUC
plt.plot(fpr, tpr, label = "ROC curve on train data");plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend()

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)   # 0.99 ; means false in outstanding region


# prediction of output according to propbability and threshold value

# filling all the cells with zeroes
l_train = len(train_data["Clicked_on_Ad"])
l_train
train_data["pred"] = np.zeros(700)
# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[pred > optimal_threshold, "pred"] = 1

# classification report
print(classification_report(train_data["Clicked_on_Ad"], train_data["pred"] ))   # accuracy = 0.97

# confusion matrix
confusion_matrx = pd.crosstab(train_data["pred"], train_data['Clicked_on_Ad'])
confusion_matrx

accuracy_train = (340 + 339)/(700)   # TP = 339, TN = 340 , FN = 19, FP = 2
print(accuracy_train)  # 0.97



#################### Predictions on Test  data  #######################################

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
l_test = len(test_data["Clicked_on_Ad"])
l_test
test_data["test_pred"] = np.zeros(300)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Clicked_on_Ad'])
confusion_matrix

accuracy_test = (157 + 136)/(300) # TP = 136, TN = 157 , FN = 6, FP = 1
accuracy_test   # 0.98

# classification report
print(classification_report(test_data["test_pred"], test_data["Clicked_on_Ad"]))  # accuracy = 0.98


#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Clicked_on_Ad"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr, label = "ROC curve on test data");plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend()

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test  # area under the curve = 0.99 : fall in outstanding region 
