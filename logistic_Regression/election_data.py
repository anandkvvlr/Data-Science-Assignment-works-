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
df = pd.read_csv("C:/Users/user/Downloads/logistic reg/election_data.csv")

# create a copy
df1 =  df.copy(deep= True)

# droping less informative columns
df1.columns
df1.drop('Election-id', axis=1, inplace = True)
df1.drop('Year', axis=1, inplace = True)

#changing column names
df1.rename({'Amount Spent':'m_spent' ,'Popularity Rank':'p_rank' }, axis=1, inplace =True)

#removing CASENUM
c1 = df1
c1.head(11)
c1.describe()
c1.info()
c1.isna().sum()   # no null values
c1.dropna()  # directly droping null value row without going for any other imputation strategy since it is less informative
c1.dropna(how = 'any',inplace = True)
c1.isna().sum()

#######  Outlier Treatment  ########  
# Boxplots 

for i in c1.columns:
  sns.boxplot(c1[i].dropna())
  plt.show()     ##  no outliers ; combined boxplot not showig any outliers
 
  
### normalisation scaling  ###

# since input variables ranges are almost near we are directly going for analysis 
# without do any scaling
  
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

logit_model = sm.logit('Result ~ m_spent + p_rank', data = train_data).fit()
             
#summary 
logit_model.summary2() # for AIC
logit_model.summary() # Both input features showing P_value > 0.05. so actually we have to rectify multicollinearity problem between input variables. But we skip that step and directly going logistic operation 

pred = logit_model.predict(train_data.iloc[ :, 1:])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(train_data.Result, pred)
optimal_idx = np.argmax(tpr - fpr)  
optimal_threshold = thresholds[optimal_idx]
optimal_threshold   #  0.679  choose the threshold point giving max diff between tpr and fpr 

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]   
# fpr       tpr     1-fpr        tf  thresholds
#0.0      0.8       1.0        -0.2    0.679359
#threshold = 0.679359 ; choose the threshold value where tpr and tnr are high and with minimum difference (sensitivity and specificity are high and with minimum diff)
# almost equal and near to optimum_threshold getting from corresponding max tpr- fpr value

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])   # graph showing tpr value almost near to 0.8 for optimum threshold point

#ROC CURVE AND AUC
plt.plot(fpr, tpr, label = "ROC curve on train data");plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend()

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)   # 0.9 ; means false in outstanding region


# prediction of output according to propbability and threshold value

# filling all the cells with zeroes
l_train = len(train_data["Result"])
l_train
train_data["pred"] = np.zeros(7)
# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[pred > optimal_threshold, "pred"] = 1

# classification report
print(classification_report(train_data["Result"], train_data["pred"] ))   # accuracy = 0.71

# confusion matrix
confusion_matrx = pd.crosstab(train_data["pred"], train_data['Result'])
confusion_matrx

accuracy_train = (2 + 3)/(7)   # TP = 3, TN = 2 , FN = 2, FP = 0
print(accuracy_train)  # 0.71


#################### Analysis on Test  data  #######################################

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
l_test = len(test_data["Result"])
l_test
test_data["test_pred"] = np.zeros(3)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Result'])
confusion_matrix

accuracy_test = (2 + 1)/(3) # TP = 1, TN = 2 , FN = 0, FP = 0
accuracy_test   # 1.0  ; have very less number of data in test set, hence showing 100% accuracy(all predictions are true)

# classification report
print(classification_report(test_data["test_pred"], test_data["Result"]))


#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Result"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr, label = "ROC curve on test data");plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend()

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test  # area under the curve = 1.0 : fall in outstanding region 



