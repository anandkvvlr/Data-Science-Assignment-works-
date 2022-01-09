#### affairs ####

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
df = pd.read_csv("C:/Users/user/Downloads/logistic reg/Affairs.csv")

# droping index column
df = df.iloc[:,1:]

#removing CASENUM
c1 = df
c1.head(11)
c1.describe()
c1.isna().sum()   # no null values

#######  Outlier Treatment  ########  
# Boxplots 

for i in c1.columns:
  sns.boxplot(c1[i].dropna())
  plt.show()     ##  no outliers ; combined boxplot not showig any outliers
  

# converting naffairs column to binary format
n= len(c1['naffairs'])
n
for i in range(0,n,1):
    if(c1['naffairs'][i] ==0):
        c1['naffairs'][i] = 0
    else:
        c1['naffairs'][i] = 1
        
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

logit_model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = train_data).fit()
      
#summary 
logit_model.summary2() # for AIC
logit_model.summary() # nan values are appearing ; this should be consequence of the first error ==> " maximum likelihood failed to converge" so need to check the format and type of varables of data properly

pred = logit_model.predict(train_data.iloc[ :, 1:])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(train_data.naffairs, pred)
optimal_idx = np.argmax(tpr - fpr)  
optimal_threshold = thresholds[optimal_idx]
optimal_threshold   #  0.2815  choose the threshold point giving max diff between tpr and fpr 

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]   
# fpr       tpr     1-fpr        tf  thresholds
#0.326019  0.673267  0.673981 -0.000714    0.238503
#threshold = 0.238503 ; choose the threshold value where tpr and tnr are high and with minimum difference (sensitivity and specificity are high and with minimum diff)
# almost equal and near to optimum_threshold getting from corresponding max tpr- fpr value

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])   # graph showing tpr value almost near to 0.67 for optimum threshold point

#ROC CURVE AND AUC
plt.plot(fpr, tpr, label = "ROC curve on train data");plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend()

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)   # 0.73 ; means false in acceptable region


# prediction of output according to propbability and threshold value

# filling all the cells with zeroes
l_train = len(train_data["naffairs"])
l_train
train_data["pred"] = np.zeros(420)
# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[pred > optimal_threshold, "pred"] = 1

# classification report
print(classification_report(train_data["naffairs"], train_data["pred"] ))   # accuracy = 0.72

# confusion matrix
confusion_matrx = pd.crosstab(train_data["pred"], train_data['naffairs'])
confusion_matrx

accuracy_train = (243 + 60)/(420)   # TP = 60, TN = 243 , FN = 41, FP = 76
print(accuracy_train)  # 0.72



#################### Prediction on Test  data  #######################################

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
l_test = len(test_data["naffairs"])
l_test
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

accuracy_test = (101 + 25)/(181) # TP = 25, TN = 101 , FN = 24, FP = 31
accuracy_test   # 0.69

# classification report
print(classification_report(test_data["test_pred"], test_data["naffairs"]))  # accuracy = 0.70


#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr, label = "ROC curve on test data");plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend()

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test  # area under the curve = 0.67 : fall under poor region 
