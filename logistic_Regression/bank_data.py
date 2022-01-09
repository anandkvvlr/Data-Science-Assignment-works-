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
df = pd.read_csv("C:/Users/user/Downloads/logistic reg/bank_data.csv")

# create a copy
df1 =  df.copy(deep= True)
df1 = df1[['y','age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign',
       'pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess',
       'poutunknown', 'con_cellular', 'con_telephone', 'con_unknown',
       'divorced', 'married', 'single', 'joadmin.', 'joblue.collar',
       'joentrepreneur', 'johousemaid', 'jomanagement', 'joretired',
       'joself.employed', 'joservices', 'jostudent', 'jotechnician',
       'jounemployed', 'jounknown']]


#changing column names
df1.rename({'joadmin.':'j' ,'con_cellular':'cc' ,'con_telephone':'ct','con_unknown':'cu','joblue.collar':'jc','joself.employed':'je'  }, axis=1, inplace =True)

#removing CASENUM
c1 = df1
c1.head(11)
c1.describe()
c1.info()
c1.isna().sum()   # no null values
       
#######  Outlier Treatment  ########  
# Boxplots 

for i in c1.columns:
  sns.boxplot(c1.iloc[:,6].dropna())
  plt.show()     ##  have outliers ; combined boxplot showig outliers

#individual boxplot for numeric data columns(except binary)
sns.boxplot(c1['age']);plt.title('box plot for age')   # outlier present
sns.boxplot(c1['balance']);plt.title('box plot for balance')   # outlier present
sns.boxplot(c1['duration']);plt.title('box plot for duration')   # outlier present

 
#outlier treatment for c1.age

q1 = c1['age'].quantile(0.25)
q1
q3 = c1['age'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#finding outliers indexes
outliers = np.where(c1['age']<lower_limit, True, np.where(c1['age']>upper_limit, True, False))
outliers

#finding outlier values
outlier_values = c1['age'][outliers]
outliers.sum()
outlier_values

#Replacing by pulling outliers to lower and upper limit

c1['age'] = np.where(c1['age']<q1, lower_limit, np.where(c1['age']>q3, upper_limit, c1['age']))
sns.boxplot(c1['age']);plt.title('box plot for age after replacing')
    
#outlier treatment for c1.balance

q1 = c1['balance'].quantile(0.25)
q1
q3 = c1['balance'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#finding outliers indexes
outliers = np.where(c1['balance']<lower_limit, True, np.where(c1['balance']>upper_limit, True, False))
outliers

#finding outlier values
outlier_values = c1['balance'][outliers]
outliers.sum()
outlier_values

#Replacing by pulling outliers to lower and upper limit

c1['balance'] = np.where(c1['balance']<q1, lower_limit, np.where(c1['balance']>q3, upper_limit, c1['balance']))
sns.boxplot(c1['balance']);plt.title('box plot for balance after replacing')
    
#outlier treatment for c1.duration

q1 = c1['duration'].quantile(0.25)
q1
q3 = c1['duration'].quantile(0.75)
q3
IQR = q3 - q1
IQR

lower_limit = q1-(1.5*IQR)
lower_limit
upper_limit = q3+(1.5*IQR)
upper_limit

#finding outliers indexes
outliers = np.where(c1['duration']<lower_limit, True, np.where(c1['duration']>upper_limit, True, False))
outliers

#finding outlier values
outlier_values = c1['duration'][outliers]
outliers.sum()
outlier_values

#Replacing by pulling outliers to lower and upper limit

c1['duration'] = np.where(c1['duration']<q1, lower_limit, np.where(c1['duration']>q3, upper_limit, c1['duration']))
sns.boxplot(c1['duration']);plt.title('box plot for duration after replacing')



######  zero variance operation   ###
c1.shape

##  importing  ###
from sklearn.feature_selection import VarianceThreshold

# Feature selector that removes all low-variance features that meets the variance threshold limit
var_thres = VarianceThreshold(threshold=0.02) # Threshold is subjective.
var_thres.fit(c1)     ###   fit the var_thres to data set c11
# Generally we remove the columns with zero variance, but i took thresold value 0.02 (Near Zero Variance)

var_thres.get_support()     ### it  giving an array out, where zero variant column treat as False value. we already fit var_thres to c1. so it gives corresponding information on c1
c1.columns[var_thres.get_support()]    ##     non-zero variant column names 
constant_columns = [column for column in c1.columns if column not in c1.columns[var_thres.get_support()]]

print(len(constant_columns))   ### number of zero variant  variables

for feature in constant_columns:
    print(feature)               ### names of corresponding zero variant columns ; "default", "jounknown"

c1 = c1.drop(constant_columns, axis = 1)    ### data set with non-zero variant variables or features after droping "default", "jounknown" columns



### normalisation scaling  ###

#Importing the Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

# define standard scaler
scaler = MinMaxScaler() # Standard Scaler or Standardization

# scaling data  except output column
c1.iloc[:,1:] = scaler.fit_transform(c1.iloc[:,1:]) #Fit to data, then transform it.

c1.iloc[:,1:].describe()


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

logit_model = sm.logit('y ~  age + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + cc + ct + cu + divorced + married + single + j + jc + joentrepreneur + johousemaid + jomanagement + joretired + je + joservices + jostudent + jotechnician + jounemployed', data= train_data).fit()

#summary 
logit_model.summary2() # for AIC
logit_model.summary() # some of the feature column showing P_value > 0.05 and nan values. so actually we have to rectify multicollinearity problem between input variables. But we skip that step and directly going logistic operation 
#nan values are appearing ; this should be consequence of the first error ==> " maximum likelihood failed to converge" so need to check the format and type of varables of data properly

pred = logit_model.predict(train_data.iloc[ :, 1:])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(train_data.y, pred)
optimal_idx = np.argmax(tpr - fpr)  
optimal_threshold = thresholds[optimal_idx]
optimal_threshold   #  0.086  choose the threshold point giving max diff between tpr and fpr 

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]   
# fpr       tpr     1-fpr        tf  thresholds
#0.206087  0.794198  0.793913  0.000285    0.118624
#threshold = 0.118624 ; choose the threshold value where tpr and tnr are high and with minimum difference (sensitivity and specificity are high and with minimum diff)
# almost equal and near to optimum_threshold getting from corresponding max tpr- fpr value

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])   # graph showing tpr value almost near to 0.81 for optimum threshold point

#ROC CURVE AND AUC
plt.plot(fpr, tpr, label = "ROC curve on train data");plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend()

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)   # 0.87 ; means false in excellent region


# prediction of output according to propbability and threshold value

# filling all the cells with zeroes
l_train = len(train_data["y"])
l_train
train_data["pred"] = np.zeros(31647)
# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[pred > optimal_threshold, "pred"] = 1

# classification report
print(classification_report(train_data["y"], train_data["pred"] ))   # accuracy = 0.74

# confusion matrix
confusion_matrx = pd.crosstab(train_data["pred"], train_data['y'])
confusion_matrx

accuracy_train = (20256 + 3215)/(31647)   # TP = 3215, TN = 20256 , FN = 439, FP = 7737
print(accuracy_train)  # 0.74


#################### Analysis on Test  data  #######################################

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
l_test = len(test_data["y"])
l_test
test_data["test_pred"] = np.zeros(13564)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = (8595 + 1409)/(13564) # TP = 1409, TN = 8595, FN = 226, FP = 3334
accuracy_test   # 0.74

# classification report
print(classification_report(test_data["test_pred"], test_data["y"]))  # accuracy = 0.74


#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr, label = "ROC curve on test data");plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend()

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test  # area under the curve = 0.87 : fall in excellent region 
