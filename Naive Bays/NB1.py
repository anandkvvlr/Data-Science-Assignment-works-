import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix


#To import dataset
df = pd.read_csv("C://Users//user//Downloads//naive bayes//SalaryData_Train.csv",encoding = "ISO-8859-1")
#To find unique values
df['Salary'].unique()
df['relationship'].unique()
df["race"].unique()
df['workclass'].unique()
df['education'].unique()

#To get dummies
df_new= pd.get_dummies(df[['workclass', 'education', 'educationno', 'maritalstatus',
       'occupation', 'relationship', 'race', 'sex', 'native']],drop_first=True)
df1=df.drop(['workclass', 'education', 'educationno', 'maritalstatus',
       'occupation', 'relationship', 'race', 'sex', 'native','Salary'],axis=1)
df_y=df.iloc[:,13]

df2=df_new.iloc[:,0:90]
df_x= pd.concat([df1,df2],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.30,random_state=0)


#Gaussian Naive Bayes
model=GaussianNB()
model.fit(x_train,y_train)

#evaluation on test data
predicted_test=model.predict(x_test)
predicted_test
accuracy_score(y_test,predicted_test)*100
pd.crosstab(predicted_test,y_test)
print(metrics.classification_report(y_test,predicted_test))
print(metrics.confusion_matrix(y_test,predicted_test))

#evaluation on train data
predicted_train=model.predict(x_train)
predicted_train
accuracy_score(y_train,predicted_train)*100
pd.crosstab(predicted_train,y_train)
print(metrics.classification_report(y_train,predicted_train))
print(metrics.confusion_matrix(y_train,predicted_train))

#after smoothing guassianNB
#var_smoothing,default=1e-9

model=GaussianNB(var_smoothing=3)
model.fit(x_train,y_train)

#evaluation on test data
predicted_test=model.predict(x_test)
predicted_test
accuracy_score(y_test,predicted_test)*100
pd.crosstab(predicted_test,y_test)
print(metrics.classification_report(y_test,predicted_test))
print(metrics.confusion_matrix(y_test,predicted_test))

#evaluation on train data
predicted_train=model.predict(x_train)
predicted_train
accuracy_score(y_train,predicted_train)*100
pd.crosstab(predicted_train,y_train)
print(metrics.classification_report(y_train,predicted_train))
print(metrics.confusion_matrix(y_train,predicted_train))
