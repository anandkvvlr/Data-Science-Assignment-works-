import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix

#To import dataset
df = pd.read_csv("C://Users//user//Downloads//naive bayes//NB_Car_Ad.csv",encoding = "ISO-8859-1")

df.columns
gender=pd.get_dummies(df[['Gender']],drop_first=True)

det = pd.concat([df, gender], join = 'outer', axis = 1)

car_dataX = det.iloc[:, [0,5,2,3]]
car_dataY = det.iloc[:, [4]]
x_train,x_test,y_train,y_test=train_test_split(car_dataX,car_dataY,test_size=0.30,random_state=0)

#Gaussian Naive Bayes
model=GaussianNB()
model.fit(x_train,y_train)

#evaluation on test data
predicted_test=model.predict(x_test)
predicted_test
accuracy_score(y_test,predicted_test)*100
print(metrics.classification_report(y_test,predicted_test))
print(metrics.confusion_matrix(y_test,predicted_test))

#evaluation on train data
predicted_train=model.predict(x_train)
predicted_train
accuracy_score(y_train,predicted_train)*100
print(metrics.classification_report(y_train,predicted_train))
print(metrics.confusion_matrix(y_train,predicted_train))

#after smoothing guassianNB
#var_smoothing,default=1e-9

model=GaussianNB(var_smoothing=2)
model.fit(x_train,y_train)

#evaluation on test data
predicted_test=model.predict(x_test)
predicted_test
accuracy_score(y_test,predicted_test)*100
print(metrics.classification_report(y_test,predicted_test))
print(metrics.confusion_matrix(y_test,predicted_test))

#evaluation on train data
predicted_train=model.predict(x_train)
predicted_train
accuracy_score(y_train,predicted_train)*100
print(metrics.classification_report(y_train,predicted_train))
print(metrics.confusion_matrix(y_train,predicted_train))

