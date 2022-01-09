import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
df = pd.read_csv("C://Users//user//Downloads//multi nominal//loan.csv")
df.head(10)

data = df.iloc[:, [16,5 , 2, 7,8,9,12,13,14,20,24,25,31,32]]

data.isna().sum()
# discretizion
data['loan_amnt'] = pd.cut(data['loan_amnt'],3,labels=['low','average','high'])
data['annual_inc'] = pd.cut(data['annual_inc'],3,labels=['low','average','high'])
data['dti'] = pd.cut(data['dti'],3,labels=['low','average','high'])
data['installment'] = pd.cut(data['installment'],3,labels=['good','average','worst'])
data['revol_bal'] = pd.cut(data['revol_bal'],3,labels=['low','average','high'])
#labelencoder
from sklearn.preprocessing import LabelEncoder
w=data.describe()
lb = LabelEncoder()

data["term"] = lb.fit_transform(data["term"])
data["loan_amnt"] = lb.fit_transform(data["loan_amnt"])
data["grade"] = lb.fit_transform(data["grade"])
data["sub_grade"] = lb.fit_transform(data["sub_grade"])
data["home_ownership"] = lb.fit_transform(data["home_ownership"])
data["purpose"] = lb.fit_transform(data["purpose"])
data["verification_status"] = lb.fit_transform(data["verification_status"])
data["annual_inc"] = lb.fit_transform(data["annual_inc"])
data["dti"] = lb.fit_transform(data["dti"])
data["installment"] = lb.fit_transform(data["installment"])
data["revol_bal"] = lb.fit_transform(data["revol_bal"])


# Boxplot of independent variable distribution for each category of prog 
sns.set()
fig, axes = plt.subplots(3, 5, figsize=(18, 18))
fig.suptitle('loan')

axes[0,0].set_title("term")
sns.boxplot(ax=axes[0,0],x = "loan_status", y = "term", data = data)
axes[0,1].set_title("dti")
sns.boxplot(ax=axes[0,1],x = "loan_status", y = "dti", data = data)
axes[0,2].set_title("installment")
sns.boxplot(ax=axes[0,2],x = "loan_status", y = "installment", data = data)
axes[0,3].set_title("grade")
sns.boxplot(ax=axes[0,3],x = "loan_status", y = "grade", data = data)
axes[0,4].set_title("sub_grade")
sns.boxplot(ax=axes[0,4],x = "loan_status", y = "sub_grade", data = data)
axes[1,0].set_title("home_ownership")
sns.boxplot(ax=axes[1,0],x = "loan_status", y = "home_ownership", data = data)
axes[1,1].set_title("annual_inc")
sns.boxplot(ax=axes[1,1],x = "loan_status", y = "annual_inc", data = data)
axes[1,2].set_title("verification_status")
sns.boxplot(ax=axes[1,2],x = "loan_status", y = "verification_status", data = data)
axes[1,3].set_title("revol_bal")
sns.boxplot(ax=axes[1,3],x = "loan_status", y = "revol_bal", data = data)
axes[1,4].set_title("purpose")
sns.boxplot(ax=axes[1,4],x = "loan_status", y = "purpose", data = data)
axes[2,0].set_title("delinq_2yrs")
sns.boxplot(ax=axes[2,0],x = "loan_status", y = "delinq_2yrs", data = data)
axes[2,1].set_title("pub_rec")
sns.boxplot(ax=axes[2,1],x = "loan_status", y = "pub_rec", data = data)
axes[2,2].set_title("loan_amnt")
sns.boxplot(ax=axes[2,2],x = "loan_status", y = "loan_amnt", data = data)

# Scatter plot for each categorical choice of car
fig, axes = plt.subplots(3, 5, figsize=(18, 18))
fig.suptitle('loan')

axes[0,0].set_title("term")
sns.stripplot(ax=axes[0,0],x = "loan_status", y = "term", jitter = True, data = data)
axes[0,1].set_title("dti")
sns.stripplot(ax=axes[0,1],x = "loan_status", y = "dti", jitter = True, data = data)
axes[0,2].set_title("installment")
sns.stripplot(ax=axes[0,2],x = "loan_status", y = "installment", jitter = True, data = data)
axes[0,3].set_title("grade")
sns.stripplot(ax=axes[0,3],x = "loan_status", y = "grade", jitter = True, data = data)
axes[0,4].set_title("sub_grade")
sns.stripplot(ax=axes[0,4],x = "loan_status", y = "sub_grade", jitter = True, data = data)
axes[1,0].set_title("home_ownership")
sns.stripplot(ax=axes[1,0],x = "loan_status", y = "home_ownership", jitter = True, data = data)
axes[1,1].set_title("annual_inc")
sns.stripplot(ax=axes[1,1],x = "loan_status", y = "annual_inc", jitter = True, data = data)
axes[1,2].set_title("verification_status")
sns.stripplot(ax=axes[1,2],x = "loan_status", y = "verification_status", jitter = True, data = data)
axes[1,3].set_title("revol_bal")
sns.stripplot(ax=axes[1,3],x = "loan_status", y = "revol_bal", jitter = True, data = data)
axes[1,4].set_title("purpose")
sns.stripplot(ax=axes[1,4],x = "loan_status", y = "purpose", jitter = True, data = data)
axes[2,0].set_title("delinq_2yrs")
sns.stripplot(ax=axes[2,0],x = "loan_status", y = "delinq_2yrs", jitter = True, data = data)
axes[2,1].set_title("pub_rec")
sns.stripplot(ax=axes[2,1],x = "loan_status", y = "pub_rec", jitter = True, data = data)
axes[2,2].set_title("loan_amnt")
sns.stripplot(ax=axes[2,2],x = "loan_status", y = "loan_amnt", jitter = True, data = data)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(data) # Normal
sns.pairplot(data, hue = "loan_amnt") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
s=data.corr()

train, test = train_test_split(data, test_size = 0.3)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)   # 0.8287176905001679
train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict)  # 0.8295744757382828

