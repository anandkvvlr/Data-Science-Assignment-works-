import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("C://Users//user//Downloads//multi nominal//mdata.csv")
data.head(10)

data.prog.value_counts()
data.isna().sum()
from sklearn.preprocessing import LabelEncoder
data.describe()
lb = LabelEncoder()
data["female"] = lb.fit_transform(data["female"])
data["ses"] = lb.fit_transform(data["ses"])
data["schtyp"] = lb.fit_transform(data["schtyp"])
data["honors"] = lb.fit_transform(data["honors"])

data = data.iloc[:, [5 , 1, 2, 3, 4, 6, 7,8,9,10]]

# Boxplot of independent variable distribution for each category of prog 


sns.set()
fig, axes = plt.subplots(3, 3, figsize=(15, 8))
fig.suptitle('loan')

axes[0,0].set_title("id")
sns.boxplot(ax=axes[0,0],x = "prog", y = "id", data = data)
axes[0,1].set_title("female")
sns.boxplot(ax=axes[0,1],x = "prog", y = "female", data = data)
axes[0,2].set_title("ses")
sns.boxplot(ax=axes[0,2],x = "prog", y = "ses", data = data)
axes[1,0].set_title("schtyp")
sns.boxplot(ax=axes[1,0],x = "prog", y = "schtyp", data = data)
axes[1,1].set_title("read")
sns.boxplot(ax=axes[1,1],x = "prog", y = "read", data = data)
axes[1,2].set_title("write")
sns.boxplot(ax=axes[1,2],x = "prog", y = "write", data = data)
axes[2,0].set_title("math")
sns.boxplot(ax=axes[2,0],x = "prog", y = "math", data = data)
axes[2,1].set_title("honors")
sns.boxplot(ax=axes[2,1],x = "prog", y = "honors", data = data)
axes[2,2].set_title("science")
sns.boxplot(ax=axes[2,2],x = "prog", y = "science", data = data)

# Scatter plot for each categorical choice of prog
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
fig.suptitle('loan')

axes[0,0].set_title("id")
sns.stripplot(ax=axes[0,0],x = "prog", y = "id", jitter = True, data = data)
axes[0,1].set_title("female")
sns.stripplot(ax=axes[0,1],x = "prog", y = "female", jitter = True, data = data)
axes[0,2].set_title("ses")
sns.stripplot(ax=axes[0,2],x = "prog", y = "ses", jitter = True, data = data)
axes[1,0].set_title("schtyp")
sns.stripplot(ax=axes[1,0],x = "prog", y = "schtyp", jitter = True, data = data)
axes[1,1].set_title("read")
sns.stripplot(ax=axes[1,1],x = "prog", y = "read", jitter = True, data = data)
axes[1,2].set_title("write")
sns.stripplot(ax=axes[1,2],x = "prog", y = "write", jitter = True, data = data)
axes[2,0].set_title("math")
sns.stripplot(ax=axes[2,0],x = "prog", y = "math", jitter = True, data = data)
axes[2,1].set_title("honors")
sns.stripplot(ax=axes[2,1],x = "prog", y = "honors", jitter = True, data = data)
axes[2,2].set_title("science")
sns.stripplot(ax=axes[2,2],x = "prog", y = "science", jitter = True, data = data)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(data) # Normal
sns.pairplot(data, hue = "prog") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
data.corr()
 
train, test = train_test_split(data, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)   # 0.60

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict)  # 0.70
