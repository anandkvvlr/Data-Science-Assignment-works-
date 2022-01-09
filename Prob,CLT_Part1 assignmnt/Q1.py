############## proability part 1 ###############
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv("C://Users//user//Downloads//Prob,CLT//cars.csv")

data.isnull().sum()  # no null values
data.dropna()
data.columns

# creating copy
data1= data.copy(deep=True)
data2= data.copy(deep=True)
data3= data.copy(deep=True)

#converting into categorical & binary
lb = LabelEncoder()

## discretize ouput column ;;MPG > 38
data1['MPG'].describe()  # max value = 53.700681
bins=[0,38,54]           # [<=38],(>38]
group_names= ['MPG_[<=38]','MPG_(>38]']    # (a,b] => a not included; but b included 
data1['MPG']= pd.cut(data1['MPG'],bins, labels = group_names)

#summary
data1.dropna()
data1.head(10)
data1.describe()

# counts
data1['MPG'].value_counts()
len(data1['MPG'])

## discretize ouput column ;;MPG < 40 
data2['MPG'].describe()  # max value = 53.700681
bins=[0,39.9999,54]           # [<40],(>=40]
group_names= ['MPG_[<40]','MPG_(>=40]']    # (a,b] => a not included; but b included 
data2['MPG']= pd.cut(data2['MPG'],bins, labels = group_names)

#summary
data2.dropna()
data2.head(10)
data2.describe()

# counts
data2['MPG'].value_counts()
len(data2['MPG'])

## discretize ouput column ;;P (20<MPG<50)  
data3['MPG'].describe()  # max value = 53.700681
bins=[0,20,49.9999,54]           # [<40],(>=40]
group_names= ['MPG_[<=20]','MPG_(20<MPG<50]','MPG_[>=50]']    # (a,b] => a not included; but b included 
data3['MPG']= pd.cut(data3['MPG'],bins, labels = group_names)

#summary
data3.dropna()
data3.head(10)
data3.describe()

# counts
data3['MPG'].value_counts()
len(data3['MPG'])

###################### normal distribution check    ##################
mean= data['MPG'].mean()
std= data['MPG'].std()
print('min. MPG = ',data['MPG'].min())
print('max. MPG = ',data['MPG'].max())
print('mean MPG = ',mean)
print('std.dev MPG = ',std)

# histogram
fig, axs = plt.subplots(figsize=(10,5))
axs.hist(data['MPG'],bins=15)
axs.set_title('histogram of MPG')
axs.set_xlabel("bins")
axs.set_ylabel('count')

df = data.MPG
x=[]

# verifying normal distribution within one std
within_one_std_deviation = data[(mean-std) > data['MPG']]
within_one_std_deviation = data[data['MPG'] < (mean+std)]

new_data = [item for item in data.MPG if item > (mean-std)]
new_data = [item for item in new_data if item < (mean+std)]
df = pd.DataFrame(new_data)
df.describe()

# percentage of data points falling within one std.dev
percn_one_std = len(df)/len(data.MPG) * 100
percn_one_std

# verifying normal distribution within 2 std
new_data = [item for item in data.MPG if item > (mean-(2*std))]
new_data = [item for item in new_data if item < (mean+(2*std))]
df2 = pd.DataFrame(new_data)
df2.describe()
                                    
# percentage of data points falling within two std.dev
percn_two_std = len(df2)/len(data.MPG) * 100         
percn_two_std

# verifying normal distribution within 3 std
new_data = [item for item in data.MPG if item > (mean-(3*std))]
new_data = [item for item in new_data if item < (mean+(3*std))]
df3 = pd.DataFrame(new_data)
df3.describe()
                                    
# percentage of data points falling within 3 std.dev
percn_three_std = len(df3)/len(data.MPG) * 100 
percn_three_std

##b)check Whether the Adipose Tissue (AT) and Waist Circumference (Waist) from wc-at data set follows Normal Distribution

d = pd.read_csv("C://Users//user//Downloads//Prob,CLT//wc-at.csv")

d.isnull().sum()  # no null values
d.dropna()
d.columns

#### for waist #######

## normal distribution check
mean= d['Waist'].mean()
std= d['Waist'].std()
print('min. Waist = ',d['Waist'].min())
print('max. Waist = ',d['Waist'].max())
print('mean Waist = ',mean)
print('std.dev Waist = ',std)

# histogram
fig, axs = plt.subplots(figsize=(10,5))
axs.hist(d['Waist'],bins=15)
axs.set_title('histogram of Waist')
axs.set_xlabel("bins")
axs.set_ylabel('count')

df = d.Waist
x=[]

# verifying normal distribution within one std
new_data = [item for item in d.Waist if item > (mean-std)]
new_data = [item for item in new_data if item < (mean+std)]
df = pd.DataFrame(new_data)
df.describe()

# percentage of data points falling within one std.dev
percn_one_std = len(df)/len(d.Waist) * 100
percn_one_std

# verifying normal distribution within 2 std
new_data = [item for item in d.Waist if item > (mean-(2*std))]
new_data = [item for item in new_data if item < (mean+(2*std))]
df2 = pd.DataFrame(new_data)
df2.describe()
                                    
# percentage of data points falling within two std.dev
percn_two_std = len(df2)/len(d.Waist) * 100         
percn_two_std

# verifying normal distribution within 3 std
new_data = [item for item in d.Waist if item > (mean-(3*std))]
new_data = [item for item in new_data if item < (mean+(3*std))]
df3 = pd.DataFrame(new_data)
df3.describe()
                                    
# percentage of data points falling within two std.dev
percn_three_std = len(df3)/len(d.Waist) * 100 
percn_three_std

#### for AT  ##########
## normal distribution check
mean= d['AT'].mean()
std= d['AT'].std()
print('min. AT = ',d['AT'].min())
print('max. AT = ',d['AT'].max())
print('mean AT = ',mean)
print('std.dev AT = ',std)

# histogram
fig, axs = plt.subplots(figsize=(10,5))
axs.hist(d['AT'],bins=15)
axs.set_title('histogram of AT')
axs.set_xlabel("bins")
axs.set_ylabel('count')

df = d.Waist
x=[]

# verifying normal distribution within one std
new_data = [item for item in d.AT if item > (mean-std)]
new_data = [item for item in new_data if item < (mean+std)]
df = pd.DataFrame(new_data)
df.describe()

# percentage of data points falling within one std.dev
percn_one_std = len(df)/len(d.AT) * 100
percn_one_std

# verifying normal distribution within 2 std
new_data = [item for item in d.AT if item > (mean-(2*std))]
new_data = [item for item in new_data if item < (mean+(2*std))]
df2 = pd.DataFrame(new_data)
df2.describe()
                                    
# percentage of data points falling within two std.dev
percn_two_std = len(df2)/len(d.AT) * 100         
percn_two_std

# verifying normal distribution within 3 std
new_data = [item for item in d.AT if item > (mean-(3*std))]
new_data = [item for item in new_data if item < (mean+(3*std))]
df3 = pd.DataFrame(new_data)
df3.describe()
                                    
# percentage of data points falling within two std.dev
percn_three_std = len(df3)/len(d.AT) * 100 
percn_three_std