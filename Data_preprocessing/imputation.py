# Imputation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("D:\\360DigiTMG\\360digitmg DS 0607\\ASSIGNMENTS\\Data Preprocessing\\DataSets\\claimants.csv")
data.isna().sum()


# Boxplots 

for i in data.describe().columns:
  sns.boxplot(data[i].dropna())
  plt.show()


sns.boxplot(data.CLMSEX);plt.title('Boxplot');plt.show() # no outliers, mean imputation
sns.boxplot(data.SEATBELT);plt.title('Boxplot');plt.show()
sns.boxplot(data.CLMAGE);plt.title('Boxplot');plt.show() # outliers present, median imputation
# for Mean,Meadian,Mode imputation we can use Simple Imputer
from sklearn.impute import SimpleImputer
# Mean Imputer 
#mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#data["CLMAGE"] = pd.DataFrame(mean_imputer.fit_transform(data[["CLMAGE"]]))
#data["CLMAGE"].isnull().sum()  # all 12 records replaced by mean 


# Median Imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(data[["CLMAGE"]]))
data["CLMAGE"].isnull().sum()  # all 189 records replaced by median


# Mode Imputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data["SEATBELT"] = pd.DataFrame(mode_imputer.fit_transform(data[["SEATBELT"]]))
data["CLMSEX"] = pd.DataFrame(mode_imputer.fit_transform(data[["CLMSEX"]]))
data["CLMINSUR"] = pd.DataFrame(mode_imputer.fit_transform(data[["CLMINSUR"]]))
data.isnull().sum()  # all SEX,MaritalDesc,Salaries records replaced by mode
