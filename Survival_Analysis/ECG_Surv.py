import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.impute import SimpleImputer
# pip install lifelines
# import lifelines
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy import stats

df = pd.read_excel("C://Users//user//Downloads//survival//ECG_Surv.xlsx")
df.head()
df.columns

print(df.isnull().sum())
print(df.shape)

# gonna implement imputation for missing values with means of each columns.

imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
COLUMNS = ['age', 'pericardialeffusion', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-score']
X = imp_mean.fit_transform(df[COLUMNS])
df_X = pd.DataFrame(X, columns = COLUMNS)
df_X.shape

COLUMNS_keep = ['survival_time_hr', 'alive']
df_keep = df[COLUMNS_keep]
df_keep.shape

df = pd.concat([df_keep, df_X], axis = 1)
df = df.dropna() # dropna function applies to survival and alive variables. Not consider imputation for that columns
print(df.isnull().sum())
print(df.shape)

# Scatter plots between survival and covariates
sns.pairplot(df)

#For alive = 1 patients, because they are alive during data collection period and we do not know their survival months after the data collection, they are regarded as censored data. Hence, the following analysis needs to consider the censored data by making dead variable below.

df.loc[df.alive == 1, 'dead'] = 0
df.loc[df.alive == 0, 'dead'] = 1
df.groupby('dead').count()
# We have 82 non-censored data and 51 censored data.

# Kaplan Meier estimates
kmf = KaplanMeierFitter()
T = df['survival_time_hr']
E = df['dead']
kmf.fit(T, event_observed = E)
kmf.plot()
plt.title("Kaplan Meier estimates")
plt.xlabel("Month after heart attack")
plt.ylabel("survival")
plt.show()

# slight negative relationship of age and wallmotion-score to survival, so used median to make two groups within each variable to see difference in survival time.

print(statistics.median(df['age']))
print(statistics.median(df['wallmotion-score']))

age_group = df['age'] < statistics.median(df['age'])
ax = plt.subplot(111)
kmf.fit(T[age_group], event_observed = E[age_group], label = 'below 62')
kmf.plot(ax = ax)
kmf.fit(T[~age_group], event_observed = E[~age_group], label = 'above 62')
kmf.plot(ax = ax)
plt.title("Kaplan Meier estimates by age group")
plt.xlabel("Month after heart attack")
plt.ylabel("survival")

score_group = df['wallmotion-score'] < statistics.median(df['wallmotion-score'])
ax = plt.subplot(111)
kmf.fit(T[score_group], event_observed = E[score_group], label = 'Low score')
kmf.plot(ax = ax)
kmf.fit(T[~score_group], event_observed = E[~score_group], label = 'High score')
kmf.plot(ax = ax)
plt.title("Kaplan Meier estimamtes by wallmotion-score group")
plt.xlabel("Month after heart attack")
plt.ylabel("survival")

# The difference by age groups seems to be weak. However, there seems to differ by wallmotion-score group for the first 24 months (2 years) after heart attack. So applied the following analysis based on wallmotion-score group

# Log-rank test
month_cut = 24
df.loc[(df.dead == 1) & (df.survival_time_hr <= month_cut), 'censored'] = 1
df.loc[(df.dead == 1) & (df.survival_time_hr > month_cut), 'censored'] = 0
df.loc[df.dead == 0, 'censored'] = 0
E_v2 = df['censored']

T_low = T[score_group]
T_high = T[~score_group]
E_low = E_v2[score_group]
E_high = E_v2[~score_group]

results = logrank_test(T_low, T_high, event_observed_A = E_low, event_observed_B = E_high)
results.print_summary()

# "test_statistic" here is a chi-square statistic. It shows chi-square statistic 9.98, and p-value is less than 5%. Thus confirm that there is a significant difference in suvival time by wallmotion score group for the first 2 year after heart attack.

# Cox proportional hazards model
cph = CoxPHFitter()
df_score_group = pd.DataFrame(score_group)
df_model = df[['survival_time_hr', 'censored', 'age']]
df_model = pd.concat([df_model, df_score_group], axis = 1)
cph.fit(df_model, 'survival_time_hr', 'censored')
cph.print_summary()
# p-value of Log-likelihood ratio test
round(stats.chi2.sf(10.68, 2),4)

# p-value of Log-likelihood ratio test
round(stats.chi2.sf(10.68, 2),4)