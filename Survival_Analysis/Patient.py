# pip install lifelines
# import lifelines

import pandas as pd
import matplotlib.pyplot as plt

# Loading the the survival un-employment data
survival_ptnt = pd.read_csv("C://Users//user//Downloads//survival//Patient.csv")

# dropping less informative columns
survival_ptnt.columns
survival_ptnt.drop('PatientID',axis=1,inplace= True)
survival_ptnt.drop('Scenario',axis=1,inplace= True)

survival_ptnt.head()

# histogram
survival_ptnt.Followup.hist();
plt.xlabel('Time before censorship');
plt.ylabel('Frequency(number of patients)');

survival_ptnt.describe()

survival_ptnt['Followup'].describe()

# 'Followup' is referring to time 
T = survival_ptnt.Followup

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=survival_ptnt.Eventtype)

# Time-line estimations plot 
kmf.plot()

survival_ptnt.Eventtype.value_counts()
# We have 6 non-censored data and 4 censored data.



