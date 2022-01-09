#Transformation
import pandas as pd
import scipy.stats as stats
import pylab
import numpy as np
import matplotlib.pyplot as plt 
df = pd.read_csv("D:\\360DigiTMG\\360digitmg DS 0607\\ASSIGNMENTS\\Data Preprocessing\\DataSets\\calories_consumed.csv")
df.columns
# Checking Whether data is normally distributed
stats.probplot(df['Weight gained (grams)'], dist="norm",plot=pylab)

stats.probplot(df['Calories Consumed'],dist="norm",plot=pylab)

stats.probplot(np.log(df['Weight gained (grams)']),dist="norm",plot=pylab)
stats.probplot(np.log(df['Calories Consumed']),dist="norm",plot=pylab)


#### Log Transformation :  y , x = log(x)

plt.scatter(np.log(df['Calories Consumed']),df['Weight gained (grams)'])
np.corrcoef(np.log(df['Calories Consumed']),df['Weight gained (grams)'])[0,1]


#### Exponential Transformation : y = log(y), x

#plt.scatter(df['Calories Consumed'], np.log(df['Weight gained (grams)']))
#np.corrcoef(df['Calories Consumed'], np.log(df['Weight gained (grams)']))[0,1]

 
