##############    airlines     ##########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
%matplotlib inline
import statsmodels.api as sm
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

#pip install pystan
#conda install -c conda -forge fbprophet

# Import the AirPassengers dataset
passengers_p = pd.read_excel('C://Users//user//Downloads//forecasting//Airlines Data.xlsx')

passengers = passengers_p.copy(deep= True)
passengers.head()

# converting date column to date time format if it is in  string format
passengers.info() # date column ds is already in date tme format
dates = pd.date_range(start='1995-01-01', freq='MS',periods=len(passengers))  # converting to date time format

#split to month and year column
passengers['Month'] = dates.month
passengers['Year'] = dates.year

passengers.head()

#To get the names of the month
passengers.dtypes
passengers.head()

import calendar
passengers['Month'] = passengers['Month'].apply(lambda x: calendar.month_abbr[x])
passengers.rename({'#Passengers':'Passengers'},axis=1,inplace=True)
passengers = passengers[['Month','Year','Passengers']]

passengers.head()

# adding date column
passengers['Date'] = dates
passengers.set_index('Date',inplace=True)
passengers.head()

# Exploratory Data Analysis
plt.figure(figsize=(10,8))
passengers.groupby('Year')['Passengers'].mean().plot(kind='bar')
plt.show()
# From the above figure we can see that passengers are increasing with the increase in the year

plt.figure(figsize=(10,8))
passengers.groupby('Month')['Passengers'].mean().reindex(index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']).plot(kind='bar')
plt.show()
# From the above figure we can see that more passengers can be seen between months June to September

#Lets plot the data to see the trend and seasonality
passengers_count = passengers['Passengers']

plt.figure(figsize=(10,8))
passengers_count.plot()
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.show()

#Now we start with time series decomposition of this data to understand underlying patterns such as trend, seasonality, cycle and irregular remainder
decompose = sm.tsa.seasonal_decompose(passengers_count,model='multiplicative',extrapolate_trend=8)

fig = decompose.plot()
fig.set_figheight(10)
fig.set_figwidth(8)
fig.suptitle('Decomposition of Time Series')
#Trend
#Time Series Decomposition: To begin with let's try to decipher trends embedded in the above tractor sales time series. It is clearly evident that there is an overall increasing trend in the data along with some seasonal variations. However, it might not always be possible to make such visual inferences.
#So, more formally, we can check stationarity using the following: Plotting Rolling Statistics: We can plot the moving average or moving variance and see if it varies with time. By moving average/variance we mean that at any instant 't', we'll take the average/variance of the last year, i.e. last 12 months. But again this is more of a visual technique.
#Now, let’s try to remove wrinkles from our time series using moving average. We will take moving average of different time periods i.e. 4,6,8, and 12 months as shown below. Here, moving average is shown in orange and actual series in blue.

# Centering moving average for the time series
passengers.Passengers.plot(label = "org")
for i in range(4, 13, 2):
    passengers['Passengers'].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
# As we could see in the above plots, 12-month moving average could produce a wrinkle free curve as desired. This on some level is expected since we are using month-wise data for our analysis and there is expected monthly-seasonal effect in our data.

#Seasonality
#Let us see how many passengers travelled in flights on a month on month basis. We will plot a stacked annual plot to observe seasonality in our data.

passengers.head()

monthly = pd.pivot_table(data=passengers,values='Passengers',index='Month',columns='Year')
monthly = monthly.reindex(index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
monthly

monthly.plot(figsize=(8,6))
plt.show()

yearly = pd.pivot_table(data=passengers,values='Passengers',index='Year',columns='Month')
yearly = yearly[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
yearly

yearly.plot(figsize=(8,6))
plt.show()

yearly.plot(kind='box',figsize=(8,6))
plt.show()
# Important Inferences
# The passengers are increasing without fail every year.
# July and August are the peak months for passengers.
# We can see a seasonal cycle of 12 months where the mean value of each month starts with a increasing trend in the beginning of the year and drops down towards the end of the year. We can see a seasonal effect with a cycle of 12 months.


############# ARIMA Modelling ########
# The most important assumption of auto regressive method is that the TS data should be stationary.
# There are two primary way to determine whether a given time series is stationary.
# 1.Rolling Statistics
# 2.Augmented Dickey-Fuller Test

# Rolling Statistics 
# Plot the rolling mean and rolling standard deviation. The time series is stationary if they remain constant with time (with the naked eye look to see if the lines are straight and parallel to the x-axis).

df = passengers_p

#eliminating index column
df.reset_index()
df.set_index('Month',inplace=True)
df.head()

#checking for bad data rows
df.tail()

rolling_mean = df.rolling(window = 12).mean()
rolling_std = df.rolling(window = 12).std()
plt.plot(df, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()
# As you can see, the rolling mean and rolling standard deviation increase with time. Therefore, we can conclude that the time series is not stationary.

# Dickey-Fuller Test
# The time series is considered stationary if the p-value is low (according to the null hypothesis) and the critical values at 1%, 5%, 10% confidence intervals are as close as possible to the ADF Statistics

# Perform Dickey-Fuller test:
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Passengers'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
#The ADF Statistic is far from the critical values and the p-value is greater than the threshold (0.05). Thus, we can conclude that the time series is not stationary.

#Taking the log of the dependent variable is as simple way of lowering the rate at which rolling mean increases.
df_log = np.log(df)
plt.plot(df_log)

#Let’s create a function to run the two tests which determine whether a given time series is stationary.
def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries['Passengers'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
        
#There are multiple transformations that we can apply to a time series to render it stationary. For instance, we subtract the rolling mean.
rolling_mean = df_log.rolling(window=12).mean()
df_log_minus_mean = df_log - rolling_mean
df_log_minus_mean.dropna(inplace=True)
get_stationarity(df_log_minus_mean)
#As we can see, after subtracting the mean, the rolling mean and standard deviation are approximately horizontal. The p-value is below the threshold of 0.05 and the ADF Statistic is close to the critical values. Therefore, the time series is stationary.

#Applying exponential decay is another way of transforming a time series such that it is stationary.
rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
df_log_exp_decay = df_log - rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)
get_stationarity(df_log_exp_decay)

# applying time shifting, we subtract every the point by the one that preceded it(1 difference log method).
df_log_shift = df_log - df_log.shift()
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift)

# applying time shifting, we subtract every the point by the six that preceded it.
df_log_shift_6 = df_log - df_log.shift(6)
df_log_shift_6.dropna(inplace=True)
get_stationarity(df_log_shift_6)
df_log_shift_6.dropna(inplace=True)
get_stationarity(df_log_shift_6)

# from all the above rolling statistics plot of 1 difference log method showing better performace. so choosing it


#ARIMA is a combination of 3 parts i.e. AR (AutoRegressive), I (Integrated), and 
#MA (Moving Average). A convenient notation for ARIMA model is ARIMA(p,d,q). Here p,d, and q 
#are the levels for each of the AR, I, and MA parts. Each of these three parts is an effort
# to make the final residuals display a white noise pattern (or no pattern at all). In each 
#step of ARIMA modeling, time series data is passed through these 3 parts


#### Identification of best fit ARIMA model from acf & pacf analysis ###

#Identification of an AR model is often best done with the PACF.
#For an AR model, the theoretical PACF “shuts off” past the order of the model. The phrase “shuts off” means that in theory the partial autocorrelations are equal to 0 beyond that point. Put another way, the number of non-zero partial autocorrelations gives the order of the AR model. By the “order of the model” we mean the most extreme lag of x that is used as a predictor.
#Identification of an MA model is often best done with the ACF rather than the PACF.
#For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner. A clearer pattern for an MA model is in the ACF. The ACF will have non-zero autocorrelations only at lags involved in the model.
#p,d,q p AR model lags d differencing q MA lags

# Time series decomposition plot 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_log_shift.iloc[2:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_log_shift.iloc[2:],lags=40,ax=ax2)
# from the acf & ppacf plot analysis best values for p=1 & q=1. and the d value indicates the differencing degree=1

# we are gonna cross check the p,q & d values through iterative process


#### Identification of best fit ARIMA model from Iterative process ###

#Iterate the process to find the best values for p, d, q 
p=1
q=0
d=1

pdq=[]
aic=[]
for q in range(12):
    try:
        model = ARIMA(df_log_shift, order = (p, d, q)).fit(disp = 0)
        x=model.aic
        x1= p,d,q
        aic.append(x)
        pdq.append(x1)
    except:
        pass
            
keys = pdq
values = aic
d = dict(zip(keys, values))
print (d)

#Best SARIMAX(1,1,7) model - AIC:-173.58099220149512 The best fit model is selected based on Akaike Information Criterion (AIC) , and Bayesian Information Criterion (BIC) values. The idea is to choose a model with minimum AIC and BIC values.

#Predict sales on in-sample date using the best fit ARIMA model
#The next step is to predict passengers for in-sample data and find out how close is the model prediction on the in-sample data to the actual truth.

# checking both the pdq values set got from previous 2 analysis

# compares the ARIMA model buits for pdq values from acf & pacf analysis to the original time series 
model = ARIMA(df_log, order=(1,1,1))
results_acf = model.fit(disp=-1)
plt.plot(df_log_shift)
plt.plot(results_acf.fittedvalues, color='red')

# compares the ARIMA model buits for pdq values from iteration to the original time series 
model = ARIMA(df_log, order=(1,1,7))
results_iter = model.fit(disp=-1)
plt.plot(df_log_shift)
plt.plot(results_iter.fittedvalues, color='red')

# the ARIMA model buits from iteration suits more with thoriginal time series. so take it as final model
# create and fit an ARIMA model with AR of order 1, differencing of order 1 and MA of order 7.

# the original output data response of the ARIMA model(eliminating log & differencing effect)
predictions_ARIMA_diff = pd.Series(results_iter.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(df_log['Passengers'].iloc[0], index=df_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df)
plt.plot(predictions_ARIMA) 

# Given that we have data going for every month going back 12 years and want to forecast the number of passengers for the next 5 years, we use (12 x 8)+ (12 x 5) = 156.
results_iter.plot_predict(1,156)  # in log fisrt difference value range
