##############    plastic     ##########
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
plastic_p = pd.read_csv('C://Users//user//Downloads//forecasting//PlasticSales.csv')

plastic = plastic_p.copy(deep= True)
plastic.head()

# converting date column to date time format if it is in  string format
plastic.info() # date column ds is already in date tme format
dates = pd.date_range(start='1949-01-01', freq='MS',periods=len(plastic))  # converting to date time format

#split to month and year column
plastic['Month'] = dates.month
plastic['Year'] = dates.year

plastic.head()

#To get the names of the month
plastic.dtypes
plastic.head()

import calendar
plastic['Month'] = plastic['Month'].apply(lambda x: calendar.month_abbr[x])
plastic.rename({'#Passengers':'Passengers'},axis=1,inplace=True)
plastic = plastic[['Month','Year','Sales']]

plastic.head()

# adding date column
plastic['Date'] = dates
plastic.set_index('Date',inplace=True)
plastic.head()

# Exploratory Data Analysis
plt.figure(figsize=(10,8))
plastic.groupby('Year')['Sales'].mean().plot(kind='bar')
plt.show()
# From the above figure we can see that Sales are increasing with the increase in the year

plt.figure(figsize=(10,8))
plastic.groupby('Month')['Sales'].mean().reindex(index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']).plot(kind='bar')
plt.show()
# From the above figure we can see that more passengers can be seen between months June to October

#Lets plot the data to see the trend and seasonality
plastic_count = plastic['Sales']

plt.figure(figsize=(10,8))
plastic_count.plot()
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()

#Now we start with time series decomposition of this data to understand underlying patterns such as trend, seasonality, cycle and irregular remainder
decompose = sm.tsa.seasonal_decompose(plastic_count,model='additive',extrapolate_trend=8)

fig = decompose.plot()
fig.set_figheight(10)
fig.set_figwidth(8)
fig.suptitle('Decomposition of Time Series')
#Trend
#Time Series Decomposition: To begin with let's try to decipher trends embedded in the above tractor sales time series. It is clearly evident that there is an overall increasing trend in the data along with some seasonal variations. However, it might not always be possible to make such visual inferences.
#So, more formally, we can check stationarity using the following: Plotting Rolling Statistics: We can plot the moving average or moving variance and see if it varies with time. By moving average/variance we mean that at any instant 't', we'll take the average/variance of the last year, i.e. last 12 months. But again this is more of a visual technique.
#Now, let’s try to remove wrinkles from our time series using moving average. We will take moving average of different time periods i.e. 4,6,8, and 12 months as shown below. Here, moving average is shown in orange and actual series in blue.

# Centering moving average for the time series
plastic.Sales.plot(label = "org")
for i in range(4, 13, 2):
    plastic['Sales'].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
# As we could see in the above plots, 12-month moving average could produce a wrinkle free curve as desired. This on some level is expected since we are using month-wise data for our analysis and there is expected monthly-seasonal effect in our data.

#Seasonality
#Let us see how many passengers travelled in flights on a month on month basis. We will plot a stacked annual plot to observe seasonality in our data.

plastic.head()

monthly = pd.pivot_table(data=plastic,values='Sales',index='Month',columns='Year')
monthly = monthly.reindex(index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
monthly

monthly.plot(figsize=(8,6))
plt.show()

yearly = pd.pivot_table(data=plastic,values='Sales',index='Year',columns='Month')
yearly = yearly[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
yearly

yearly.plot(figsize=(8,6))
plt.show()

yearly.plot(kind='box',figsize=(8,6))
plt.show()
# Important Inferences
# The sales are increasing without fail every year.
# July and october are the peak months for sales.
# We can see a seasonal cycle of 12 months where the mean value of each month starts with a increasing trend in the beginning of the year and drops down towards the end of the year. We can see a seasonal effect with a cycle of 12 months.


############# ARIMA Modelling ########
# The most important assumption of auto regressive method is that the TS data should be stationary.
# There are two primary way to determine whether a given time series is stationary.
# 1.Rolling Statistics
# 2.Augmented Dickey-Fuller Test

# Rolling Statistics 
# Plot the rolling mean and rolling standard deviation. The time series is stationary if they remain constant with time (with the naked eye look to see if the lines are straight and parallel to the x-axis).

df = plastic[["Sales"]]

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
# As you can see, the rolling mean increase with time. Therefore, we can conclude that the time series is not stationary.

# Dickey-Fuller Test
# The time series is considered stationary if the p-value is low (according to the null hypothesis) and the critical values at 1%, 5%, 10% confidence intervals are as close as possible to the ADF Statistics

# Perform Dickey-Fuller test:
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Sales'])
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
    result = adfuller(timeseries['Sales'])
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
#As we can see, after subtracting the mean, the rolling mean and standard deviation are approximately horizontal. BUt p-value is below the threshold of 0.05 and the ADF Statistic is close to the critical values. Therefore, the time series is stationary.

#Applying exponential decay is another way of transforming a time series such that it is stationary.
rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
df_log_exp_decay = df_log - rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)
get_stationarity(df_log_exp_decay)
#As we can see, after subtracting the mean, the rolling mean and standard deviation are not horizontal. BUt p-value is below the threshold of 0.05 and the ADF Statistic is close to the critical values. Therefore, the time series is not perfectly stationary.

# applying time shifting, we subtract every the point by the one that preceded it(1 difference log method).
df_log_shift = df_log - df_log.shift()
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift)
# As we can see, after subtracting the mean, the rolling mean and standard deviation are horizontal.So taking time series is as stationary.

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
fig = sm.graphics.tsa.plot_pacf(df_log_shift.iloc[2:],lags=25,ax=ax2)
# from the acf & ppacf plot analysis best values for p=1 & q=1. and the d value indicates the differencing degree=1


# ARIMA model builts for pdq values from acf & pacf analysis to the original time series 
model = ARIMA(df_log, order=(1,1,1))
results = model.fit()
plt.plot(df_log_shift)
plt.plot(results.fittedvalues, color='red')


# the original output data response of the ARIMA model(eliminating log & differencing effect)
predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(df_log['Sales'].iloc[0], index=df_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df)
plt.plot(predictions_ARIMA) 

# Given that we have data going for every month going back 12 years and want to forecast the number of passengers for the next 1 years, we use (12 x 5)+ (12 x 1) = 72.
results.plot_predict(1,72)  # in log fisrt difference value range

