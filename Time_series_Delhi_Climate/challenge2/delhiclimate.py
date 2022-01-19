#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings('ignore')
#%%
df_test = pd.read_csv('DailyDelhiClimateTest.csv')
df_train = pd.read_csv('DailyDelhiClimateTrain.csv')

df_train = df_train[df_train.columns[0:2]]
df_test = df_test.iloc[:,0:2]
print(df_train.shape)
print(df_test.shape)

#%%
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])
df_train = df_train.set_index('date')
df_test = df_test.set_index('date')
#%%
#plot the temprature
df_train['meantemp'].plot(figsize=(12,4))

from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_result = seasonal_decompose(df_train['meantemp'])
plt.rcParams['figure.figsize'] = [13, 10]
seasonal_result.plot()
#%%
#Running Arima model
# Using parameters 
# p: the number of lag observations in the model(AR)
# d: the number of times that raw data are differenced, also known as degree of observing (I)
# q: the size of the moving average window, (MA)
index_4_months = pd.date_range(df_train.index[-1], freq = 'D', periods = 114)
# print(index_4_months)
model_arima = ARIMA(df_train, order=(1,2,0)) #stat 2-0-0 and move to the best
model_arima_fit = model_arima.fit()
fcast = model_arima_fit.forecast(114)[0]
#creating a series
fcast = pd.Series(fcast, index= index_4_months)
fcast = fcast.rename('Arima')

#plotting on the graph
fig, ax = plt.subplots(figsize=(15,8))
chart=sns.lineplot(x='date', y='meantemp', data = df_train);
chart.set_title('Delhi Climate')
fcast.plot(ax=ax, color='red', marker='o', legend=True)
df_test.plot(ax=ax, color='blue', marker='o', legend=True)

#printing result
print(fcast.shape)
print('the MSE of Arima is:', mean_squared_error(df_test['meantemp'].values, fcast.values, squared=False))


#-----------------------------------------------------------------------------------
#%%
#finding the best parameters
# p=d=q=range(0,10)
# pdq = list(itertools.product(p,d,q))
# # print(pdq)
# for param in pdq:
#     try:
#         model_arima = ARIMA(df_train, order=param)
#         model_arima_fit = model_arima.fit()
#         print(param, model_arima_fit.aic)
#     except:
#         continue
#%%
# The aic(AIC: Akaike information criterion) is the insample mean squared error predictor in regression 
# We will choose the samllest which is (0,2,8)


model_arima1 = ARIMA(df_train, order=(20,2,8)) #stat 2-0-0 and move to the best
model_arima_fit = model_arima1.fit()
fcast1 = model_arima_fit.forecast(114)[1]
#creating a series
fcast1 = pd.Series(fcast1, index= index_4_months)
fcast1 = fcast1.rename('Arima')
fcast1




#%%
#plotting on the graph
fig, ax = plt.subplots(figsize=(15,8))
chart=sns.lineplot(x='date', y='meantemp', data = df_train)
chart.set_title('Delhi Climate')
fcast1.plot(ax=ax, color='red', marker='o', legend=True)
df_test.plot(ax=ax, color='blue', marker='o', legend=True)
plt.plot(fcast1, color='green', marker='o')
plt.ylim(0, 42)
#printing result
print(fcast1.shape)
print('the MSE of Arima is:', mean_squared_error(df_test['meantemp'].values, fcast1.values, squared=False))


#%%
df_train
# %%
