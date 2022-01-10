#%%
import streamlit as st
# st.set_page_config(layout='wide')
from datetime import datetime, timedelta
import pandas_datareader as pds
import pandas as pd
import numpy as np
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf


START = "2018-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

st.title('CryptoCurrency Stock Forecast App with LSTM model')
col1, col2 = st.columns(2)
stocks = ('BTC', 'XRP', 'ETH')
against_currency1 = ('USD', 'EUR', 'GBP', 'INR')
selected_stock = col1.selectbox('Select dataset for prediction', stocks)
against_currency = col2.selectbox('Select currency to compare with', against_currency1)





@st.cache
def load_data(ticker):
	data = pds.DataReader(ticker, 'yahoo', START, TODAY)
	data.reset_index(inplace=True)
	data['Date'] = pd.to_datetime(data['Date'])
	data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    
	return data

# Loading the data in dataframe	
data_load_state = st.text('Loading data...')
data = load_data(f"{selected_stock}-{against_currency}")
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.dataframe(data.tail(7), width=1200)
st.write('')

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig, use_container_width=True)	
plot_raw_data()

#%%
# Preprocessing the data
df = data['Close']
df1 = data[['Date','Close']]
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(np.array(df).reshape(-1,1))

# Train and Test split
training_size = int(len(df) * 0.70)
test_size = len(df) - training_size
train_data, test_data = df[0:training_size, :], df[training_size:len(df), :1]



# Create dataset for prediction

time_step = 100
def create_dataset(dataset, time_step=100):
	dataX, dataY = [], []
	for i in range(len(dataset) - time_step - 1):
		a = dataset[i:(i+time_step), 0]
		dataX.append(a)
		dataY.append(dataset[i+ time_step, 0])
	return np.array(dataX), np.array(dataY)

# Creating dataset
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshaping the data to pass to LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
data_load_state1 = st.text('Running LSTM model, please be patient while we predict ...')
try:
    model = tf.keras.models.load_model('saved_model')
except:
# Creating a LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape = (100,1)))
    model.add(LSTM(50, return_sequences=True ))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer='adam')
    #Fittingthe model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64, verbose=1)
    # Saving the model
    model.save('saved_model')

# Performing the prediction
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transforming the data back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance metrics
math.sqrt(mean_squared_error(y_train, train_predict))
math.sqrt(mean_squared_error(y_test, test_predict))

# Show and plot train & test data
st.subheader('Prediction')


print(len(test_data))
x_input=test_data[len(test_data)-100:].reshape(1,-1)
print(x_input.shape)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
print(len(temp_input))


#%%


n_years = st.slider('Days of prediction:', 1, 50,1)
lst_output=[]
n_steps=100
i=0
while(i<n_years):
    
    if(len(temp_input)>100):
        
        x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input).shape
        yhat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
    
#%%
# lst_output
test_data = scaler.inverse_transform(lst_output) 
flat_ls = []
for i in test_data:
    for j in i:
        flat_ls.append(j)
flat_ls = pd.DataFrame(flat_ls, columns=['Close'])


# %%
r = []
for x in range(len(flat_ls)):
    r.append(pd.DatetimeIndex([data['Date'].iloc[-1]]) + pd.DateOffset(x+1))
    if len(r) == len(flat_ls):
        break

print(r)
flat_ls['Date'] = pd.DataFrame(r)
flat_ls['Date'] = pd.to_datetime(flat_ls['Date'])
flat_ls["Date"] = flat_ls["Date"].dt.strftime("%Y-%m-%d")
#%%
flat_ls
data_load_state1.text('Loading data... done!')
# %%

def plot_result():
	fig1 = go.Figure()
	fig1.add_trace(go.Scatter(x=flat_ls['Date'], y= flat_ls['Close'], name="predicted data"))
	fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig1.layout.update(title_text='Predicted Data with range slider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig1, use_container_width=True)	
plot_result()
# %%
data_load_state2 = st.text('loading the chart...')
data_load_state2.text('Loading chart... done!')

st.write('created by Murtaza')