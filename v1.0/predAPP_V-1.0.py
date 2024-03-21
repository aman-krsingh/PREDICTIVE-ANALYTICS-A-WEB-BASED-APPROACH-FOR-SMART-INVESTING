from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf

import math 
from sklearn.metrics import mean_squared_error

from numpy import array

#ticker='GOOG'
#ticker='AAPL'


#API_key = 'DYSPWV4O3FEDK2NC'

API_key = 'WSKF50ODKWY4WP1O'
ts = TimeSeries(key= API_key, output_format='pandas')

#res = ts.get_daily(ticker, outputsize='full')


#df=res[0]
#df.to_csv('AAPL.csv')
#data = df.reset_index()['4. close']


df = pd.read_csv('AAPL.csv')
data = df.reset_index()['4. close']


#print (data)
# plt.plot(data)
# plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(np.array(data).reshape(-1,1))
#print(data)

#splitting data.
training_size = int(len(data) * 0.40)
test_size = len(data) - training_size

train_data, test_data = data[0:training_size,:], data[training_size:len(data), :1]

#print(training_size, test_size)

#array into matrix
## creating function for creating dataset for train and test.

def create_dataset(dataset, time_step = 1):
    dataX, dataY =[], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step =100
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

#print(X_train.shape), print(Y_train.shape)
#print(Y_train)

# making data ready for LSTM model reshaping for value to give input in model.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#good to go to create model.

model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

#model.summary()

model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs=25,batch_size=64,verbose=1)
model.save('predModel_V1.h5')



##################################### testing ############################################

#prediction for test

#from tensorflow.keras.models import load_model
#model = load_model('predModel_V1.h5')

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

#print (math.sqrt(mean_squared_error(Y_train, train_predict)))
#print(math.sqrt(mean_squared_error(Y_test,test_predict)))

#plotting

#shift train
look_back=100
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

#shift test
testpredicrPlot = np.empty_like(data)
testpredicrPlot[:, :] = np.nan
testpredicrPlot[len(train_predict) + (look_back*2)+1: len(data)-1, :] =test_predict

#plot baseline and predictions

# plt.plot(scaler.inverse_transform(data))
# plt.plot(trainPredictPlot)
# plt.plot(testpredicrPlot)
# plt.show()

l= len(test_data)
#print(l)
length = l-100
X_input = test_data[length:].reshape(1,-1)

#print(X_input.shape)


temp_input = list(X_input)

temp_input = temp_input[0].tolist()

print(temp_input)
