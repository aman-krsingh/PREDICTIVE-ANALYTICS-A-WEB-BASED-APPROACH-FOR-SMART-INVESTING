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

#API_key = 'WSKF50ODKWY4WP1O'
#ts = TimeSeries(key= API_key, output_format='pandas')

#res = ts.get_daily(ticker, outputsize='full')
data = pd.read_csv('AAPL_1.csv')

#df=df[0]
#data.to_csv('AAPL_V2.csv')
data = data.reset_index()['4. close']
# data.to_csv('codedata.csv')

backup_data = data
#print(data)
# data.to_csv('AAPL.csv')


#print (data)
# plt.plot(data)
# plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(np.array(data).reshape(-1,1))
# print(data)
# print(len(data))


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

time_step =5
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# print(X_train.shape), print(Y_train.shape)
# print(Y_train)

# making data ready for LSTM model reshaping for value to give input in model.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#good to go to create model.

model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

#model.summary()

model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs=150,batch_size=64,verbose=1)
#model.save('predModel_V1.h5')

# #prediction for test

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
#print("10,25:")
print (math.sqrt(mean_squared_error(Y_train, train_predict)))
print(math.sqrt(mean_squared_error(Y_test,test_predict)))


#plotting

look_back=time_step

#shift train
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

#shift test
testpredicrPlot = np.empty_like(data)
testpredicrPlot[:, :] = np.nan
testpredicrPlot[len(train_predict) + (look_back*2)+1: len(data)-1, :] =test_predict

#plot baseline and predictions

#original data
# plt.plot(scaler.inverse_transform(data), color = 'blue')
# #prediction on traning data
# plt.plot(trainPredictPlot, color = 'red')
# #prediction on testing data
# plt.plot(testpredicrPlot, color ="green")

# plt.show()



#future prediction


l= len(test_data)
#print(l)
length = l-time_step
X_input = test_data[length:].reshape(1,-1)

print(X_input.shape)


temp_input = list(X_input)

temp_input = temp_input[0].tolist()

#print(temp_input)


days =30

lst_output=[]
n_steps=time_step
i=0

while(i<days):
    if(len(temp_input) > time_step):
        #print(temp_input)
        X_input=np.array(temp_input[1:])
        #print("{} days input{}".format(i,X_input))
        X_input=X_input.reshape(1,-1)
        X_input = X_input.reshape((1, n_steps, 1))
        #print(X_input)
        yhat = model.predict(X_input, verbose=0)
        #print("{} day output{}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        X_input=X_input.reshape((1, n_steps,1))
        yhat = model.predict(X_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
#print(lst_output)

time_step_plus1= time_step+1
#number from 1 to time_step + 1
days_new=np.arange(1, time_step_plus1)

# days_pred =np.arange(101, length)
days_pred =np.arange(time_step_plus1, time_step_plus1 + len(lst_output))

ld=len(data)
#print(ld)
L = ld-time_step
#print(L)

new_data=data.tolist()
new_data.extend(lst_output)

# days_new=scaler.inverse_transform(data[L:])
# days_pred=scaler.inverse_transform(lst_output)
# plt.plot(days_new, color ="red")
# plt.plot(days_pred, color = "green")

# plt.plot(days_new,scaler.inverse_transform(data[L:]), color ="red")
plt.plot(days_pred,scaler.inverse_transform(lst_output), color = "green")
# plt.plot(days_new,scaler.inverse_transform(data[L:]), color ="red")
plt.show()

new_data = scaler.inverse_transform(new_data).tolist()
plt.plot(backup_data, color='black')
plt.plot(days_pred,scaler.inverse_transform(lst_output), color = "green")
plt.show()

new_data=data.tolist()
new_data.extend(lst_output)
# plt.plot(new_data)
# plt.show()

new_data = scaler.inverse_transform(new_data).tolist()
plt.plot(backup_data, color='black')
plt.plot(new_data, color='green')
plt.show()

