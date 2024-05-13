import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math 
from numpy import array

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf

from sklearn.metrics import mean_squared_error



data = pd.read_csv('./data/AAPL.csv')

size = len(data)
year = 365 * 5
df = data[size - year:]

data = df.reset_index()['Close']

#data.to_csv('./data/AAPL_5yrs.csv')

#print (data)
plt.plot(data)
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(np.array(data).reshape(-1,1))
# print(data)


#splitting data.
training_size = int(len(data) * 0.40)
test_size = len(data) - training_size

train_data, test_data = data[0:training_size,:], data[training_size:len(data), :1]
#print(training_size, test_size)


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

#good to go for createing model.

model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

#model.summary()

#traning the model and saving it.
model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs=150,batch_size=64,verbose=1)
model.save('predModel_V2.h5')


#prediction for test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

#RSME value. (lesser the value better the model is)
print (math.sqrt(mean_squared_error(Y_train, train_predict)))
print(math.sqrt(mean_squared_error(Y_test,test_predict)))


#prepraing for visual repersation

look_back=time_step

#shift train data
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

#shift test data
testpredicrPlot = np.empty_like(data)
testpredicrPlot[:, :] = np.nan
testpredicrPlot[len(train_predict) + (look_back*2)+1: len(data)-1, :] =test_predict

#plot original data and predictions

#original data
plt.plot(scaler.inverse_transform(data), color = 'blue')

#prediction on traning data
plt.plot(trainPredictPlot, color = 'red')

#prediction on testing data
plt.plot(testpredicrPlot, color ="green")

plt.show()


