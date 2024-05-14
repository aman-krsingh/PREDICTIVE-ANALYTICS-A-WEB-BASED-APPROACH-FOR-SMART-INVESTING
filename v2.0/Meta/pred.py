import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy import array
from tensorflow.keras.models import load_model

ticker='META'

model = load_model(f'predModel_{ticker}.h5')

data = pd.read_csv(f'./data/{ticker}.csv')

size = len(data)
year = 365 * 5
df = data[size - year:]

data = df.reset_index()['Close']

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(np.array(data).reshape(-1,1))

#splitting data.
training_size = int(len(data) * 0.60)
test_size = len(data) - training_size

test_data = data[training_size:len(data), :1]

time_step =5

#future prediction

l= len(test_data)
length = l-time_step

X_input = test_data[length:].reshape(1,-1)

#print(X_input.shape)

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


time_step_plus1 = time_step + 1

#number from 1 to time_step + 1
days_new=np.arange(1, time_step_plus1)



#number from time_step +1 to 30 more (6 to 35 [30 numbers])
days_pred =np.arange(time_step_plus1, time_step_plus1 + len(lst_output))

ld=len(data)
L = ld-time_step

new_data=data.tolist()
new_data.extend(lst_output)


#   last 5 days data.

# plt.plot(days_new,scaler.inverse_transform(data[L:]), color ="red")
# days_new=scaler.inverse_transform(data[L:])
# print(days_new)

#   future 30days predicted value.
days_pred = scaler.inverse_transform(lst_output)
plt.plot(days_pred, color = "red")
plt.show()


new_data=data.tolist()
new_data.extend(lst_output)
new_data = scaler.inverse_transform(new_data).tolist()
data = scaler.inverse_transform(data)

plt.plot(new_data, color='red')
plt.plot(data, color='blue')
plt.show()
