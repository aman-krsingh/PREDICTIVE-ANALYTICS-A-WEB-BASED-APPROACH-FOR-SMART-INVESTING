import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy import array
from tensorflow.keras.models import load_model
model = load_model('predModel_V1.h5')


df = pd.read_csv('data/AAPL.csv')
data = df.reset_index()['4. close']
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(np.array(data).reshape(-1,1))


training_size = int(len(data) * 0.40)
test_size = len(data) - training_size

test_data = data[training_size:len(data), :1]
X_input = test_data[len(test_data)-100:].reshape(1,-1)


temp_input = list(X_input)
temp_input = temp_input[0].tolist()
print(temp_input)


#future prediction

days =30

lst_output=[]
n_steps=100
i=0

while(i<days):
    if(len(temp_input) > 100):
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


days_new=np.arange(1, 101)
# days_pred =np.arange(101, length)
days_pred =np.arange(101, 101 + len(lst_output))

ld=len(data)
#print(ld)
L = ld-100
#print(L)

new_data=data.tolist()
new_data.extend(lst_output)
plt.plot(days_new,scaler.inverse_transform(data[L:]))
plt.plot(days_pred,scaler.inverse_transform(lst_output))
#plt.plot(days_new,scaler.inverse_transform(data[L:]))

plt.show()

new_data=data.tolist()
new_data.extend(lst_output)
plt.plot(new_data[6000:])
plt.show()

new_data = scaler.inverse_transform(new_data).tolist()
plt.plot(new_data)
plt.show()




################## boom boom ###############


# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
# from numpy import array

# from tensorflow.keras.models import load_model
# model = load_model('predModel_V1.h5')


# df = pd.read_csv('data/AAPL.csv')
# data = df.reset_index()['4. close']

# data = data.iloc[:5758]
# data_oneyear=data.iloc[5659:]
# plt.plot(data_oneyear)
# plt.show()


scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(np.array(data).reshape(-1,1))


training_size = int(len(data) * 0.40)
test_size = len(data) - training_size

test_data = data[training_size:len(data), :1]
X_input = test_data[len(test_data)-100:].reshape(1,-1)


temp_input = list(X_input)
temp_input = temp_input[0].tolist()
print(temp_input)


#future prediction

days =30

lst_output=[]
n_steps=100
i=0

while(i<days):
    if(len(temp_input) > 100):
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


days_new=np.arange(1, 101)
# days_pred =np.arange(101, length)
days_pred =np.arange(101, 101 + len(lst_output))

ld=len(data)
#print(ld)
L = ld-100
#print(L)

new_data=data.tolist()
new_data.extend(lst_output)
plt.plot(days_new,scaler.inverse_transform(data[L:]))
plt.plot(days_pred,scaler.inverse_transform(lst_output))
#plt.plot(days_new,scaler.inverse_transform(data[L:]))

plt.show()

new_data=data.tolist()
new_data.extend(lst_output)
plt.plot(new_data[4500:], color='green')
plt.plot(data_oneyear, color='red')
plt.show()

new_data = scaler.inverse_transform(new_data).tolist()
plt.plot(new_data, color='green')
plt.plot(data_oneyear, color='red')
plt.show()