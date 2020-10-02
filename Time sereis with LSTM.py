# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import optimizers
import json
import math

# define your dataset or you can import one using pandas
X = array([[0,0,0], [0,0,0], [0,0,1], [1,1,2],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,5,5],[28,30,31],[34,39,43],[43,56,62],[73,82,102],[113,119,142],[156,194,244]])
y = array([0,1,2,3,3,3,3,3,3,3,3,3,5,31,43,62,102,142,244,300])

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# define model
lr = 0.001
adam = optimizers.Adam(lr)
filepath=r"C:\Users\OneDrive - UiPath\Desktop\covid\model.h5"
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=1000,verbose=0)

# demonstrate prediction
x_input = array([310,370,450])
x_input = x_input.reshape((1, 3, 1))
op = model.predict(x_input, verbose=0)
op=math.ceil(op)
print("predicted no of cases:"+str(op))
model_json = model.to_json()

with open(r"C:\Users\OneDrive - UiPath\Desktop\covid\model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights(filepath)
print("Saved model to disk")
