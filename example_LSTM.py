import numpy
from keras import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import WildCardLib

data = []
for i in range(300):
    data.append([numpy.sin(i*0.1),0,numpy.cos(i*0.1)])

print(data)
look_back = 20
trainX, trainY = WildCardLib.create_dataset(data, look_back)

print(trainX)
model = Sequential()
model.add(LSTM(20, input_shape=(look_back, 3)))
# model.add(Dense(100, input_dim=3, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)
trainPredict = model.predict(trainX)
new_len = int(len(trainX)*1)
Y = WildCardLib.extended_this(model=model, trainX=trainX[0:new_len], trainY=trainY[0:new_len],look_back=look_back, multi=1.5)
# plt.plot(trainPredict)

plt.plot(Y)
plt.plot(trainY)
plt.show()