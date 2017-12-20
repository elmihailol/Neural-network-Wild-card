import numpy
from keras import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


def extended_this(model, trainX, trainY, look_back, multi=2):
    X = []
    Y = []
    for i in range(len(trainX)):
        X.append(trainX[i])
    for i in range(len(trainY)):
        Y.append(trainY[i])
    last_X =[trainX[len(trainX)-1]]
    for i in range(int(len(X) * multi)):
        print(X[0])
        buf = numpy.array(X[len(X)-1])
        print(buf)
        newX = numpy.array([buf])
        last_y = model.predict(newX)
        # newX = numpy.array([trainX[len(trainX)-1]])
        # last_y = model.predict(newX)
        merge = []
        for a in range(look_back - 1, 0, -1):
            merge.append(Y[i - a])
        merge.extend([last_y[0]])
        Y.append(last_y[0])
        print(numpy.array(merge))
        X.append(numpy.array(merge))
    return Y

def example_LSTM():
    data = []
    for i in range(300):
        data.append([numpy.sin(i * 0.1), 0, numpy.cos(i * 0.1)])

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
    new_len = int(len(trainX) * 1)
    Y = WildCardLib.extended_this(model=model, trainX=trainX[0:new_len], trainY=trainY[0:new_len], look_back=look_back,
                                  multi=1.5)
    # plt.plot(trainPredict)

    plt.plot(Y)
    plt.plot(trainY)
    plt.show()