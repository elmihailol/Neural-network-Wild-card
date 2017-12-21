import numpy
from keras import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# Create dataset, where X = [n-look_back, n-look_back+1, ...., n-1] and Y = [n]
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


# Make reccurent predictions based on some model
def extended_this(model, trainX, trainY, look_back, multi=2):
    # Create dataset in comfortable type
    X = []
    Y = []
    for i in range(len(trainX)):
        X.append(trainX[i])
    for i in range(len(trainY)):
        Y.append(trainY[i])
    # Make len(X) * multi predictions
    for i in range(int(len(X) * multi)):
        # Show every 100 iters
        if i % 100 == 0:
            print(i)
        # Get last element in X. last
        last_x = numpy.array([X[len(X) - 1]])
        # Make prediction based on last_x
        last_y = model.predict(last_x)
        # Create list which will contains new X = [n-look_back, n-look_back+1, ...., n-1]
        merge = []
        # Fill that new X
        for a in range(look_back - 1, 0, -1):
            merge.append(Y[i - a])
        # Add new prediction to new X
        merge.extend([last_y[0]])
        # Add new data to datasets
        Y.append(last_y[0])
        X.append(numpy.array(merge))
    # Return new datalist
    return Y


def example_LSTM():
    data = []
    for i in range(300):
        data.append([numpy.sin(i * 0.1), 0, numpy.cos(i * 0.1)])

    print(data)
    look_back = 20
    trainX, trainY = create_dataset(data, look_back)

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
    Y = extended_this(model=model, trainX=trainX[0:new_len], trainY=trainY[0:new_len], look_back=look_back,
                      multi=3)
    plt.plot(Y)
    plt.plot(trainY)
    plt.show()
