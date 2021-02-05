import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras.utils import to_categorical
import time

X = [[0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5],
    [0.4, 0.5, 0.6]]
Y= [[0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5],
    [0.4, 0.5, 0.6],
    [0.5, 0.6, 0.7]]

X = np.array(X).reshape(4, 3, 1)
Y = np.array(Y).reshape(4, 3)

def simple(X,Y) :
    print('=================== Simple ====================')
    model = Sequential()
    model.add(SimpleRNN(50,  return_sequences=False, input_shape=(3,1)))
    model.add(Dense(3))
    model.summary()
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X,Y,epochs=200,  verbose=2)
    X_test = np.array([[[0.5, 0.6, 0.7],
                  [0.6, 0.7, 0.8]]]).reshape(2,3,1)
    print(model.predict(X_test))


def deep(X, Y):
    print('\n\n\n')
    print('=================== Deep ====================')
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(3, 1), return_sequences=True))
    model.add(SimpleRNN(50, return_sequences=True))
    model.add(SimpleRNN(50, return_sequences=False))
    model.add(Dense(3))
    model.summary()
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X, Y, epochs=200, verbose=2)
    X_test = np.array([[[0.5, 0.6, 0.7],
                        [0.6, 0.7, 0.8]]]).reshape(2, 3, 1)
    print(model.predict(X_test))


simple(X,Y)
deep(X,Y)
