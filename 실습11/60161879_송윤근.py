import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import imdb
import pandas as pd
df_price = pd.read_csv('samsung.csv')
seq_len = 50 #window 값이 50
sequence_length = seq_len + 1
high_prices = df_price['고가'].values
low_prices = df_price['저가'].values
mid_prices = (high_prices + low_prices)/2
result = []

for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index:index + sequence_length])

normalized_data = []

for window in result:
    normalized_window = [[(float(p) / float(window[0]))-1]for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

x_train = result[:8000,:-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = result[:8000, -1]

x_test = result[8000:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
y_test = result[8000:, -1]
print(x_train.shape,y_train.shape)
model = Sequential()
model.add(LSTM(50,input_shape = (50,1),activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=15,callbacks=[es, mc], batch_size=60, validation_split=0.2)
y_hat = model.predict(x_test)
print(y_hat)

plt.plot(y_test, label='test')
plt.plot(y_hat, label = 'pred')
plt.legend()
plt.show()