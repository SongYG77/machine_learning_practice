import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
class_names = ["0","1","2","3","4","5","6","7","8","9"]
def makemodel(weight_init,X_train, y_train, X_valid, y_valid):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(300, weight_init, activation="relu"))
    model.add(keras.layers.Dense(100,weight_init, activation="relu"))
    model.add(keras.layers.Dense(10,weight_init, activation="softmax"))
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    return model

def makemodel_drop(weight_init,X_train, y_train, X_valid, y_valid):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(300, weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(100,weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(10, weight_init,activation="softmax"))
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    return model
def data_normalization(X_train_full, y_train_full):
    X_valid, X_train = X_train_full[:200] , X_train_full[200:]
    y_valid, y_train = y_train_full[:200], y_train_full[200:]
    return X_valid, X_train, y_valid, y_train
def printmod(model, x_test,y_test) :
    model.evaluate(x_test, y_test)
    x_new = x_test[:10]
    y_proba = model.predict(x_new)
    plt.figure(figsize=(10 * 1.2, 10 * 1.2))
    for i in range(10):
        pic = x_test[i].reshape(8,8)
        plt.subplot(1, 10, i+1)
        plt.imshow(pic, cmap="binary", interpolation="nearest")
        plt.axis('off')
        yindex = list(y_proba[i]).index(y_proba[i].max())
        print(yindex)
        plt.title(class_names[y_test[i]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()
def modelpredict(model, X_train, y_train, X_valid, y_valid):
    # 시간 측정
    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    start = time.time()
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid), callbacks=[tb_hist])
    print("time :", time.time() - start)
    return history
def plot_history(histories, key='accuracy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()
def main():
    digits = load_digits()
    x_data = digits.data
    print(x_data[0].shape)
    y_data = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    x_valid, x_train, y_valid, y_train = data_normalization(x_train, y_train)
    model = makemodel('he_normal',x_train, y_train, x_valid, y_valid)
    dropmodel = makemodel_drop('he_normal',x_train, y_train, x_valid, y_valid)
    hist_he = modelpredict(model, x_train, y_train, x_test, y_test)
    hist_he_drop = modelpredict(dropmodel, x_train, y_train, x_test, y_test)
    plot_history([('Normal', hist_he), ('Dropout', hist_he_drop)])
main()
