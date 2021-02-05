import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import tensorflow as tf

a = plt.imread('./edge_detection_ex.jpg')
a.astype(np.float)
img = a.reshape(1,720,1280,3)
img = tf.constant(img, dtype=tf.float64)

print(img.shape)
def make_filter():
    weight = np.array([[[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]],

                       [[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]],

                       [[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]]])
    weight = weight.reshape((1,3,3,3))
    print("weight.shape", weight.shape)
    weight_init = tf.constant_initializer(weight)
    print(type(weight_init))
    return weight_init
def cnn_valid(image, weight, option):
    conv2d = keras.layers.Conv2D(filters=1, kernel_size=3, padding=option, kernel_initializer=weight)(image)
    plt.imshow(conv2d.numpy().reshape(720,1280), cmap='gray')
    plt.show()
def main(img) :
    filter1 = make_filter()
    cnn_valid(img, filter1, 'SAME')
main(img)