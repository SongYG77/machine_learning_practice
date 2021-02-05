import matplotlib.pyplot as plt
import keras
import numpy as np
import tensorflow as tf
def make_toyimg():
    image = np.array([[1.],[2.],[3.],
    [4.],[5.],[6.],
    [7.],[8.],[9.]])
    image.astype(np.float)
    print(image.shape)
    plt.imshow(image.reshape((-1,3)), cmap='Greys')
    plt.show()
    # data format should be change to batch_shape + [height, width, channels].
    image = image.reshape((1, 3, 3, 1))
    image = tf.constant(image, dtype=tf.float64)
    return image
def make_toyfilter():
    weight = np.array([[1.],[1.],
    [1.],[1.]])
    weight=weight.reshape((1,2,2,1))
    print(weight.shape)
    print("weight.shape", weight.shape)
    weight_init = tf.constant_initializer(weight)
    return weight_init
def cnn_valid(image, weight, option, sizeshape):
    conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding=option,
    kernel_initializer=weight)(image)
    print("conv2d.shape", conv2d.shape)
    print(conv2d.numpy().reshape(sizeshape,sizeshape))
    plt.imshow(conv2d.numpy().reshape(sizeshape,sizeshape), cmap='gray')
    plt.show()
def main():
    img = make_toyimg()
    filter = make_toyfilter()
    cnn_valid(img, filter, 'VALID',2)
    cnn_valid(img, filter, 'SAME',3)
main()