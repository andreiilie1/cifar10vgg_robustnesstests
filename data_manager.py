import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
import numpy as np
import tensorflow.keras.utils

def load_data(name = "cifar10"):
    if(name == "cifar10"):
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    elif(name == "mnist"):
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = tf.expand_dims(x_train, 3)
        x_test = tf.expand_dims(x_test, 3)

        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    else:
        raise Exception("Invalid data name")
        
    return (x_train, y_train), (x_test, y_test)