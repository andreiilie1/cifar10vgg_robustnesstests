from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow.keras.utils

def load_data(name = "cifar10"):
    if(name == "cifar10"):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
        y_test = tensorflow.keras.utils.to_categorical(y_test, 10)
    else:
        raise Exception("Invalid data name")
        
    return (x_train, y_train), (x_test, y_test)