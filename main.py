import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, ZeroPadding2D

LeNet1 = Sequential()
LeNet1.add(Conv2D(filters=4, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet1.add(AveragePooling2D(pool_size=2, strides=2))
LeNet1.add(Conv2D(filters=12, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet1.add(AveragePooling2D(pool_size=2, strides=2))
LeNet1.add(Flatten())
LeNet1.add(Dense(units=10, activation='softmax'))

LeNet1.build(input_shape=(None, 28, 28, 1))
LeNet1.summary()


LeNet4 = Sequential()
LeNet4.add(ZeroPadding2D(padding=2))
LeNet4.add(Conv2D(filters=4, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet4.add(AveragePooling2D(pool_size=2, strides=2))
LeNet4.add(Conv2D(filters=16, kernel_size=5, padding='valid', strides=1,
                  actvation='tanh'))
LeNet4.add(AveragePooling2D(pool_size=2, strides=2))
LeNet4.add(Flatten())
LeNet4.add(Dense(units=120, activation='tanh'))
LeNet4.add(Dense(units=10, activation='softmax'))

LeNet4.build(input_shape=(None, 28, 28, 1))
LeNet4.summary()


LeNet5 = Sequential()
LeNet5.add(ZeroPadding2D(padding=2))
LeNet5.add(Conv2D(filters=6, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet5.add(AveragePooling2D(pool_size=2, strides=2))
LeNet5.add(Conv2D(filters=16, kernel_size=5, padding='valid', strides=1,
                  actvation='tanh'))
LeNet5.add(AveragePooling2D(pool_size=2, strides=2))
LeNet5.add(Flatten())
LeNet5.add(Dense(units=140, activation='tanh'))
LeNet5.add(Dense(units=84, activation='tanh'))
LeNet5.add(Dense(units=10, activation='softmax'))

LeNet5.build(input_shape=(None, 28, 28, 1))
LeNet5.summary()


