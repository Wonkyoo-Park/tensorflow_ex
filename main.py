<<<<<<< Updated upstream
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


=======
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, ZeroPadding2D


class LeNet1(Model):
    def __init__(self):
        super(LeNet1, self).__init__()

        # feature extractor
        self.conv1 = Conv2D(filters=4, kernel_size=5, padding='valid', strides=1,
                            activation='tanh')
        self.conv1_pool = AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = Conv2D(filters=12, kernel_size=5, padding='valid', strides=1,
                            activation='tanh')
        self.conv2_pool = AveragePooling2D(pool_size=2, strides=2)

        # claassifier
        self.flatten = Flatten()
        self.dense1 = Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)

        return x


class LeNet4(Model):
    def __init__(self):
        super(LeNet4, self).__init__()

        # feature extractor
        self.zero_padding = ZeroPadding2D(padding=2)
        self.conv1 = Conv2D(filters=4, kernel_size=5, padding='valid', strides=1,
                            activation='tanh')
        self.conv1_pool = AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = Conv2D(filters=16, kernel_size=5, padding='valid', strides=1,
                            activation='tanh')
        self.conv2_pool = AveragePooling2D(pool_size=2, strides=2)

        # claassifier
        self.flatten = Flatten()
        self.dense1 = Dense(units=120, activation='tanh')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.zero_padding(x)
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()

        # feature extractor
        self.zero_padding = ZeroPadding2D(padding=2)
        self.conv1 = Conv2D(filters=6, kernel_size=5, padding='valid', strides=1,
                            activation='tanh')
        self.conv1_pool = AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = Conv2D(filters=16, kernel_size=5, padding='valid', strides=1,
                            activation='tanh')
        self.conv2_pool = AveragePooling2D(pool_size=2, strides=2)

        # claassifier
        self.flatten = Flatten()
        self.dense1 = Dense(units=140, activation='tanh')
        self.dense2 = Dense(units=84, activation='tanh')
        self.dense3 = Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.zero_padding(x)
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

>>>>>>> Stashed changes
