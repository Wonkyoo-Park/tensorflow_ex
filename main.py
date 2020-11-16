import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, ZeroPadding2D


class LeNet1(Model):
    def __init__(self):
        super(LeNet1, self).__init__()

        # feature extractor
        self.feature_extractor = Sequential()
        self.feature_extractor.add(Conv2D(filters=4, kernel_size=5, padding='valid', strides=1,
                                   activation='tanh'))
        self.feature_extractor.add(AveragePooling2D(pool_size=2, strides=2))
        self.feature_extractor.add(Conv2D(filters=12, kernel_size=5, padding='valid', strides=1,
                                   activation='tanh'))
        self.feature_extractor.add(AveragePooling2D(pool_size=2, strides=2))

        # classifier
        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=10, activation='softmax'))

    def call(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x


class LeNet4(Model):
    def __init__(self):
        super(LeNet4, self).__init__()

        # feature extractor
        self.feature_extractor = Sequential()
        self.feature_extractor.add(Conv2D(filters=4, kernel_size=5, padding='valid', strides=1,
                                          activation='tanh'))
        self.feature_extractor.add(AveragePooling2D(pool_size=2, strides=2))
        self.feature_extractor.add(Conv2D(filters=16, kernel_size=5, padding='valid', strides=1,
                                          activation='tanh'))
        self.feature_extractor.add(AveragePooling2D(pool_size=2, strides=2))

        # classifier
        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=120, activation='tanh'))
        self.classifier.add(Dense(units=10, activation='softmax'))

    def call(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()

        # feature extractor
        self.zero_padding = ZeroPadding2D(padding=2)
        self.feature_extractor = Sequential()
        self.feature_extractor.add(Conv2D(filters=6, kernel_size=5, padding='valid', strides=1,
                                          activation='tanh'))
        self.feature_extractor.add(AveragePooling2D(pool_size=2, strides=2))
        self.feature_extractor.add(Conv2D(filters=16, kernel_size=5, padding='valid', strides=1,
                                          activation='tanh'))
        self.feature_extractor.add(AveragePooling2D(pool_size=2, strides=2))

        # classifier
        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=140, activation='tanh'))
        self.classifier.add(Dense(units=84, activation='tanh'))
        self.classifier.add(Dense(units=10, activation='softmax'))

    def call(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x

