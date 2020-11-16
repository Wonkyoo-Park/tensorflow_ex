from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, ZeroPadding2D


class FeatureExtrator(Layer):
    def __init__(self, filter1, filter2):
        super(FeatureExtrator, self).__init__()

        self.conv1 = Conv2D(filters=filter1, kernerl_size=5, padding='valid',
                            strides=1, activation='tanh')
        self.conv1_pool = AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = Conv2D(filters=filter2, kernel_size=5, padding='valid',
                            strides=1, activation='tanh')
        self.conv2_pool = AveragePooling2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)

        return x


class LeNet1(Model):
    def __init__(self):
        super(LeNet1, self).__init__()

        # feature extractor
        self.feature_extractor = FeatureExtrator(4, 12)

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
        self.zero_padding = ZeroPadding2D(padding=2)
        self.feature_extractor = FeatureExtrator(4, 16)

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
        self.feature_extractor = FeatureExtrator(6, 16)

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
