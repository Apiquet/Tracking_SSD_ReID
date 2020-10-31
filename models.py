from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten


class VGG16(keras.Model):

    def __init__(self):
        super(VGG16, self).__init__()
        # available layers
        # Typo: layerType_Stage_NumberInStage_Info
        self.conv_1_1_64 = Conv2D(input_shape=(224, 224, 3),
                                  filters=64,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu")
        self.conv_1_2_64 = Conv2D(input_shape=(224, 224, 64),
                                  filters=64,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu")
        self.maxpool_1_3_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv_2_1_128 = Conv2D(input_shape=(112, 112, 64),
                                   filters=128,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.conv_2_2_128 = Conv2D(input_shape=(112, 112, 64),
                                   filters=128,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.maxpool_2_3_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(4096, activation='relu')
        self.dense2 = Dense(4096, activation='relu')
        self.dense3 = Dense(2, activation='softmax')

    def call(self, x):
        # Stage 1
        x = self.conv_1_1_64(x)
        x = self.conv_1_2_64(x)
        x = self.maxpool_1_3_2x2(x)

        # Stage 2
        x = self.conv_2_1_128(x)
        x = self.conv_2_2_128(x)
        x = self.maxpool_2_3_2x2(x)

        # Final stage
        x = self.dense1(x)
        x = self.dense3(x)
        x = self.dense3(x)
        return x
