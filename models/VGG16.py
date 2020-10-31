from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten


class VGG16():

    def __init__(self):
        super(VGG16, self).__init__()

        '''
            Available layers
            Typo: layerType_Stage_NumberInStage_Info
        '''
        self.conv_1_1_64 = Conv2D(input_shape=(224, 224, 3),
                                  filters=64,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu")
        self.conv_1_2_64 = Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu")
        self.maxpool_1_3_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_2_1_128 = Conv2D(filters=128,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.conv_2_2_128 = Conv2D(filters=128,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.maxpool_2_3_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_3_1_256 = Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.conv_3_2_256 = Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.conv_3_3_256 = Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.maxpool_3_4_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_4_1_512 = Conv2D(filters=512,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.conv_4_2_512 = Conv2D(filters=512,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.conv_4_3_512 = Conv2D(filters=512,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.maxpool_4_4_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_5_1_512 = Conv2D(filters=512,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.conv_5_2_512 = Conv2D(filters=512,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.conv_5_3_512 = Conv2D(filters=512,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu")
        self.maxpool_5_4_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten_6_1 = Flatten()
        self.dense_6_2_4096 = Dense(4096, activation='relu')
        self.dense_6_3_4096 = Dense(4096, activation='relu')
        self.dense_6_4_2 = Dense(2, activation='softmax')

    def model(self):
        '''
            Model Implementation
        '''
        self.model = keras.models.Sequential()

        # Stage 1
        self.model.add(self.conv_1_1_64)
        self.model.add(self.conv_1_2_64)
        self.model.add(self.maxpool_1_3_2x2)
        # Stage 2
        self.model.add(self.conv_2_1_128)
        self.model.add(self.conv_2_2_128)
        self.model.add(self.maxpool_2_3_2x2)
        # Stage 3
        self.model.add(self.conv_3_1_256)
        self.model.add(self.conv_3_2_256)
        self.model.add(self.conv_3_3_256)
        self.model.add(self.maxpool_3_4_2x2)
        # Stage 4
        self.model.add(self.conv_4_1_512)
        self.model.add(self.conv_4_2_512)
        self.model.add(self.conv_4_3_512)
        self.model.add(self.maxpool_4_4_2x2)
        # Stage 5
        self.model.add(self.conv_5_1_512)
        self.model.add(self.conv_5_2_512)
        self.model.add(self.conv_5_3_512)
        self.model.add(self.maxpool_5_4_2x2)
        # Stage 5
        self.model.add(self.flatten_6_1)
        self.model.add(self.dense_6_2_4096)
        self.model.add(self.dense_6_3_4096)
        self.model.add(self.dense_6_4_2)

        return self.model

    def call(self, x):
        return self.model(x)
