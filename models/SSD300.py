from .VGG16 import VGG16
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten


class SSD300():

    def __init__(self):
        super(SSD300, self).__init__()

        '''
            Model Implementation
        '''
        self.backbone = VGG16(input_shape=(300, 300, 3))
        self.backbone_model = self.backbone.getModel()

        self.model = keras.models.Sequential(
            self.backbone_model.layers[:self.backbone.getIdxLastMAxPoolLayer()])
        # fc6 to Conv
        self.model.add(Conv2D(filters=1024,
                              kernel_size=(3, 3),
                              padding="same",
                              activation="relu",
                              dilation_rate=6))
        # fc7 to dilated conv
        self.model.add(Conv2D(filters=1024,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu"))
        # conv8_1
        self.model.add(Conv2D(filters=256,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu"))
        # conv8_2
        self.model.add(Conv2D(filters=512,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              activation="relu"))
        # conv9_1
        self.model.add(Conv2D(filters=128,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu"))
        # conv9_2
        self.model.add(Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              activation="relu"))
        # conv10_1
        self.model.add(Conv2D(filters=128,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu"))
        # conv10_2
        self.model.add(Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              activation="relu"))
        # conv11_1
        self.model.add(Conv2D(filters=128,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu"))
        # conv11_2
        self.model.add(Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              activation="relu"))
        # Temp output to remove
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

    def getModel(self):
        return self.model

    def call(self, x):
        return self.model(x)
