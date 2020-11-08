#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSD300 implementation: https://arxiv.org/abs/1512.02325
"""

from .VGG16 import VGG16
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten


class SSD300():

    def __init__(self, num_categories=10):
        super(SSD300, self).__init__()

        '''
            Cone Implementation
        '''
        self.VGG16 = VGG16(input_shape=(300, 300, 3))
        self.model = self.VGG16.getBackbone()

        # fc6 to dilated conv
        self.model.add(Conv2D(filters=1024,
                              kernel_size=(3, 3),
                              padding="same",
                              activation="relu",
                              dilation_rate=6,
                              name="FC6_to_Conv6"))
        # fc7
        self.model.add(Conv2D(filters=1024,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu",
                              name="FC7_to_Conv7"))
        # conv8_1
        self.model.add(Conv2D(filters=256,
                              kernel_size=(1, 1),
                              activation="relu",
                              name="Conv8_1"))
        # conv8_2
        self.model.add(Conv2D(filters=512,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              activation="relu",
                              name="Conv8_2"))
        # conv9_1
        self.model.add(Conv2D(filters=128,
                              kernel_size=(1, 1),
                              activation="relu",
                              name="Conv9_1"))
        # conv9_2
        self.model.add(Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              activation="relu",
                              name="Conv9_2"))
        # conv10_1
        self.model.add(Conv2D(filters=128,
                              kernel_size=(1, 1),
                              activation="relu",
                              name="Conv10_1"))
        # conv10_2
        self.model.add(Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              activation="relu",
                              name="Conv10_2"))
        # conv11_1
        self.model.add(Conv2D(filters=128,
                              kernel_size=(1, 1),
                              activation="relu",
                              name="Conv11_1"))
        # conv11_2
        self.model.add(Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              activation="relu",
                              name="Conv11_2"))
        # Temp output to remove
        self.model.add(Flatten())
        self.model.add(Dense(num_categories, activation='softmax'))

        '''
            Confidence layers for each block
        '''
        self.conv_conf_stage5 = Conv2D(filters=4*(num_categories+4),
                                       kernel_size=(3, 3),
                                       padding="same",
                                       activation="relu",
                                       name="conf_stage5")
        self.conv_conf_stage7 = Conv2D(filters=6*(num_categories+4),
                                       kernel_size=(3, 3),
                                       padding="same",
                                       activation="relu",
                                       name="conf_stage7")
        self.conv_conf_stage8 = Conv2D(filters=6*(num_categories+4),
                                       kernel_size=(3, 3),
                                       padding="same",
                                       activation="relu",
                                       name="conf_stage8")
        self.conv_conf_stage9 = Conv2D(filters=6*(num_categories+4),
                                       kernel_size=(3, 3),
                                       padding="same",
                                       activation="relu",
                                       name="conf_stage9")
        self.conv_conf_stage10 = Conv2D(filters=4*(num_categories+4),
                                        kernel_size=(3, 3),
                                        padding="same",
                                        activation="relu",
                                        name="conf_stage10")
        self.conv_conf_stage11 = Conv2D(filters=4*(num_categories+4),
                                        kernel_size=(3, 3),
                                        padding="same",
                                        activation="relu",
                                        name="conf_stage11")

    def getModel(self):
        return self.model

    def call(self, x):
        return self.model(x)
