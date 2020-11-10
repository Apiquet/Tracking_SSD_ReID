#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSD300 implementation: https://arxiv.org/abs/1512.02325
"""

from .VGG16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
keras = tf.keras


class SSD300():

    def __init__(self, num_categories=10):
        super(SSD300, self).__init__()
        self.num_categories = num_categories

        '''
            Cone Implementation
        '''
        self.VGG16 = VGG16(input_shape=(300, 300, 3))
        self.VGG16_stage_4 = self.VGG16.getUntilStage4()
        self.VGG16_stage_5 = self.VGG16.getStage5()

        # fc6 to dilated conv
        self.stage_6_1_1024 = Conv2D(filters=1024,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     activation="relu",
                                     dilation_rate=6,
                                     name="FC6_to_Conv6")
        # fc7
        self.stage_7_1_1024 = Conv2D(filters=1024,
                                     kernel_size=(1, 1),
                                     padding="same",
                                     activation="relu",
                                     name="FC7_to_Conv7")
        # conv8_1
        self.stage_8_1_256 = Conv2D(filters=256,
                                    kernel_size=(1, 1),
                                    activation="relu",
                                    name="Conv8_1")
        # conv8_2
        self.stage_8_2_512 = Conv2D(filters=512,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same",
                                    activation="relu",
                                    name="Conv8_2")
        # conv9_1
        self.stage_9_1_128 = Conv2D(filters=128,
                                    kernel_size=(1, 1),
                                    activation="relu",
                                    name="Conv9_1")
        # conv9_2
        self.stage_9_2_256 = Conv2D(filters=256,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="same",
                                    activation="relu",
                                    name="Conv9_2")
        # conv10_1
        self.stage_10_1_128 = Conv2D(filters=128,
                                     kernel_size=(1, 1),
                                     activation="relu",
                                     name="Conv10_1")
        # conv10_2
        self.stage_10_2_256 = Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     activation="relu",
                                     name="Conv10_2")
        # conv11_1
        self.stage_11_1_128 = Conv2D(filters=128,
                                     kernel_size=(1, 1),
                                     activation="relu",
                                     name="Conv11_1")
        # conv11_2
        self.stage_11_2_256 = Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     activation="relu",
                                     name="Conv11_2")

        '''
            Confidence layers for each block
        '''
        self.stage_4_conf = Conv2D(filters=4*num_categories,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu",
                                   name="conf_stage4")
        self.stage_7_conf = Conv2D(filters=6*num_categories,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu",
                                   name="conf_stage7")
        self.stage_8_conf = Conv2D(filters=6*num_categories,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu",
                                   name="conf_stage8")
        self.stage_9_conf = Conv2D(filters=6*num_categories,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu",
                                   name="conf_stage9")
        self.stage_10_conf = Conv2D(filters=4*num_categories,
                                    kernel_size=(3, 3),
                                    padding="same",
                                    activation="relu",
                                    name="conf_stage10")
        self.stage_11_conf = Conv2D(filters=4*num_categories,
                                    kernel_size=(3, 3),
                                    padding="same",
                                    activation="relu",
                                    name="conf_stage11")

        '''
            Localization layers for each block
        '''
        self.stage_4_loc = Conv2D(filters=4*4,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu",
                                  name="loc_stage4")
        self.stage_7_loc = Conv2D(filters=6*4,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu",
                                  name="loc_stage7")
        self.stage_8_loc = Conv2D(filters=6*4,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu",
                                  name="loc_stage8")
        self.stage_9_loc = Conv2D(filters=6*4,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu",
                                  name="loc_stage9")
        self.stage_10_loc = Conv2D(filters=4*4,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu",
                                   name="loc_stage10")
        self.stage_11_loc = Conv2D(filters=4*4,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   activation="relu",
                                   name="loc_stage11")

    def train(self):
        return None

    def getCone(self):
        return keras.models.Sequential([
            self.VGG16_stage_4,
            self.VGG16_stage_5,
            self.stage_6_1_1024,
            self.stage_7_1_1024,
            self.stage_8_1_256,
            self.stage_8_2_512,
            self.stage_9_1_128,
            self.stage_9_2_256,
            self.stage_10_1_128,
            self.stage_10_2_256,
            self.stage_11_1_128,
            self.stage_11_2_256])

    def call(self, x):
        confs_per_stage = []
        locs_per_stage = []

        x = self.VGG16_stage_4(x)
        confs_per_stage.append(self.stage_4_conf(x))
        locs_per_stage.append(self.stage_4_loc(x))

        x = self.VGG16_stage_5(x)
        x = self.stage_6_1_1024(x)
        x = self.stage_7_1_1024(x)
        confs_per_stage.append(self.stage_7_conf(x))
        locs_per_stage.append(self.stage_7_loc(x))

        x = self.stage_8_1_256(x)
        x = self.stage_8_2_512(x)
        confs_per_stage.append(self.stage_8_conf(x))
        locs_per_stage.append(self.stage_8_loc(x))

        x = self.stage_9_1_128(x)
        x = self.stage_9_2_256(x)
        confs_per_stage.append(self.stage_9_conf(x))
        locs_per_stage.append(self.stage_9_loc(x))

        x = self.stage_10_1_128(x)
        x = self.stage_10_2_256(x)
        confs_per_stage.append(self.stage_10_conf(x))
        locs_per_stage.append(self.stage_10_loc(x))

        x = self.stage_11_1_128(x)
        x = self.stage_11_2_256(x)
        confs_per_stage.append(self.stage_11_conf(x))
        locs_per_stage.append(self.stage_11_loc(x))

        return confs_per_stage, locs_per_stage
