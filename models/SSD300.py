#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSD300 implementation: https://arxiv.org/abs/1512.02325
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

from .VGG16 import VGG16


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
        self.stage_4_batch_norm = tf.keras.layers.BatchNormalization()
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

        '''
            Default boxes parameters
        '''
        self.ratios = [[1, 1/2, 2],
                       [1, 1/2, 2, 1/3, 3],
                       [1, 1/2, 2, 1/3, 3],
                       [1, 1/2, 2, 1/3, 3],
                       [1, 1/2, 2],
                       [1, 1/2, 2]]
        self.scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        self.fm_resolutions = [38, 19, 10, 5, 3, 1]

        self.default_boxes = self.getDefaultBoxes()
        self.stage_4_boxes = self.default_boxes[0]
        self.stage_7_boxes = self.default_boxes[1]
        self.stage_8_boxes = self.default_boxes[2]
        self.stage_9_boxes = self.default_boxes[3]
        self.stage_10_boxes = self.default_boxes[4]
        self.stage_11_boxes = self.default_boxes[5]

    def train(self):
        return None

    def getCone(self):
        return tf.keras.models.Sequential([
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

    def getDefaultBoxes(self):
        boxes = []
        for fm_idx in range(len(self.fm_resolutions)):
            boxes_fm_i = []
            for i in range(self.fm_resolutions[fm_idx]):
                for j in range(self.fm_resolutions[fm_idx]):
                    # box with scale 0.5
                    boxes_fm_i.append([i, j,
                                       self.scales[fm_idx]/2.,
                                       self.scales[fm_idx]/2.])
                    # box with aspect ratio
                    for ratio in self.ratios[fm_idx]:
                        boxes_fm_i.append([
                            i, j,
                            self.scales[fm_idx] * np.sqrt(ratio),
                            self.scales[fm_idx] / np.sqrt(ratio)])
            boxes.append(tf.constant((boxes_fm_i)))
        return boxes

    def call(self, x):
        confs_per_stage = []
        locs_per_stage = []

        # stage 4
        x = self.VGG16_stage_4(x)
        x_normed = self.stage_4_batch_norm(x)
        confs_per_stage.append(self.stage_4_conf(x_normed))
        locs_per_stage.append(self.stage_4_loc(x_normed))

        # stage 7
        x = self.VGG16_stage_5(x)
        x = self.stage_6_1_1024(x)
        x = self.stage_7_1_1024(x)
        confs_per_stage.append(self.stage_7_conf(x))
        locs_per_stage.append(self.stage_7_loc(x))

        # stage 8
        x = self.stage_8_1_256(x)
        x = self.stage_8_2_512(x)
        confs_per_stage.append(self.stage_8_conf(x))
        locs_per_stage.append(self.stage_8_loc(x))

        # stage 9
        x = self.stage_9_1_128(x)
        x = self.stage_9_2_256(x)
        confs_per_stage.append(self.stage_9_conf(x))
        locs_per_stage.append(self.stage_9_loc(x))

        # stage 10
        x = self.stage_10_1_128(x)
        x = self.stage_10_2_256(x)
        confs_per_stage.append(self.stage_10_conf(x))
        locs_per_stage.append(self.stage_10_loc(x))

        # stage 11
        x = self.stage_11_1_128(x)
        x = self.stage_11_2_256(x)
        confs_per_stage.append(self.stage_11_conf(x))
        locs_per_stage.append(self.stage_11_loc(x))

        return confs_per_stage, locs_per_stage
