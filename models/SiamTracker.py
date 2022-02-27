#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Siamese network for tracking
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16 as VGG16_original

try:
    from .VGG16 import VGG16
except Exception:
    from VGG16 import VGG16


class SiamTracker(tf.keras.Model):
    def __init__(self, input_shape=(300, 300, 3)):
        super(SiamTracker, self).__init__()
        self.VGG16 = VGG16(input_shape=input_shape)
        self.feature_extractor_1 = self.VGG16.getUntilStage5()
        self.feature_extractor_2 = self.VGG16.getUntilStage5()

    def load_vgg16_imagenet_weights(self):
        """ Use pretrained weights from imagenet """
        vgg16_original = VGG16_original(weights='imagenet')

        for i in range(len(self.VGG16_stage_4.layers)):
            self.VGG16_stage_4.get_layer(index=i).set_weights(
                vgg16_original.get_layer(index=i + 1).get_weights()
            )

        for j in range(len(self.VGG16_stage_5.layers)):
            self.VGG16_stage_5.get_layer(index=j).set_weights(
                vgg16_original.get_layer(index=i + j + 2).get_weights()
            )

    def call(self, intput_1, input_2):
        """
        TODO
        """
        x_1 = self.feature_extractor_1(intput_1)
        x_2 = self.feature_extractor_2(intput_2)

        return None
