#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Siamese network for tracking
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16 as VGG16_original
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input


class SiamTracker(tf.keras.Model):

    def __init__(self):
        super(SiamTracker, self).__init__()

    def call(self, x):
        """
        TODO
        """
        return None
