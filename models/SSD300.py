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

        '''
            Loss utils
        '''
        self.before_mining_crossentropy =\
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                          reduction='none')
        self.after_mining_crossentropy =\
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.smooth_l1 = tf.keras.losses.Huber(reduction='sum',
                                               name='smooth_L1')

    def train(self):
        return None

    def getCone(self):
        """
        Method to get the cone of the SSD architecture
        """
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
        """
        Method to generate all default boxes for all the feature maps
        There are 6 stages to output boxes so this method returns a list of
        size 6 with all the boxes:
            width feature maps * height feature maps * number of boxes per loc
        For instance with the stage 4: 38x38x4=5776 default boxes

        Return:
            - (list of tf.Tensor) boxes per stage, 4 parameters cx, cy, w, h
        """
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

    def reshapeConfLoc(self, conf, loc, number_of_boxes):
        """
        Method to reshape the confidences and localizations convolutions
        W = width of the feature map
        H = height of the feature map
        B = mini-batch size
        N = number of boxes per location (should be 4 or 6)
        Confidences from [B, W, H, N * number of classes]
                    to   [B, number of default boxes, number of classes]
        loc         from [B, W, H, N * 4]
                    to   [B, number of default boxes, 4]
        
        Args:
            - (tf.Tensor) confidences of shape [B, W, H, N * number classes]
            - (tf.Tensor) loc of shape [B, W, H, N * 4]
            - (int) number of boxes

        Return:
            - (tf.Tensor) confidences of shape [B, n boxes, n classes]
            - (tf.Tensor) loc of shape [B, number of default boxes, 4]
        """
        conf = tf.reshape(conf, [conf.shape[0], number_of_boxes,
                                 self.num_categories])
        loc = tf.reshape(loc, [loc.shape[0], number_of_boxes, 4])
        return conf, loc

    def calculateLoss(self, confs_pred, confs_gt, locs_pred, locs_gt):
        """
        Method to calculate loss for confidences and localization offsets
        B = mini-batch size
        
        Args:
            - (tf.Tensor) confidences prediction: [B, N boxes, n classes]
            - (tf.Tensor) confidence ground truth:  [B, N boxes, 1]
            - (tf.Tensor) localization offsets prediction: [B, N boxes, 4]
            - (tf.Tensor) localization offsets ground truth: [B, N boxes, 4]

        Return:
            - (tf.Tensor) confidences of shape [B, 1]
            - (tf.Tensor) loc of shape [B, 1]
        """
        positives_idx = confs_gt > 0
        positives_number = tf.reduce_sum(
            tf.dtypes.cast(positives_idx, tf.float32), axis=1)
        confs_loss_before_mining = self.before_mining_crossentropy(confs_gt,
                                                                   confs_pred)

        # Negatives mining with <3:1 ratio for negatives:positives
        negatives_number = positives_number * 3
        negatives_rank = tf.argsort(confs_loss_before_mining, axis=1,
                                    direction='DESCENDING')
        rank_idx = tf.keras.backend.eval(negatives_rank)
        negatives_idx = tf.expand_dims(rank_idx <= negatives_number, 2)

        # loss calculation (pos+neg for conf, pos for loc)
        confs_idx = tf.math.logical_or(positives_idx, negatives_idx)
        confs_idx_rpt = tf.repeat(confs_idx, repeats=[10], axis=-1)
        confs_loss = self.after_mining_crossentropy(
            tf.reshape(confs_gt[confs_idx], [-1, confs_gt.shape[-1]]),
            tf.reshape(confs_pred[confs_idx_rpt], [-1, confs_pred.shape[-1]]))

        positives_idx_repeated = tf.repeat(confs_idx, repeats=[4], axis=-1)
        locs_loss = self.smooth_l1(locs_gt[positives_idx_repeated],
                                   locs_pred[positives_idx_repeated])

        confs_loss = confs_loss / positives_number
        locs_loss = locs_loss / positives_number

        return confs_loss, locs_loss

    def __call__(self, x):
        confs_per_stage = []
        locs_per_stage = []

        # stage 4
        x = self.VGG16_stage_4(x)
        x_normed = self.stage_4_batch_norm(x)
        conf, loc = self.reshapeConfLoc(self.stage_4_conf(x_normed),
                                        self.stage_4_loc(x_normed),
                                        self.stage_4_boxes.shape[0])
        confs_per_stage.append(conf)
        locs_per_stage.append(loc)

        # stage 7
        x = self.VGG16_stage_5(x)
        x = self.stage_6_1_1024(x)
        x = self.stage_7_1_1024(x)
        conf, loc = self.reshapeConfLoc(self.stage_7_conf(x),
                                        self.stage_7_loc(x),
                                        self.stage_7_boxes.shape[0])
        confs_per_stage.append(conf)
        locs_per_stage.append(loc)

        # stage 8
        x = self.stage_8_1_256(x)
        x = self.stage_8_2_512(x)
        conf, loc = self.reshapeConfLoc(self.stage_8_conf(x),
                                        self.stage_8_loc(x),
                                        self.stage_8_boxes.shape[0])
        confs_per_stage.append(conf)
        locs_per_stage.append(loc)

        # stage 9
        x = self.stage_9_1_128(x)
        x = self.stage_9_2_256(x)
        conf, loc = self.reshapeConfLoc(self.stage_9_conf(x),
                                        self.stage_9_loc(x),
                                        self.stage_9_boxes.shape[0])
        confs_per_stage.append(conf)
        locs_per_stage.append(loc)

        # stage 10
        x = self.stage_10_1_128(x)
        x = self.stage_10_2_256(x)
        conf, loc = self.reshapeConfLoc(self.stage_10_conf(x),
                                        self.stage_10_loc(x),
                                        self.stage_10_boxes.shape[0])
        confs_per_stage.append(conf)
        locs_per_stage.append(loc)

        # stage 11
        x = self.stage_11_1_128(x)
        x = self.stage_11_2_256(x)
        conf, loc = self.reshapeConfLoc(self.stage_11_conf(x),
                                        self.stage_11_loc(x),
                                        self.stage_11_boxes.shape[0])
        confs_per_stage.append(conf)
        locs_per_stage.append(loc)

        fake_conf_gt = tf.random.uniform(shape=[confs_per_stage[0].shape[0],
                                                confs_per_stage[0].shape[1],
                                                1],
                                         minval=0, maxval=9, dtype=tf.int32)
        self.calculateLoss(confs_per_stage[0], fake_conf_gt,
                           locs_per_stage[0], locs_per_stage[0])

        return confs_per_stage, locs_per_stage
