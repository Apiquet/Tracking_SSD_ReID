#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSD300 implementation: https://arxiv.org/abs/1512.02325
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16 as VGG16_original
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

from .VGG16 import VGG16


class SSD300(tf.keras.Model):

    def __init__(self, num_categories=10, floatType=32):
        super(SSD300, self).__init__()
        self.num_categories = num_categories
        if floatType == 32:
            self.floatType = tf.float32
        elif floatType == 16:
            self.floatType = tf.float16
        else:
            raise Exception('floatType should be either 32 or 16')

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
                                   name="conf_stage4")
        self.stage_7_conf = Conv2D(filters=6*num_categories,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   name="conf_stage7")
        self.stage_8_conf = Conv2D(filters=6*num_categories,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   name="conf_stage8")
        self.stage_9_conf = Conv2D(filters=6*num_categories,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   name="conf_stage9")
        self.stage_10_conf = Conv2D(filters=4*num_categories,
                                    kernel_size=(3, 3),
                                    padding="same",
                                    name="conf_stage10")
        self.stage_11_conf = Conv2D(filters=4*num_categories,
                                    kernel_size=(3, 3),
                                    padding="same",
                                    name="conf_stage11")

        '''
            Localization layers for each block
        '''
        self.stage_4_loc = Conv2D(filters=4*4,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  name="loc_stage4")
        self.stage_7_loc = Conv2D(filters=6*4,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  name="loc_stage7")
        self.stage_8_loc = Conv2D(filters=6*4,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  name="loc_stage8")
        self.stage_9_loc = Conv2D(filters=6*4,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  name="loc_stage9")
        self.stage_10_loc = Conv2D(filters=4*4,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   name="loc_stage10")
        self.stage_11_loc = Conv2D(filters=4*4,
                                   kernel_size=(3, 3),
                                   padding="same",
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

        self.default_boxes_per_stage, self.default_boxes = \
            self.getDefaultBoxes()
        self.stage_4_boxes = self.default_boxes_per_stage[0]
        self.stage_7_boxes = self.default_boxes_per_stage[1]
        self.stage_8_boxes = self.default_boxes_per_stage[2]
        self.stage_9_boxes = self.default_boxes_per_stage[3]
        self.stage_10_boxes = self.default_boxes_per_stage[4]
        self.stage_11_boxes = self.default_boxes_per_stage[5]

        '''
            Loss utils
        '''
        self.before_mining_crossentropy =\
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                          reduction='none')
        self.after_mining_crossentropy =\
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                          reduction='sum')
        self.smooth_l1 = tf.keras.losses.Huber(reduction='sum',
                                               name='smooth_L1')

    def train(self):
        return None

    def getCone(self):
        """ Method to get the cone of the SSD architecture """
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

    def load_vgg16_imagenet_weights(self):
        """ Use pretrained weights from imagenet """
        vgg16_original = VGG16_original(weights='imagenet')

        for i in range(len(self.VGG16_stage_4.layers)):
            self.VGG16_stage_4.get_layer(index=i).set_weights(
                vgg16_original.get_layer(index=i+1).get_weights())

        for j in range(len(self.VGG16_stage_5.layers)):
            self.VGG16_stage_5.get_layer(index=j).set_weights(
                vgg16_original.get_layer(index=i+j+2).get_weights())

    def getDefaultBoxes(self):
        """
        Method to generate all default boxes for all the feature maps
        There are 6 stages to output boxes so this method returns a list of
        size 6 with all the boxes:
            width feature maps * height feature maps * number of boxes per loc
        For instance with the stage 4: 38x38x4=5776 default boxes

        Return:
            - (list of tf.Tensor) boxes per stage, 4 parameters cx, cy, w, h
                [number of stage, number of default boxes per stage, 4]
            - (list of tf.Tensor) boxes, 4 parameters cx, cy, w, h
                [number of default boxes, 4]
        """
        boxes_per_stage = []
        boxes = []
        for fm_idx in range(len(self.fm_resolutions)):
            boxes_fm_i = []
            step = 1/self.fm_resolutions[fm_idx]
            for j in np.arange(0, 1, step):
                for i in np.arange(0, 1, step):
                    # box with scale 0.5
                    boxes_fm_i.append([i + step/2, j + step/2,
                                       self.scales[fm_idx]/2.,
                                       self.scales[fm_idx]/2.])
                    boxes.append([i + step/2, j + step/2,
                                  self.scales[fm_idx]/2.,
                                  self.scales[fm_idx]/2.])
                    # box with aspect ratio
                    for ratio in self.ratios[fm_idx]:
                        boxes_fm_i.append([
                            i + step/2, j + step/2,
                            self.scales[fm_idx] / np.sqrt(ratio),
                            self.scales[fm_idx] * np.sqrt(ratio)])
                        boxes.append([
                            i + step/2, j + step/2,
                            self.scales[fm_idx] / np.sqrt(ratio),
                            self.scales[fm_idx] * np.sqrt(ratio)])

            boxes_per_stage.append(tf.constant((boxes_fm_i)))
        return boxes_per_stage, tf.convert_to_tensor(boxes,
                                                     dtype=self.floatType)

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
            - (tf.Tensor) confidence ground truth:  [B, N boxes]
            - (tf.Tensor) localization offsets prediction: [B, N boxes, 4]
            - (tf.Tensor) localization offsets ground truth: [B, N boxes, 4]

        Return:
            - (tf.Tensor) confidences of shape [B, 1]
            - (tf.Tensor) loc of shape [B, 1]
        """
        positives_idx = confs_gt > 0
        positives_number = tf.reduce_sum(
            tf.dtypes.cast(positives_idx, self.floatType), axis=1)
        confs_loss_before_mining = self.before_mining_crossentropy(confs_gt,
                                                                   confs_pred)

        # Negatives mining with <3:1 ratio for negatives:positives
        negatives_number = tf.dtypes.cast(positives_number, tf.int32) * 3
        negatives_rank = tf.argsort(confs_loss_before_mining, axis=1,
                                    direction='DESCENDING')
        rank_idx = tf.argsort(negatives_rank, axis=1)
        negatives_idx = rank_idx <= tf.expand_dims(negatives_number, 1)

        # loss calculation (pos+neg for conf, pos for loc)
        confs_idx = tf.math.logical_or(positives_idx, negatives_idx)
        confs_loss = self.after_mining_crossentropy(confs_gt[confs_idx],
                                                    confs_pred[confs_idx])

        locs_loss = self.smooth_l1(locs_gt[positives_idx],
                                   locs_pred[positives_idx])

        confs_loss = confs_loss / tf.reduce_sum(positives_number)
        locs_loss = locs_loss / tf.reduce_sum(positives_number)

        return confs_loss, locs_loss

    def computeJaccardIdx(self, ref_box: tf.Tensor, boxes: tf.Tensor,
                          iou_threshold: float):
        """
        Method to get the boolean tensor where iou is superior to
        the specified threshold between the gt box and the default one
        D: number of default boxes

        Args:
            - (tf.Tensor) box with 4 parameters: cx, cy, w, h [4]
            - (tf.Tensor) box with 4 parameters: cx, cy, w, h [D, 4]
            - (float) iou threshold to use

        Return:
            - (tf.Tensor) 0 if iou > threshold, 1 otherwise [D]
        """
        # convert to xmin, ymin, xmax, ymax
        boxes = tf.concat([
            boxes[:, :2] - boxes[:, 2:] / 2,
            boxes[:, :2] + boxes[:, 2:] / 2], axis=-1)
        ref_box = tf.concat([ref_box[:2] - ref_box[2:] / 2,
                             ref_box[:2] + ref_box[2:] / 2], axis=-1)
        ref_box = tf.expand_dims(ref_box, 0)
        ref_box = tf.repeat(ref_box, repeats=[boxes.shape[0]], axis=0)

        # compute intersection
        inter_xymin = tf.math.maximum(boxes[:, :2],
                                      ref_box[:, :2])
        inter_xymax = tf.math.minimum(boxes[:, 2:],
                                      ref_box[:, 2:])
        inter_width_height = tf.clip_by_value(inter_xymax - inter_xymin,
                                              0.0, 300.0)
        inter_area = inter_width_height[:, 0] * inter_width_height[:, 1]

        # compute area of the boxes
        ref_box_width_height =\
            tf.clip_by_value(ref_box[:, 2:] - ref_box[:, :2],
                             0.0, 300.0)
        ref_box_width_height_area = ref_box_width_height[:, 0] *\
            ref_box_width_height[:, 1]

        boxes_width_height =\
            tf.clip_by_value(boxes[:, 2:] - boxes[:, :2],
                             0.0, 300.0)
        boxes_width_height_area = boxes_width_height[:, 0] *\
            boxes_width_height[:, 1]

        # compute iou
        iou = inter_area / (ref_box_width_height_area +
                            boxes_width_height_area - inter_area)
        return iou >= iou_threshold

    def recursive_nms(self, boxes_origin: tf.Tensor,
                      classes_origin: tf.Tensor,
                      scores_origin: tf.Tensor):
        """
        Method to filter boxes with score < 0.01
        Get maximum 200 boxes per image
        If IoU>=0.45 between two boxes of the same class, the biggest is kept
        B = mini-batch size
        F = number of non filtered boxes

        Args:
            - (tf.Tensor) boxes predicted: [N boxes, 4]
            - (tf.Tensor) classes of each box:  [N boxes]
            - (tf.Tensor) scores of each box:  [N boxes]

        Return:
            - (tf.Tensor) boxes predicted: [F, 4]
            - (tf.Tensor) class for each box:  [F]
            - (tf.Tensor) scores for each box:  [F]
        """
        score_threshold_idx = scores_origin >= 0.01
        boxes = boxes_origin[score_threshold_idx]
        classes = classes_origin[score_threshold_idx]
        scores = scores_origin[score_threshold_idx]

        rank = tf.argsort(scores, axis=0, direction='DESCENDING')
        rank_threshold_idx = rank <= 200

        boxes = boxes[rank_threshold_idx]
        classes = classes[rank_threshold_idx]
        scores = scores[rank_threshold_idx]

        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []
        for category in tf.unique(classes)[0]:
            cat_idx = classes == category
            cat_boxes = boxes[cat_idx]
            cat_scores = scores[cat_idx]

            while cat_boxes.shape[0] != 0:
                iou = self.computeJaccardIdx(cat_boxes[0], cat_boxes, 0.3)
                not_iou = tf.math.logical_not(iou)

                overlap_scores = cat_scores[iou]
                iou = tf.expand_dims(iou, 1)
                iou = tf.repeat(iou, repeats=[4], axis=1)
                overlap_boxes = tf.reshape(cat_boxes[iou],
                                           (overlap_scores.shape[0], 4))
                mean_box = tf.math.reduce_mean(overlap_boxes, axis=0)
                filtered_boxes.append(mean_box)
                max_scores = tf.math.reduce_max(overlap_scores, axis=0)
                filtered_scores.append(max_scores)
                filtered_classes.append(category)

                cat_boxes = cat_boxes[not_iou]
                cat_scores = cat_scores[not_iou]

        final_boxes = tf.convert_to_tensor(filtered_boxes, dtype=tf.float32)
        final_classes = tf.convert_to_tensor(filtered_classes, dtype=tf.int16)
        final_scores = tf.convert_to_tensor(filtered_scores, dtype=tf.float32)
        if boxes_origin.shape != final_boxes.shape:
            final_boxes, final_classes, final_scores = \
                self.recursive_nms(final_boxes, final_classes, final_scores)

        return final_boxes, final_classes, final_scores

    def getPredictionsFromConfsLocs(self, confs_pred, locs_pred,
                                    score_threshold=0.2,
                                    box_encoding="center",
                                    default=False):
        """
        Method to convert output offsets to boxes
        and scores to maximum class number
        Return boxes with score superior to score_threshold
        and non undefined class

        Args:
            - (tf.Tensor) scores for each box:  [B, N boxes, N classes]
            - (tf.Tensor) offsets for each box:  [B, N boxes, 4]
            - Optional: score threshold on confs_pred to accept prediction
            - Optional: box encoding: center: cx, cy, w, h;
                                      corner: xmin, ymin, xmax, ymax
            - Optional: default: displaying default boxes selected as gt

        Return:
            - (tf.Tensor) Predicted boxes (cx, cy, w, h): [B, N boxes, 4]
            - (tf.Tensor) Predicted class: [B, N boxes]
            - (tf.Tensor) Predicted score: [B, N boxes]
        """
        boxes_per_img = []
        classes_per_img = []
        scores_per_img = []
        for i in range(len(confs_pred)):
            idx_sup_thresh = tf.ones([confs_pred[i].shape[0]], tf.int32) == 1
            if default:
                boxes = self.default_boxes
                classes = confs_pred[i]
            else:
                boxes = self.default_boxes + locs_pred[i]
                scores = tf.reduce_max(confs_pred[i], axis=1)

                idx_sup_thresh = scores >= score_threshold

                classes = tf.argmax(confs_pred[i], axis=1)
            non_undefined_idx = classes > 0

            idx_to_keep = tf.logical_and(idx_sup_thresh, non_undefined_idx)

            scores = scores[idx_to_keep]
            scores_per_img.append(scores)

            classes = classes[idx_to_keep]
            classes_per_img.append(classes)

            boxes = boxes[idx_to_keep]
            if box_encoding == "corner":
                boxes = tf.concat([boxes[:, :2] - boxes[:, 2:] / 2,
                                   boxes[:, :2] + boxes[:, 2:] / 2],
                                  axis=-1)
            boxes_per_img.append(boxes)
        return boxes_per_img, classes_per_img, scores_per_img

    def call(self, x):
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

        return confs_per_stage, locs_per_stage
