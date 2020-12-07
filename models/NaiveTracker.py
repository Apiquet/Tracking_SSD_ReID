#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Naive tracking class based on subject category and localization
"""

import numpy as np
import tensorflow as tf


class _subjectTracked():

    def __init__(self, category: int, loc: tf.Tensor, identity: int):
        super(_subjectTracked, self).__init__()
        # object class
        self.category = category
        # localization bounding box (cx, cy, w, h)
        self.loc = loc
        # lifespan increased when subject no seen on an image
        self.lifespan = 0
        # bool to know if subject was seen in the current frame
        self.seen = False
        # subject id
        self.identity = -1

class NaiveTracker():

    def __init__(self):
        super(NaiveTracker, self).__init__()

        # list of tracked subjects
        self.subjects = []
        self.max_id = -1

    def setSeenAttribute(self, value=False):
        for subject in self.subjects:
            subject.seen = value

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

    def __call__(self, categories, boxes):
        """
        Get subjects' identity from categories and boxes on current frame
        N: number of subjects found in frame

        Args:
            - (tf.Tensor) category of each subject (N,)
            - (tf.Tensor) boxes containing subjects format cx, cy, w, h (N, 4)

        Return:
            - (tf.Tensor) Identity for each subject (N,)
        """
        self.setSeenAttribute(False)
        identity = np.zeros(categories.shape) * -1
        iou = np.zeros(categories.shape)
        iou_bin_masks = []
        for category in tf.unique(categories)[0]:
            cat_idx = categories == category

            for subject in self.subjects:
                iou = self.computeJaccardIdx(subject.loc, boxes, 0.3)
                iou = tf.math.logical_and(iou, cat_idx)
                iou = tf.dtypes.cast(iou, tf.int16)
                if tf.reduce_sum(iou, axis=0) != 0:

                    subject.seen = True
                    identity[iou.numpy()] = subject.identity
                    
                    for mask in iou_bin_masks:
                        iou = tf.clip_by_value(iou - mask, 0, 1)
                    iou_bin_masks.append(iou)

        for subject in self.subjects:
            if subject.seen == False:
                subject.lifespan += 1

        for i, box in enumerate(boxes):
            if not iou[i]:
                self.max_id += 1
                new_subject = _subjectTracked(categories[i], box, self.max_id)
                self.subjects.append(new_subject)
                identity[i] = self.max_id
        return tf.dtypes.cast(identity, tf.int16)
