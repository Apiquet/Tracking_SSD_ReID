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
        self.identity = identity


class NaiveTracker():

    def __init__(self):
        super(NaiveTracker, self).__init__()

        # list of tracked subjects
        self.subjects = []
        self.max_id = -1

    def setSeenAttribute(self, value=False):
        for subject in self.subjects:
            subject.seen = value

    def clearSubjects(self):
        keep = []
        for subject in self.subjects:
            if subject.seen <= 5:
                keep.append(True)
            else:
                keep.append(False)
        self.subjects = [i for (i, v) in zip(self.subjects, keep) if v]

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
        return iou >= iou_threshold, iou

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

        # remove subject with lifespan > threshold
        self.clearSubjects()
        identity = np.ones(categories.shape) * -1

        # loop over the categories
        for category in tf.unique(categories)[0]:
            cat_idx = categories == category

            for subject in self.subjects:
                # no need to compute IoU is subject has a different class
                if subject.category != category:
                    continue
                iou, values = self.computeJaccardIdx(subject.loc, boxes, 0.3)
                iou = tf.math.logical_and(iou, cat_idx)
                iou = tf.dtypes.cast(iou, tf.int16)
                # verify if at least a box has and IoU >= 0.3
                if tf.reduce_sum(iou, axis=0) != 0:
                    subject.seen = True
                    # give subject's id to box with max IoU score
                    idx_max_iou = values.numpy().argmax(axis=0)
                    subject.loc = boxes[idx_max_iou]
                    subject.lifespan = 0
                    identity[idx_max_iou] = subject.identity
        # increase lifespan for any subject that was not seen
        for subject in self.subjects:
            if subject.seen is False:
                subject.lifespan += 1
        # create a new subject for each box that did not get an id
        for i, box in enumerate(boxes):
            if identity[i] == -1:
                self.max_id += 1
                new_subject = _subjectTracked(categories[i], box, self.max_id)
                self.subjects.append(new_subject)
                identity[i] = self.max_id
        return tf.dtypes.cast(identity, tf.int16)
