#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Naive tracking class based on subject category and localization
"""

import numpy as np


class _subjectTracked():

    def __init__(self, category: int, loc: tf.Tensor):
        super(_subjectTracked, self).__init__()
        # object class
        self.category = category
        # localization bounding box (cx, cy, w, h)
        self.loc = loc


class NaiveTracking():

    def __init__(self):
        super(NaiveTracking, self).__init__()

        # list of tracked subjects
        self.subjects = []

    def call(self, categories, boxes):
        """
        Get subjects' identity from categories and boxes on current frame
        N: number of subjects found in frame

        Args:
            - (tf.Tensor) category of each subject (N,)
            - (tf.Tensor) boxes containing subjects format cx, cy, w, h (N, 4)

        Return:
            - (tf.Tensor) Identity for each subject (N,)
        """
