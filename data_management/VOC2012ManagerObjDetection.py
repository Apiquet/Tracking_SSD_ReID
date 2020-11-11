#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pascal VOC2012 dataset manager
"""

import numpy as np
import os
import tensorflow as tf


class VOC2012ManagerObjDetection():

    def __init__(self, path):
        super(VOC2012ManagerObjDetection, self).__init__()
        self.path = path
        self.img_resolution = (300, 300, 3)
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        self.images_path = path + "/JPEGImages/"
        self.annotations_path = path + "/Annotations/"
        self.images_name = [im.replace(".jpg", "")
                            for im in os.listdir(self.images_path)
                            if os.path.isfile(os.path.join(self.images_path,
                                                           im))]
        self.number_samples = len(self.images_name)

    def getImages(self, images_name: list):
        images = []
        for img in images_name:
            image = tf.keras.preprocessing.image.load_img(
                self.images_path + img + ".jpg",
                target_size=self.img_resolution)
            images.append(tf.keras.preprocessing.image.img_to_array(image))
        return tf.convert_to_tensor(images, dtype=tf.float32)
