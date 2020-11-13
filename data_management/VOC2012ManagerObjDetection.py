#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pascal VOC2012 dataset manager
"""

import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import xml.etree.ElementTree as ET


class VOC2012ManagerObjDetection():

    def __init__(self, path):
        super(VOC2012ManagerObjDetection, self).__init__()
        self.path = path
        self.img_resolution = (300, 300)
        self.classes = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
                        'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
                        'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12,
                        'motorbike': 13, 'person': 14, 'pottedplant': 15,
                        'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
        self.images_path = path + "/JPEGImages/"
        self.annotations_path = path + "/Annotations/"
        self.images_name = [im.replace(".jpg", "")
                            for im in os.listdir(self.images_path)
                            if os.path.isfile(os.path.join(self.images_path,
                                                           im))]
        self.number_samples = len(self.images_name)

    def getRawData(self, images_name: list):
        """
        Method to get images and annotations from a list of images name

        Args:
            - (list) images name without extension

        Return:
            - (tf.Tensor) Tensor of shape:
                [number of images, self.img_resolution]
            - (list of tf.Tensor) Boxes of shape:
                [number of images, number of objects, 4]
            - (list tf.Tensor) Classes of shape:
                [number of images, number of objects, 1]
        """
        images = []
        boxes = []
        classes = []
        for img in tqdm(images_name):
            image = tf.keras.preprocessing.image.load_img(
                self.images_path + img + ".jpg")
            w, h = image.size[0], image.size[1]
            image = tf.image.resize(np.array(image), self.img_resolution)
            images_array = \
                tf.keras.preprocessing.image.img_to_array(image) / 255.
            images.append(images_array)

            # annotation
            boxes_img_i, classes_img_i = self.getAnnotations(img, (w, h))
            boxes.append(boxes_img_i)
            classes.append(classes_img_i)
        return tf.convert_to_tensor(images, dtype=tf.float16), boxes, classes

    def getAnnotations(self, image_name: str, resolution: tuple):
        """
        Method to get annotation: boxes and classes

        Args:
            - (str) image name without extension
            - (tuple) image resolution (W, H, C) or (W, H)

        Return:
            - (tf.Tensor) Boxes of shape: [number of objects, 4]
            - (tf.Tensor) Classes of shape: [number of objects, 1]
        """
        boxes = []
        classes = []
        objects = ET.parse(
            self.annotations_path + image_name + ".xml").findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / resolution[0]
            ymin = float(bndbox.find('ymin').text) / resolution[1]
            xmax = float(bndbox.find('xmax').text) / resolution[0]
            ymax = float(bndbox.find('ymax').text) / resolution[1]

            # calculate cx, cy, width, height
            width = xmax-xmin
            height = ymax-ymin
            if xmin + width > 1. or ymin + height > 1. or\
               xmin < 0. or ymin < 0.:
                print("Boxe outside picture: (xmin, ymin, xmax, ymax):\
                      ({} {}, {}, {})".format(xmin, ymin, xmax, ymax))

            boxes.append([xmin+width/2., ymin+height/2., width, height])

            # get class
            name = obj.find('name').text.lower().strip()
            classes.append(self.classes[name])

        return tf.convert_to_tensor(boxes, dtype=tf.float16),\
            tf.convert_to_tensor(classes, dtype=tf.uint8)

    def getImagesAndGt(self, image_name: str, resolution: tuple):
        """
        Method to get the groud truth for confidence and localization
        S: number of stage
        D: number of default boxes

        Args:
            - (list) images name without extension
            - (list of tf.Tensor) default boxes per stage: [S, D, 4]
                4 parameters: cx, cy, w, h

        Return:
            - (tf.Tensor) confidence ground truth: [S, D, 1]
            - (tf.Tensor) loc ground truth: [S, D, 4]
        """
        return 0

    def computeRectangleArea(self, xmin, ymin, xmax, ymax):
        return (xmax - xmin) * (ymax - ymin)

    def computeJaccardIdx(self, box_1: tf.Tensor, box_2: tf.Tensor):
        """
        Method to get the Intersection-Over-Union between two boxes

        Args:
            - (tf.Tensor) box with 4 parameters: cx, cy, w, h [4]
            - (tf.Tensor) box with 4 parameters: cx, cy, w, h [4]

        Return:
            - (float) IoU value
        """
        xmin_box_1 = box_1[0] - box_1[2]/2.
        ymin_box_1 = box_1[1] - box_1[3]/2.
        xmax_box_1 = box_1[0] + box_1[2]/2.
        ymax_box_1 = box_1[1] + box_1[3]/2.

        xmin_box_2 = box_2[0] - box_2[2]/2.
        ymin_box_2 = box_2[1] - box_2[3]/2.
        xmax_box_2 = box_2[0] + box_2[2]/2.
        ymax_box_2 = box_2[1] + box_2[3]/2.

        xmin_intersection = max(xmin_box_1, xmin_box_2)
        ymin_intersection = max(ymin_box_1, ymin_box_2)
        xmax_intersection = min(xmax_box_1, xmax_box_2)
        ymax_intersection = min(ymax_box_1, ymax_box_2)

        if xmin_intersection > xmax_intersection or\
           ymin_intersection > ymax_intersection:
            return 0.0
        intersection = self.computeRectangleArea(xmin_intersection,
                                                 ymin_intersection,
                                                 xmax_intersection,
                                                 ymax_intersection)

        union = self.computeRectangleArea(xmin_box_1, ymin_box_1,
                                          xmax_box_1, ymax_box_1) +\
            self.computeRectangleArea(xmin_box_2, ymin_box_2,
                                      xmax_box_2, ymax_box_2) -\
            intersection

        return intersection/union
