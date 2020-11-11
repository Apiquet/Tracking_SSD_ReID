#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pascal VOC2012 dataset manager
"""

import numpy as np
import os
import tensorflow as tf
import xml.etree.ElementTree as ET


class VOC2012ManagerObjDetection():

    def __init__(self, path):
        super(VOC2012ManagerObjDetection, self).__init__()
        self.path = path
        self.img_resolution = (300, 300, 3)
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

    def getImages(self, images_name: list):
        """
        Method to get images in a tensor shape

        Args:
            - (list) images name without extension

        Return:
            - (tf.Tensor) Tensor of shape:
                [number of images, self.img_resolution]
        """
        images = []
        annotations = []
        for img in images_name:
            image = tf.keras.preprocessing.image.load_img(
                self.images_path + img + ".jpg")
            resolution = image.shape
            image = smart_resize(image, self.img_resolution)
            images_array = \
                tf.keras.preprocessing.image.img_to_array(image) / 255.
            images.append(images_array)
            boxes, classes = self.getAnnotations(img, resolution)
        return tf.convert_to_tensor(images, dtype=tf.float32)

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
            self.annotations_path + img + ".xml").findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / resolution.shape[0]
            ymin = float(bndbox.find('ymin').text) / resolution.shape[1]
            xmax = float(bndbox.find('xmax').text) / resolution.shape[0]
            ymax = float(bndbox.find('ymax').text) / resolution.shape[1]
            # calculate cx, cy, width, height
            width = xmax-xmin
            height = ymax-ymin

            boxes.append([xmin+width/2., ymin+height/2., width, height])

            # get class
            name = obj.find('name').text.lower().strip()
            classes.append(self.classes[name])

        return tf.convert_to_tensor(boxes_per_images, dtype=tf.float16),\
            tf.convert_to_tensor(classes_per_images, dtype=tf.uint8)
