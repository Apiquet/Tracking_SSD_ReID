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

    def __init__(self, path, trainRatio=0.7, batch_size=32):
        super(VOC2012ManagerObjDetection, self).__init__()
        self.path = path
        self.img_resolution = (300, 300)
        self.classes = {'undefined': 0,
                        'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
                        'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9,
                        'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13,
                        'motorbike': 14, 'person': 15, 'pottedplant': 16,
                        'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}
        self.images_path = path + "/JPEGImages/"
        self.annotations_path = path + "/Annotations/"
        self.images_name = [im.replace(".jpg", "")
                            for im in os.listdir(self.images_path)
                            if os.path.isfile(os.path.join(self.images_path,
                                                           im))]
        self.number_samples = len(self.images_name)
        self.train_samples = int(self.number_samples * trainRatio)
        self.train_set = self.images_name[:self.train_samples - self.train_samples%batch_size]
        self.val_set = self.images_name[self.train_samples:]
        self.batches = [self.train_set[i:i + batch_size]
                        for i in range(0, len(self.train_set), batch_size)]

    def getRawData(self, images_name: list):
        """
        Method to get images and annotations from a list of images name

        Args:
            - (list) images name without extension

        Return:
            - (tf.Tensor) Images of shape:
                [number of images, self.img_resolution]
            - (list of tf.Tensor) Boxes of shape:
                [number of images, number of objects, 4]
            - (list tf.Tensor) Classes of shape:
                [number of images, number of objects]
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
            - (tf.Tensor) Classes of shape: [number of objects]
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

    def getImagesAndGt(self, images_name: list, default_boxes: list):
        """
        Method to get the groud truth for confidence and localization
        S: number of stage
        D: number of default boxes
        B: batch size (number of images)

        Args:
            - (list) images name without extension
            - (tf.Tensor) default boxes per stage: [D, 4]
                4 parameters: cx, cy, w, h

        Return:
            - (tf.Tensor) Images of shape:
                [number of images, self.img_resolution]
            - (tf.Tensor) confs ground truth: [B, D]
            - (tf.Tensor) locs ground truth: [B, D, 4]
        """
        images, boxes, classes = self.getRawData(images_name)
        gt_confs = []
        gt_locs = []
        for i, gt_boxes_img in tqdm(enumerate(boxes)):
            gt_confs_per_default_box = []
            gt_locs_per_default_box = []
            for d, default_box in enumerate(default_boxes):
                for g, gt_box in enumerate(gt_boxes_img):
                    iou = self.computeJaccardIdx(gt_box, default_box)
                    gt_conf = self.classes['undefined']
                    gt_loc = tf.Variable([0.0, 0.0, 0.0, 0.0])
                    if iou >= 0.5:
                        gt_conf = tf.Variable(classes[i][g])
                        gt_loc = self.getLocOffsets(gt_box, default_box)
                gt_confs_per_default_box.append(gt_conf)
                gt_locs_per_default_box.append(gt_loc)
            gt_confs.append(gt_confs_per_default_box)
            gt_locs.append(gt_locs_per_default_box)

        return images,\
            tf.convert_to_tensor(gt_confs, dtype=tf.uint8),\
            tf.convert_to_tensor(gt_locs, dtype=tf.float16)

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

    def getLocOffsets(self, box_gt: tf.Tensor, box_pred: tf.Tensor):
        """
        Method to get the offset from box_pred to box_gt on cx, cy, w, h

        Args:
            - (tf.Tensor) box with 4 parameters: cx, cy, w, h [4]
            - (tf.Tensor) box with 4 parameters: cx, cy, w, h [4]

        Return:
            - (tf.Tensor) offset for the 4 parameters: cx, cy, w, h [4]
        """

        return tf.Variable([box_gt[0] - box_pred[0],
                            box_gt[1] - box_pred[1],
                            box_gt[2] - box_pred[2],
                            box_gt[3] - box_pred[3]])

    def computeJaccardIdxSpeedUp(self, gt_box: tf.Tensor,
                                 default_boxes: tf.Tensor,
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
        default_boxes = tf.concat([
            default_boxes[:, :2] - default_boxes[:, 2:] / 2,
            default_boxes[:, :2] + default_boxes[:, 2:] / 2], axis= -1)
        gt_box = tf.concat([gt_box[:2] - gt_box[2:] / 2,
                            gt_box[:2] + gt_box[2:] / 2], axis= -1)
        gt_box = tf.expand_dims(gt_box, 0)
        gt_box = tf.repeat(gt_box, repeats=[default_boxes.shape[0]], axis=0)

        # compute intersection
        inter_xymin = tf.math.maximum(default_boxes[:, :2],
                                          gt_box[:, :2])
        inter_xymax = tf.math.minimum(default_boxes[:, 2:],
                                          gt_box[:, 2:])
        inter_width_height = tf.clip_by_value(inter_xymax - inter_xymin,
                                              0.0, 300.0)
        inter_area = inter_width_height[:, 0] * inter_width_height[:, 1]

        # compute area of the boxes
        gt_box_width_height =\
            tf.clip_by_value(gt_box[:, 2:] - gt_box[:, :2],
                             0.0, 300.0)
        gt_box_width_height_area = gt_box_width_height[:, 0] *\
                                   gt_box_width_height[:, 1]

        default_boxes_width_height =\
            tf.clip_by_value(default_boxes[:, 2:] - default_boxes[:, :2],
                             0.0, 300.0)
        default_boxes_width_height_area = default_boxes_width_height[:, 0] *\
                                          default_boxes_width_height[:, 1]

        # compute iou
        iou = inter_area / (gt_box_width_height_area +
                            default_boxes_width_height_area - inter_area)
        return tf.dtypes.cast(iou >= iou_threshold, tf.uint8)

    def getLocOffsetsSpeedUp(self, gt_box: tf.Tensor, iou_bin: tf.Tensor,
                             default_boxes: tf.Tensor):
        """
        Method to get the offset from default boxes to box_gt on cx, cy, w, h
        where iou_idx is 1
        D: number of default boxes

        Args:
            - (tf.Tensor) box with 4 parameters: cx, cy, w, h [4]
            - (tf.Tensor) 1 if iou > threshold, 0 otherwise [D]
            - (tf.Tensor) default boxes with 4 parameters: cx, cy, w, h [D, 4]

        Return:
            - (tf.Tensor) offset for the 4 parameters: cx, cy, w, h [4]
        """
        offsets = tf.concat([gt_box[0] - default_boxes[:, 0],
                             gt_box[1] - default_boxes[:, 1],
                             gt_box[2] - default_boxes[:, 2],
                             gt_box[3] - default_boxes[:, 3]], axis = 0)
        iou_bin = tf.expand_dims(iou_bin, 1)
        iou_bin = tf.repeat(iou_bin, repeats=[4], axis=1)
        offsets = default_boxes * tf.dtypes.cast(iou_bin, tf.float16)
        return offsets

    def getImagesAndGtSpeedUp(self, images_name: list, default_boxes: list):
        """
        Method to get the groud truth for confidence and localization
        S: number of stage
        D: number of default boxes
        B: batch size (number of images)

        Args:
            - (list) images name without extension
            - (tf.Tensor) default boxes per stage: [D, 4]
                4 parameters: cx, cy, w, h

        Return:
            - (tf.Tensor) Images of shape:
                [number of images, self.img_resolution]
            - (tf.Tensor) confs ground truth: [B, D]
            - (tf.Tensor) locs ground truth: [B, D, 4]
        """
        images, boxes, classes = self.getRawData(images_name)
        gt_confs = []
        gt_locs = []
        for i, gt_boxes_img in tqdm(enumerate(boxes)):
            gt_confs_per_image = tf.zeros([len(default_boxes)], tf.uint8)
            gt_locs_per_image = tf.zeros([len(default_boxes), 4], tf.float16)
            for g, gt_box in enumerate(gt_boxes_img):
                iou_bin = self.computeJaccardIdxSpeedUp(gt_box,
                                                        default_boxes,
                                                        0.5)
                gt_confs_per_image = gt_confs_per_image +\
                    iou_bin * classes[i][g]
                gt_locs_per_image = gt_locs_per_image +\
                    self.getLocOffsetsSpeedUp(gt_box, iou_bin, default_boxes)
            gt_confs.append(gt_confs_per_image)
            gt_locs.append(gt_locs_per_image)

        return images,\
            tf.convert_to_tensor(gt_confs, dtype=tf.uint8),\
            tf.convert_to_tensor(gt_locs, dtype=tf.float16)
