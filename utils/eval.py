#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to train SSD
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm
import tensorflow as tf


def pltPredGt(model, db_manager, images_names, score_threshold=0.1):
    """
    Method to plot images with predicted and gt boxes

    Args:
        - SSD300 class from models/SSD300.py
        - VOC2012ManagerObjDetection class from data/
        - (list) images names
        - (float) display predicted boxes if score > to specified number
    """
    imgs, boxes_gt, classes_gt = db_manager.getRawData(images_names)

    columns = int(len(imgs) / 2)
    rows = int(len(imgs) / 2) + 1
    size = 3
    fig = plt.figure(figsize=(columns*size, rows*size))

    num_pic = 1
    for i, img in enumerate(imgs):
        img_ssd = tf.expand_dims(img, 0)
        confs, locs = model(img_ssd)
        confs_pred = tf.concat(confs, axis=1)
        locs_pred = tf.concat(locs, axis=1)
        confs_pred = tf.math.softmax(confs_pred, axis=2)
        classes, boxes = model.getPredictionsFromConfsLocs(
            confs_pred, locs_pred,
            score_threshold=score_threshold,
            box_encoding="corner")
        fig.add_subplot(rows, columns, num_pic)
        img_res = img*255
        img_pil = Image.fromarray(img_res.numpy().astype(np.uint8))
        draw = ImageDraw.Draw(img_pil)
        for b, box in enumerate(boxes[0]):
            min_point = int(box[0] * 300), int(box[1] * 300)
            end_point = int(box[2] * 300), int(box[3] * 300)
            draw.rectangle((min_point, end_point), outline='red')
            draw.text((min_point[0]+5, min_point[1]+5),
                      list(db_manager.classes.keys())[classes[0][b]],
                      fill=(255, 0, 0, 0))
        for b, box in enumerate(boxes_gt[i]):
            box = tf.concat([box[:2] - box[2:] / 2,
                             box[:2] + box[2:] / 2], axis=-1)
            min_point = int(box[0] * 300), int(box[1] * 300)
            end_point = int(box[2] * 300), int(box[3] * 300)
            draw.rectangle((min_point, end_point), outline='green')
            draw.text((min_point[0]+5, min_point[1]+5),
                      list(db_manager.classes.keys())[classes_gt[i][b]],
                      fill=(0, 255, 0, 0))
        plt.title("Epoch: {}".format(images_names[i]))
        plt.imshow(img_pil)
        num_pic += 1
    plt.show()
