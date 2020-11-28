#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to train SSD
"""

import cv2
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import random
import tensorflow as tf
from tqdm import tqdm


COLORS = [(0, 102, 51),
          (0, 153, 153),
          (102, 0, 102),
          (156, 76, 0)]


def pltPredOnImg(img, boxes, classes, scores, db_manager):
    """
    Method to plot boxes, classes and scores on images

    Args:
        - (tf.Tensor) image:  [Any, Any, Any]
        - (tf.Tensor) boxes of each box:  [N boxes, 4]
        - (tf.Tensor) classes of each box:  [N boxes]
        - (tf.Tensor) confidence of each box:  [N boxes]
        - VOC2012ManagerObjDetection class from data/
    """
    fig = plt.figure(figsize=(8, 8))

    img_pil = Image.fromarray(img.numpy().astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    for b, box in enumerate(boxes):
        color = random.choice(COLORS)
        box = tf.concat([box[:2] - box[2:] / 2,
                         box[:2] + box[2:] / 2], axis=-1)
        min_point = int(box[0] * 300), int(box[1] * 300)
        end_point = int(box[2] * 300), int(box[3] * 300)
        draw.rectangle((min_point, end_point), outline=color)
        draw.text((min_point[0]+5, min_point[1]+5),
                  "{}: {:.02f}".format(
                  list(db_manager.classes.keys())[classes[b]],
                  scores[b]),
                  fill=color)
    plt.imshow(img_pil)
    plt.show()


def pltPredGt(model, db_manager, images_names: str,
              score_threshold: float = 0.1, draw_default: bool = False):
    """
    Method to plot images with predicted and gt boxes

    Args:
        - SSD300 class from models/SSD300.py
        - VOC2012ManagerObjDetection class from data/
        - (list) images names
        - (float) display predicted boxes if score > to specified number
        - (bool) default: displaying default boxes selected as gt
    """
    imgs, boxes_gt, classes_gt = db_manager.getRawData(images_names)
    images, confs_gt, locs_gt = db_manager.getImagesAndGtSpeedUp(
        images_names, model.default_boxes)

    columns = int(len(imgs) / 2)
    rows = int(len(imgs) / 2) + 1
    size = 6
    fig = plt.figure(figsize=(columns*size, rows*size))

    num_pic = 1
    for i, img in enumerate(imgs):
        img_ssd = tf.expand_dims(img, 0)
        confs, locs = model(img_ssd)
        confs_pred = tf.concat(confs, axis=1)
        locs_pred = tf.concat(locs, axis=1)
        confs_pred = tf.math.softmax(confs_pred, axis=2)
        if draw_default:
            boxes, classes, scores = model.getPredictionsFromConfsLocs(
                tf.expand_dims(confs_gt[i], 0), tf.expand_dims(locs_gt[i], 0),
                score_threshold=score_threshold,
                box_encoding="corner", default=True)
        else:
            boxes, classes, scores = model.getPredictionsFromConfsLocs(
                confs_pred, locs_pred,
                score_threshold=score_threshold,
                box_encoding="corner")
        fig.add_subplot(rows, columns, num_pic)
        img_res = img*255
        img_pil = Image.fromarray(img_res.numpy().astype(np.uint8))
        draw = ImageDraw.Draw(img_pil)
        for b, box in enumerate(boxes_gt[i]):
            box = tf.concat([box[:2] - box[2:] / 2,
                             box[:2] + box[2:] / 2], axis=-1)
            min_point = int(box[0] * 300), int(box[1] * 300)
            end_point = int(box[2] * 300), int(box[3] * 300)
            draw.rectangle((min_point, end_point), outline='green')
            draw.text((min_point[0]+5, min_point[1]+5),
                      list(db_manager.classes.keys())[classes_gt[i][b]],
                      fill=(0, 255, 0, 0), width=30)
        for b, box in enumerate(boxes[0]):
            min_point = int(box[0] * 300), int(box[1] * 300)
            end_point = int(box[2] * 300), int(box[3] * 300)
            draw.rectangle((min_point, end_point), outline='red')
            draw.text((min_point[0]+5, min_point[1]+5),
                      list(db_manager.classes.keys())[classes[0][b]],
                      fill=(255, 0, 0, 0))
        plt.title("Epoch: {}".format(images_names[i]))
        plt.imshow(img_pil)
        num_pic += 1
    plt.show()


def pltPredOnVideo(model, db_manager, video_path: str, out_gif: str,
                   score_threshold: float = 0.6, start_idx: int = 0,
                   end_idx: int = -1):
    """
    Method to infer a model on a MP4 video
    Create a gif with drawn boxes, classes and confidence

    Args:
        - SSD300 class from models/SSD300.py
        - VOC2012ManagerObjDetection class from data/
        - (str) video path (MP4)
        - (str) video path (MP4)
        - (float) score threshold to draw a box
        - (int) start frame idx, default is 0
        - (int) end frame idx, default is -1
    """
    cap = cv2.VideoCapture(video_path)
    imgs = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if i <= start_idx:
            continue
        elif end_idx >= 0 and i > end_idx:
            break
        orig_height, orig_width = frame.shape[0], frame.shape[1]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_ssd = tf.image.resize(np.array(img), (300, 300)) / 255.
        img_ssd = tf.convert_to_tensor(img_ssd, dtype=tf.float32)
        img_ssd = tf.expand_dims(img_ssd, 0)
        confs, locs = model(img_ssd)
        confs_pred = tf.concat(confs, axis=1)
        locs_pred = tf.concat(locs, axis=1)
        confs_pred = tf.math.softmax(confs_pred, axis=2)

        boxes, classes, scores = model.getPredictionsFromConfsLocs(
                confs_pred, locs_pred,
                score_threshold=score_threshold,
                box_encoding="corner")

        boxes, classes, scores = model.nms(boxes[0], classes[0], scores[0])

        draw = ImageDraw.Draw(img)
        for b, box in enumerate(boxes[0]):
            color = random.choice(COLORS)
            min_point = int(box[0] * orig_width), int(box[1] * orig_height)
            end_point = int(box[2] * orig_width), int(box[3] * orig_height)
            draw.rectangle((min_point, end_point), outline=color)
            draw.text((min_point[0]+5, min_point[1]+5),
                      "{}: {:.02f}".format(
                      list(db_manager.classes.keys())[classes[0][b]],
                      scores[0][b]),
                      fill=color)
        imgs.append(img)
    imgs[0].save(out_gif, format='GIF', append_images=imgs[1:],
                 save_all=True, duration=100, loop=0)
