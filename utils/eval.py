#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to train SSD
"""

import cv2
from glob import glob
import imageio
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import random
import tensorflow as tf
from tqdm import tqdm


nb_colors = 100
COLORS = [(random.randint(50, 200),
           random.randint(50, 200),
           random.randint(50, 200)) for i in range(nb_colors)]


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
                   end_idx: int = -1, nms: bool = True, skip: int = 1,
                   tracker=None, resize: tuple = None, fps: int = 30,
                   lifespan_thres: int = 5):
    """
    Method to infer a model on a MP4 video
    Create a gif with drawn boxes, classes and confidence
    Infer SSD from models/SSD300.py on each image in sequence
    Tracker from models.NaiveTracker can be used to keep IDs on the subjects

    Args:
        - SSD300 class from models/SSD300.py
        - VOC2012ManagerObjDetection class from data/
        - (str) video path (MP4)
        - (str) out_gif: output path (.gif)
        - (float) score_threshold: score threshold to draw a box
        - (int) start_idx: start frame idx, default is 0
        - (int) end_idx: end frame idx, default is -1
        - (bool) nms: use non-maximum suppression
        - (int) skip: idx%skip != 0 is skipped
        - Tracker: models.NaiveTracker instance if wanted
        - (tuple) resize: target resolution for the gif
        - (int) fps: fps of the output gif
        - (int) lifespan_thres: min times subject was found before displaying
    """
    cap = cv2.VideoCapture(video_path)
    imgs = []
    i = 0
    number_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_idx != -1:
        number_of_frame = end_idx
    for _ in tqdm(range(number_of_frame)):
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if i <= start_idx:
            continue
        elif end_idx >= 0 and i > end_idx:
            break
        if i % skip != 0:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_ssd = tf.image.resize(np.array(img), (300, 300)) / 255.
        img_ssd = tf.convert_to_tensor(img_ssd, dtype=tf.float32)
        img_ssd = tf.expand_dims(img_ssd, 0)

        if resize:
            img.thumbnail(resize, Image.ANTIALIAS)
        orig_height, orig_width = img.size[1], img.size[0]
        line_width = int(0.006 * orig_width)
        font = ImageFont.truetype("arial.ttf", line_width*10)

        confs, locs = model(img_ssd)
        confs_pred = tf.concat(confs, axis=1)
        locs_pred = tf.concat(locs, axis=1)
        confs_pred = tf.math.softmax(confs_pred, axis=2)

        boxes, classes, scores = model.getPredictionsFromConfsLocs(
                confs_pred, locs_pred,
                score_threshold=score_threshold,
                box_encoding="center")
        if nms:
            boxes, classes, scores = model.recursive_nms(boxes[0],
                                                         classes[0],
                                                         scores[0])
        else:
            boxes, classes, scores = boxes[0], classes[0], scores[0]
        if tracker:
            identity, lifespan = tracker(classes, boxes)

        draw = ImageDraw.Draw(img)
        for b, box in enumerate(boxes):
            if lifespan[b] < lifespan_thres:
                continue
            if tracker:
                color = random.seed(identity.numpy()[b] * 10)
            color = random.choice(COLORS)
            box = tf.concat([box[:2] - box[2:] / 2,
                             box[:2] + box[2:] / 2], axis=-1)

            min_point = int(box[0] * orig_width), int(box[1] * orig_height)
            end_point = int(box[2] * orig_width), int(box[3] * orig_height)
            draw.rectangle((min_point, end_point), outline=color,
                           width=line_width)
            text = "{}: {:.02f}".format(
                list(db_manager.classes.keys())[classes[b]], scores[b])
            draw_underlined_text(draw, (min_point[0]+5, min_point[1]+2), text,
                                 font, fill=color, line_width=line_width)
            if tracker:
                draw.text((min_point[0]+1, min_point[1]-22),
                          f"ID: {identity[b]}", font=font, fill=color)
        imgs.append(img)
    imgs[lifespan_thres].save(out_gif, format='GIF',
                              append_images=imgs[lifespan_thres+1:],
                              save_all=True, loop=0)
    gif = imageio.mimread(out_gif)
    imageio.mimsave(out_gif, gif, fps=fps)


def draw_underlined_text(draw, pos, text, font, fill, line_width=2):
    twidth, theight = draw.textsize(text, font=font)
    lx, ly = pos[0], pos[1] + theight
    draw.text(pos, text, font=font, fill=fill)
    draw.line((lx, ly, lx + twidth, ly), fill=fill, width=line_width)


def pltPredOnVideoTfHub(model, video_path: str, out_gif: str,
                        score_threshold: float = 0.6, start_idx: int = 0,
                        end_idx: int = -1, skip: int = 1, tracker=None,
                        resize: tuple = None, fps: int = 30,
                        input_shape: tuple = (640, 640),
                        targets: list = None, lifespan_thres: int = 5):
    """
    Method to infer a model on a MP4 video
    Create a gif with drawn boxes, classes and confidence
    Infer model from TensorFlow Hub on each image in sequence
    Tracker from models.NaiveTracker can be used to keep IDs on the subjects

    Args:
        - SSD640 from tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1
        - (str) video path (MP4)
        - (str) out path (.gif file)
        - (float) score threshold to draw a box
        - (int) start frame idx, default is 0
        - (int) skip: idx%skip != 0 is skipped
        - Tracker: models.NaiveTracker instance if wanted
        - (tuple) resize: target resolution for the gif
        - (int) fps: fps of the output gif
        - (tuple) input_shape: model input shape
        - (list) targets: list of the target class name like ['Dog','Person']
        - (int) lifespan_thres: min times subject was found before displaying
    """
    cap = cv2.VideoCapture(video_path)
    imgs = []
    i = 0
    number_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_idx != -1:
        number_of_frame = end_idx
    for _ in tqdm(range(number_of_frame)):
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if i <= start_idx:
            continue
        elif end_idx >= 0 and i > end_idx:
            break
        if i % skip != 0:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if resize:
            img.thumbnail(resize, Image.ANTIALIAS)
        orig_height, orig_width = img.size[1], img.size[0]
        line_width = int(0.00565 * orig_width)
        font = ImageFont.truetype("arial.ttf", line_width*9)

        img_ssd = tf.image.resize(np.array(img), input_shape)
        img_ssd = tf.convert_to_tensor(img_ssd, dtype=tf.float32)/255.

        img_ssd = tf.image.convert_image_dtype(
            img_ssd, tf.float32)[tf.newaxis, ...]
        result = model(img_ssd)
        result = {key: value.numpy() for key, value in result.items()}

        scores_tokeep = result["detection_scores"] > score_threshold

        boxes = result["detection_boxes"][scores_tokeep]

        boxes = tf.concat([[boxes[:, 1]], [boxes[:, 0]],
                           [boxes[:, 3]], [boxes[:, 2]]],
                          axis=0)
        boxes = tf.transpose(boxes)

        boxes = tf.concat([
            (boxes[:, :2] + boxes[:, 2:]) / 2,
            boxes[:, 2:] - boxes[:, :2]], axis=-1)

        classes = result["detection_class_labels"][scores_tokeep]
        classes_name = result["detection_class_entities"][scores_tokeep]
        scores = result["detection_scores"][scores_tokeep]

        if targets is not None:
            idx_targets = np.zeros(classes_name.shape)
            for target in targets:
                idx_targets += classes_name == target.encode("ascii")
            idx_targets = idx_targets == 1
            classes = classes[idx_targets]
            classes_name = classes_name[idx_targets]
            scores = scores[idx_targets]
            boxes = boxes[idx_targets]

        if tracker:
            identity, lifespan = tracker(classes, boxes)

        draw = ImageDraw.Draw(img)
        for b, box in enumerate(boxes):
            if lifespan[b] < lifespan_thres:
                continue
            if tracker:
                color = random.seed(identity.numpy()[b] * 10)
            color = random.choice(COLORS)
            box = tf.concat([box[:2] - box[2:] / 2,
                             box[:2] + box[2:] / 2], axis=-1)
            min_point = int(box[0] * orig_width), int(box[1] * orig_height)
            end_point = int(box[2] * orig_width), int(box[3] * orig_height)
            draw.rectangle((min_point, end_point), outline=color,
                           width=line_width)
            text = "{}: {:.02f}".format(classes_name[b].decode("ascii"),
                                        scores[b])
            draw_underlined_text(draw, (min_point[0]+5, min_point[1]+2), text,
                                 font, fill=color, line_width=line_width)
            if tracker:
                draw.text((min_point[0]+1, min_point[1]-18),
                          f"ID: {identity[b]}", font=font, fill=color)
        imgs.append(img)
    imgs[lifespan_thres].save(out_gif, format='GIF',
                              append_images=imgs[lifespan_thres+1:],
                              save_all=True, loop=0)
    gif = imageio.mimread(out_gif)
    imageio.mimsave(out_gif, gif, fps=fps)
