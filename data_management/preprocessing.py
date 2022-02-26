#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VOC2012 preprocessing to speed up training
"""
import sys

from .VOC2012ManagerObjDetection import VOC2012ManagerObjDetection
from models.SSD300 import SSD300

import numpy as np
from tqdm import tqdm
import os

from glob import glob
import tensorflow as tf

sys.path.insert(1, '../')


def saveGTdata(voc2012path, output_path):
    """
    Method to save data in a proper format to train the network

    Args:
        - (str) path to save data
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    db_manager = VOC2012ManagerObjDetection(voc2012path + '/',
                                            batch_size=32, floatType=32)
    SSD300_model = SSD300(21, floatType=32)
    for i, batch in enumerate(tqdm(db_manager.batches)):
        # get data from batch
        imgs, confs, locs = db_manager.getImagesAndGtSpeedUp(
            batch, SSD300_model.default_boxes)
        np.save(output_path + "/imgs_{:05d}.npy".format(i),
                imgs, allow_pickle=True)
        np.save(output_path + "/confs_{:05d}.npy".format(i),
                confs, allow_pickle=True)
        np.save(output_path + "locs_{:05d}.npy".format(i),
                locs, allow_pickle=True)


def loadGTdata(path, nb_data_to_load=-1):
    """
    Method to load needed data for training
    N: batch size
    B: batch size
    O: number of objects in a given image
    D: number of default boxes

    Args:
        - (str) path to load

    Return:
        - (list of tf.Tensor) images (B, 300, 300, 3)
        - (list of tf.Tensor) confs gt (N, B, O, D)
        - (list of tf.Tensor) locs gt (N, B, O, D, 4)
    """
    imgs = []
    for batch in tqdm(sorted(glob(path + "/imgs*.npy"))[:nb_data_to_load]):
        # get data from batch
        imgs.append(tf.convert_to_tensor(np.load(batch, allow_pickle=True)))

    confs = []
    for batch in tqdm(sorted(glob(path + "/confs*.npy"))[:nb_data_to_load]):
        # get data from batch
        confs.append(tf.convert_to_tensor(np.load(batch, allow_pickle=True)))

    locs = []
    for batch in tqdm(sorted(glob(path + "/locs*.npy"))[:nb_data_to_load]):
        # get data from batch
        locs.append(tf.convert_to_tensor(np.load(batch, allow_pickle=True)))

    return imgs, confs, locs


def loadSpecificGTdata(path, idx):
    """
    Method to load a particular batch
    B: batch size
    N: number of objects in a given image
    D: number of default boxes

    Args:
        - (str) path to load

    Return:
        - (list of tf.Tensor) images (B, 300, 300, 3)
        - (list of tf.Tensor) confs gt (B, N, D)
        - (list of tf.Tensor) locs gt (B, N, D, 4)
    """
    batch = sorted(glob(path + "/imgs*.npy"))[idx]
    imgs = tf.convert_to_tensor(np.load(batch, allow_pickle=True))

    batch = sorted(glob(path + "/confs*.npy"))[idx]
    confs = tf.convert_to_tensor(np.load(batch, allow_pickle=True))

    batch = sorted(glob(path + "/locs*.npy"))[idx]
    locs = tf.convert_to_tensor(np.load(batch, allow_pickle=True))

    return imgs, confs, locs
