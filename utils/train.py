#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to train SSD
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def train(model, optimizer, db_manager, imgs, confs, locs, weights_path,
          num_epoch=250, inter_save=5):
    """
    Method to train SSD architecture

    Args:
        - SSD300 class from models/SSD300.py
        - (tf.keras.optimizers) optimizer to use
        - VOC2012ManagerObjDetection class from data/
        - (str) path to save weights
        - (list) batches of images name
        - (int) number of epochs
        - (int) interval to save weights
    """
    for epoch in range(num_epoch):
        losses = []
        for i in tqdm(range(len(imgs))):
            # get data from batch
            images, confs_gt, locs_gt = imgs[i], confs[i], locs[i]

            # get predictions and losses
            with tf.GradientTape() as tape:
                confs_pred, locs_pred = model(images)
                # concat tensor to delete the dimension of feature maps
                confs_pred = tf.concat(confs_pred, axis=1)
                locs_pred = tf.concat(locs_pred, axis=1)

                # calculate loss
                confs_loss, locs_loss = model.calculateLoss(
                    confs_pred, confs_gt, locs_pred, locs_gt)
                loss = confs_loss + 1*locs_loss  # alpha equals 1
                l2 = [tf.nn.l2_loss(t)
                      for t in SSD300_model.trainable_variables]
                loss = loss + 0.001 * tf.math.reduce_sum(l2)
                losses.append(loss)

            # back propagation
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
        print("Mean loss: {} on epoch {}".format(np.mean(losses), epoch))
        if epoch % inter_save == 0:
            model.save_weights(weights_path +
                               "/ssd_weights_epoch_{:03d}.h5".format(epoch))
