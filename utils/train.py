#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to train SSD
"""

import tensorflow as tf
from tqdm import tqdm


def train(model, optimizer, db_manager, batches, weights_path,
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
        for batch in tqdm(batches):
            # get data from batch
            images, confs_gt, locs_gt = db_manager.getImagesAndGtSpeedUp(
                batch, model.default_boxes)

            # get predictions and losses
            with tf.GradientTape() as tape:
                confs_pred, locs_pred = model(images)
                # concat tensor to delete the dimension of feature maps
                confs_pred = tf.concat(confs_pred, axis=1)
                locs_pred = tf.concat(locs_pred, axis=1)

                # calculate loss
                confs_loss, locs_loss = model.calculateLoss(
                    confs_pred, confs_gt, locs_pred, locs_gt)
                loss = confs_loss + 1*locs_loss

            # back propagation
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
        if epoch % inter_save == 0:
            model.save_weights(weights_path +
                               "/ssd_weights_epoch_{:03d}.h5".format(epoch))
