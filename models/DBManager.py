#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pascal VOC2012 dataset manager
"""

import pandas as pd


class DBManager():

    def __init__(self):
        super(DBManager, self).__init__()

    def getTrainValDfClass(self, path, class_name=""):
        train_filepath = path + '/ImageSets/Main/' + class_name + "_train.txt"
        val_filepath = path + '/ImageSets/Main/' + class_name + "_val.txt"

        train_df = pd.read_csv(train_filepath, delimiter=r"\s+", header=None)
        train_df.columns = ["ImgName", "Value"]
        train_df = train_df.drop(train_df[train_df.values == -1].index)

        val_df = pd.read_csv(val_filepath, delimiter=r"\s+", header=None)
        val_df.columns = ["ImgName", "Value"]
        val_df = val_df.drop(val_df[val_df.values == -1].index)

        return train_df, val_df

        
