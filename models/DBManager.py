#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pascal VOC2012 dataset manager
"""

import pandas as pd
import numpy as np
import os
from shutil import copyfile


class DBManager():

    def __init__(self):
        super(DBManager, self).__init__()

    def getTrainValDfClass(self, voc2012path: str, class_name: str):
        """
        Method to get train and validation images name in two dataframes

        Args:
            - (str) VOC2012 path
            - (str) Class name

        Return:
            - (pandas.DataFrame) Df with train data [imgName, ClassName]
            - (pandas.DataFrame) Df with valid data [imgName, ClassName]
        """
        train_filepath = voc2012path + '/ImageSets/Main/' +\
            class_name + "_train.txt"
        val_filepath = voc2012path + '/ImageSets/Main/' +\
            class_name + "_val.txt"

        train_df = pd.read_csv(train_filepath, delimiter=r"\s+", header=None)
        train_df.columns = ["imgName", "value"]
        train_df = train_df.drop(train_df[train_df.value == -1].index)
        train_df["className"] = class_name
        train_df = train_df.drop(["value"], axis=1)

        val_df = pd.read_csv(val_filepath, delimiter=r"\s+", header=None)
        val_df.columns = ["imgName", "value"]
        val_df = val_df.drop(val_df[val_df.value == -1].index)
        val_df["className"] = class_name
        val_df = val_df.drop(["value"], axis=1)

        return train_df, val_df

    def getCombinedClassesDf(self,
                             voc2012path: str,
                             classes: list,
                             combinedName: str):
        """
        Method to combine classes to a new one.
        For instance, to convert bicycle and bottle to new class 'obj' call:
        getCombinedClassesDf(voc2012path, ['bicycle', 'bottle', 'obj')

        Args:
            - (str)  VOC2012 path
            - (list) List of classes to combine
            - (str)  new class name

        Return:
            - (pandas.DataFrame) Df with train data [imgName, combinedName]
            - (pandas.DataFrame) Df with valid data [imgName, combinedName]
        """
        classes_train_df = pd.DataFrame(columns=["imgName", "className"])
        classes_val_df = pd.DataFrame(columns=["imgName", "className"])

        for class_el in classes:
            train_df, val_df = self.getTrainValDfClass(voc2012path, class_el)
            classes_train_df = classes_train_df.append(train_df)
            classes_val_df = classes_val_df.append(val_df)

        classes_train_df.className = combinedName
        classes_val_df.className = combinedName
        return classes_train_df, classes_val_df

    def shuffleDf(self, df):
        return df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

    def createDb(self,
                 voc2012path: str,
                 df: pd.DataFrame,
                 output_path: str,
                 db_name: str = "DB"):
        db_path = output_path + '/' + db_name + '/'
        os.makedir(db_path)
        for category in df.className.unique():
            os.makedir(db_path + 'class_' + category)

        images_path = voc2012path + "JPEGImages/"
        for index, row in df.iterrows():
            original_filepath = voc2012path + '/' + row.imgName + '.jpg'
            db_folder = db_path + 'class_' + row.className + '/'
            db_filepath = db_folder + row.className + '_image_' +\
                str(index).zfill(8) + '.jpg'
            copyfile(original_filepath, db_filepath)
