# SSD implementation with TensorFlow 2
 
SSD implementation from paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) and [official code](https://github.com/weiliu89/caffe/tree/ssd) (29 Dec 2016)


## Overview

* Deep neural network used for object detection
* Default set of bounding boxes of different aspect ratios and scales (per feature map)
* At prediction time: scores for the presence of each object class in each default box
* also produces ajustements to the box the better match object shape (offset from the default box to the ground truth)
* Two possible input resolution implemente: 300x300, 512x512
* High speed network mainly due to the elimination of bounding box proposals and resampling stage
* Small convolutional filters are used to predict object categories and offsets in bounding box locations
* Detection at different scales are allowed by using these filters on differents feature maps

## Principal

* At training time, SSD needs an input images and ground truth boxes (for each object to detect)
* For each default box at each scale, offset (to the ground truth box) and confidence for all object categories are predicted
* Default boxes are selected as positives and the rest as negatives: if there are close to the ground truth (several boxes can be matched for one ground truth box)
* The loss is a weighted sum between localization loss (smooth L1) and confidence loss.

## The Model

* Produces fixed-size collection of bounding boxes and scores for the presence of object class instances
* Non-maximum suppression is added to produce final detection
* VGG-16 is used as base network for high quality image classification
* Auxiliary structure is then added to produce detections with:
	* Multi-scale feature maps: layers that decrease in size progressively (for multi-scale detection)
	* Convolutional predictor: each feature map can produce a fixed set of detection using a set of convolutional filters. For a feature layer of nxm with p channels, the basic element for predicting parameters of a detection is 3x3xp small kernel that produce a score for a category, or a shape of offset relative to the default box. At each of the mxn location where the kernel is applied, it produces an output value
	* Default boxes: the set of default bounding boxes is fixed in position. At each feature map, offsets are calculated relative to the default box shapes. So, as we have 4 offsets and a confidence for c classes, it leads to (c+4)*kmn outputs for mxn feature map with k the number of groundtruth.

## Training

