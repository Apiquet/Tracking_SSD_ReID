# SSD300 implementation with TensorFlow 2 and re-Identification module

SSD implementation from paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) and [official code](https://github.com/weiliu89/caffe/tree/ssd) (29 Dec 2016)

Full explanation of SSD implmentation [here](https://apiquet.com/2020/11/07/ssd300-implementation/)

Explanation of re-Id strategy [here](https://apiquet.com/2020/12/06/tracking-and-box-proposal/)

Overview of the recent object detection models is available [here](https://apiquet.com/2022/03/01/sota_object_detection/)

## Usage

* The notebook training.ipynb can be used to train the SSD300 model.
* The notebook SSD300_test.ipynb can be used to test the object detection model SSD300.
* The notebook tracker_test.ipynb can be used to test the tracker (SSD300 model + tracking module for multi-Object Tracking)

## Illustrations

### Naïve tracking from models/NaiveTracker and box proposal from models/SSD300

![Tracking on person and dog](imgs/person_dog_tracking.gif)

### Naïve tracking from models/NaiveTracker and box proposal from TF Hub SSD

![Tracking on person, dog and horses](imgs/horses_ssd_tfhub_tracking.gif)

## Overview

* Deep neural network used for object detection
* Default set of bounding boxes of different aspect ratios and scales (per feature map)
* At prediction time: scores for the presence of each object class in each default box
* also produces ajustements to the box to better match object shape (offset from the default box to the ground truth)
* Two possible input resolution implemente: 300x300, 512x512
* High speed network mainly due to the elimination of bounding box proposals and resampling stage
* Small convolutional filters are used to predict object categories and offsets in bounding box locations
* Detection at different scales are allowed by using these filters on differents feature maps

## Principal

* At training time, SSD needs an input images and ground truth boxes (for each object to detect)
* For each default box at each scale, offset (to the ground truth box) and confidence for all object categories are predicted
* Default boxes are selected as positives and the rest as negatives: if there are close to the ground truth (several boxes can be matched for one ground truth box)
* The loss is a weighted sum between localization loss (smooth L1) and confidence loss (softmax loss: softmax activation + cross entropy loss)

## The Model

* Produces fixed-size collection of bounding boxes and scores for the presence of object class instances
* Non-maximum suppression is added to produce final detection
* VGG-16 is used as base network for high quality image classification
* Auxiliary structure is then added to produce detections with:
    * Multi-scale feature maps: layers that decrease in size progressively (for multi-scale detection)
    * Convolutional predictor: each feature map can produce a fixed set of detection using a set of convolutional filters. For a feature layer of nxm with p channels, the basic element for predicting parameters of a detection is 3x3xp small kernel that produce a score for a category, or a shape of offset relative to the default box. At each of the mxn location where the kernel is applied, it produces an output value
    * Default boxes: the set of default bounding boxes is fixed in position. At each feature map, offsets are calculated relative to the default box shapes. So, as we have 4 offsets (center cx, cy and with/height) and a confidence for c classes, it leads to (c+4)*kmn outputs for mxn feature map with k the number of groundtruth.

## Training

* groudn truth information needs to be assigned to specific outputs in the fixed set of detector outputs
* Once the assignment is determined, the loss function and back propagation are applied end-to-end
* Also need to choose the set of default boxes and scales for detection
* Matching strategy: must determine which default boxes correspond to the ground truth one. This selection starts with the best jaccard overlap, then all boxes with a jaccard overlap > 0.5. This simplifies the learning problem, allowing the network to predict high scores for multiple overlapping default boxes rather than requiring to pick only the one with maximum overlap.
* The loss is the weighted sum of localization loss and confidence loss divided by the number of matched default boxes.
* If N=0, Lglobal = 0
* Lloc is a smooth L1 between predicted box and the ground truth (for center cx,cy and width/height)
* Lconf is a softmax loss (softmax activation + cross entropy loss: sum of negative logarithm of the probabilities)
* a was set to 1 (found with cross validation)
* possible aspect ratios: 1, 2, 3, 1/2, 1/3
* min size of default box over the network: 0.2, max: 0.9
* width = size * sqrt(ratio), height = size / sqrt(ratio)
* each default box has location: ((i+0.5)/|fk|, (j+0.5)/|fk|) with |fk| the size of the k-th square feature map (i and k are in [0, |fk|)
* Hard negative mining: after matching step most of the boxes are negatives, this leads to significant imbalance between positive and negative training examples. Instead of using all negative examples, they are sorted using the highest confidence loss for each default box and pick the top ones so that the ratio neg/pos is at most 3:1 (produce faster optimization and more stable training)
* Data augmentation: each image is randomly sampled by one of the following options:
	* use initial image
	* sample a patch so that the minimum jaccard overlap with objects is 0.1, 0.3, 0.5, 0.7 or 0.9
	* Randomly sample a patch
* The size of sampled patch is in [0.1, 1] of the original image size, aspect ratio is between 1/2 and 2. The patch is kept if the center of the ground truth is in it
* Each sampled patch is resized to fixed size and horizontally flipped with probability=0.5
* Other photo-metric distortions is applied

## Final model explanation

![SSD300](imgs/model_expand_concat_explained.png)
