# SSD implementation with TensorFlow 2
 
SSD implementation from paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) and [official code](https://github.com/weiliu89/caffe/tree/ssd) (29 Dec 2016)


* Deep neural network used for object detection
* Default set of bounding boxes of different aspect ratios and scales (per feature map)
* At prediction time: scores for the presence of each object class in each default box
* also produces ajustements to the box the better match object shape (offset from the default box to the groundtruth)
* Two possible input resolution implemente: 300x300, 512x512
* High speed network mainly due to the elimination of bounding box proposals and resampling stage
* Small convolutional filters are used to predict object categories and offsets in bounding box locations
* Detection at different scales are allowed by using these filters on differents feature maps
