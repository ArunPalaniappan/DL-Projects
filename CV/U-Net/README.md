# U-Net: Convolutional Neural Network

## Contents

* [paper](paper.pdf)
* [implementation](model.py)

## Summary
This model is generally used for **Biomedical Image Segmentation**  

![Architecture](https://user-images.githubusercontent.com/89085916/173041942-a7c3e072-060d-4e87-9e91-0f6f9396e585.png)

### Network Architecture

The network architecture is illustrated in Figure 1. It consists of a contracting
path (left side) and an expansive path (right side). The contracting path follows
the typical architecture of a convolutional network. It consists of the repeated
application of two 3x3 convolutions (unpadded convolutions), each followed by
a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
for downsampling. At each downsampling step we double the number of feature
channels. Every step in the expansive path consists of an upsampling of the
feature map followed by a 2x2 convolution (“up-convolution”) that halves the
number of feature channels, a concatenation with the correspondingly cropped
feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in
every convolution. At the final layer a 1x1 convolution is used to map each 64-
component feature vector to the desired number of classes. In total the network
has 23 convolutional layers.  
[source](paper.pdf)
