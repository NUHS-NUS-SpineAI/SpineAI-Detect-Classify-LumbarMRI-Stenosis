# ROI Classification Model

[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

![train-classifier](../imgs/Rad_Fig_v4_Classifier.png)

## Training Setup

Models are implemented with Tensorflow 1.15 and trained on NVIDIA GeForce RTX/GTX GPU devices with CUDA version 9 or 10.

Classification model is developed with Keras sequential class, consisting of a stack of six convolution layers and two fully-connected layers that end with softmax activation for multi-class classification.

## Weighted Categorical Cross-entropy Loss

We use a cost-sensitive loss for training due to the highly imbalanced data distirbution.

See Keras Github for more detailed discussion on the implementation:
https://github.com/keras-team/keras/issues/2115


## References
- Keras official tutorial on image classification models: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

