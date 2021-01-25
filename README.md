<h1 align="center">
  <p align="center">SpineAI Paper with Code</p>
  <img src="imgs/spineAI-logo.png" alt="SpineAI-logo" height="150">
</h1>

This repository contains **code for our paper submitted to Radiology**:
> Deep learning models for automated detection and classification of central canal, lateral recess, and neural foraminal stenosis on lumbar spine MRI

In the paper, we present a two-step system to automatically detect and classify lumbar spinal stenosis on MRI. The steps and implementations of our system are documented in their respective folders.

- **Object Detection**: Based on Tensorflow object detection API, we pick Faster R-CNN with Resnet101 architecture pre-trained on COCO dataset to detect region of interest (ROI).
- **Classification**: CNN architecture consisting of six convolutional layers, outputing four-grade classification predictions.
- **Interpretatbiliy**: We use the Tensorflow officitual tutorial on Integrated Gradients as the basis.
