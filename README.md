<h1 align="center">
  <p align="center">SpineAI Paper with Code</p>
  <img src="imgs/spineAI-logo.png" alt="SpineAI-logo" height="150">
</h1>


## About

This repository contains **code for our paper submitted to Radiology**:

> "Deep learning models for automated detection and classification of central canal, lateral recess, and neural foraminal stenosis on lumbar spine MRI"

In the paper, we develop and apply AI techniques to automatically detect and classify lumbar spinal stenosis on MRI images. The setups and implementations of our system are documented in their respective folders:

- [**Object Detection**](Object-Detection/): Based on Tensorflow object detection API, we pick Faster R-CNN with Resnet101 architecture pre-trained on COCO dataset to detect region of interest (ROI).
- [**Classification**](Classification/): CNN architecture consisting of six convolutional layers, outputing four-grade classification predictions.
- [**Inference**](Inference/): Predict relevant spinal regions (ROI) and infer the disease grades, automatic generation of XML outputs and bounding boxes with probability overlays.
- [**Performance Analysis**](Performance-Analysis/): Confusion matrices and preliminary inter-rater agreement calculations. Analysis for statistical evaluation during model development.
- [**Interpretability**](Interpretability/): Explainable AI technique using Integrated Gradients provided by Tensorflow Core.


## Contact

TBA