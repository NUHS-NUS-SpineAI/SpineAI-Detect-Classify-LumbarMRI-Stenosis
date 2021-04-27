<h1 align="center">
  <p align="center">SpineAI Paper with Code</p>
  <img src="imgs/spineAI-logo.png" alt="SpineAI-logo" height="150">
</h1>


## 📄 About

This repository contains **code for our paper submitted to Radiology**:

> "Deep learning model for automated detection and classification of central canal, lateral recess and neural foraminal stenosis on lumbar spine MRI" (in press)

In the paper, we develop and apply AI techniques to automatically detect and classify lumbar spinal stenosis on MRI images.

## 🎓 What’s In This Repo

The setups and implementations of our system are documented in their respective folders:

- [**Object Detection**](Object-Detection/)

Based on Tensorflow object detection API, we pick Faster R-CNN with Resnet101 architecture pre-trained on COCO dataset to detect region of interest (ROI).

- [**Classification**](Classification/)

CNN architecture consisting of six convolutional layers, outputing four-grade classification predictions.

- [**Inference**](Inference/)

Predict relevant spinal regions (ROI) and infer the disease grades, automatic generation of XML outputs and bounding boxes with probability overlays.

- [**Interpretability**](Interpretability/)

Explainable AI technique using Integrated Gradients provided by Tensorflow Core.


## 🤝 Referencing and Citing SpineAI

If you find our work useful in your research and would like to cite our Radiology paper, please use the following citation:

```
@article{SpineAINUHS-NUS2021-stenosis,
title={Deep learning model for automated detection and classification of central canal, lateral recess and neural foraminal stenosis on lumbar spine MRI},
url={ https://doi.org/10.1148/radiol.2021204289},
author={
  ...
}
```

## 💜 Contact

Address correspondence to J.T.P.D.H. (e-mail: james_hallinan AT nuhs.edu.sg)

### _Disclaimer_

_This code base is for research purposes and no warranty is provided. We are not responsible for any medical usage of our code._
