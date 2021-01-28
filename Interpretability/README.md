# Integrated Gradients to Interpret Classification
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

## Installation and Set up for Heatmap Experiments

```sh
python3 -m venv ./venv
source ./venv/bin/activate

pip install --upgrade pip

# this installs tensorflow-2.3.0-cp36
pip install --upgrade tensorflow

pip install jupyterlab
pip install matplotlib
pip install opencv-python

# try out tf-explain
pip install tf-explain
```


## References

- Tensorflow Core Tutorial on IG https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
- Detailed version of the TF tutorial https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/blogs/integrated_gradients
- Original Github implementation of the IG method https://github.com/ankurtaly/Integrated-Gradients

