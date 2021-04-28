#!/usr/bin/env bash

echo ">>>>    "on machine: $(hostnamectl)

# === for models to work ===
# echo ">>>>    "activate protoc and export PYTHONPATH

cd ./research/

pwd

# From tensorflow/models/research/
# protoc object_detection/protos/*.proto --python_out=.
# From tensorflow/models/research/
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo ">>>>    "$PYTHONPATH

# then cd to obj-det folder
echo ">>>>    "cd to obj-det folder

cd ./object_detection/

pwd

echo first clean up training dir

rm training/pipeline.config
rm training/checkpoint
rm training/model.ckpt*
rm training/graph.pbtxt
rm training/events.out.*

echo ">>>>    "run the python script for train obj-det model

# Resnet
PRE_TRAINED_MODEL_CONFIG="faster_rcnn_resnet101_coco_baseON_2018_07_13_sag.config"

python3 legacy/train.py \
  --logtostderr \
  --train_dir=training/ \
  --pipeline_config_path=$PRE_TRAINED_MODEL_CONFIG
