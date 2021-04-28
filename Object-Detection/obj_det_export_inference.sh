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

echo ">>>>    "run the python script for train obj-det model

# From tensorflow/models/research/object_detection

# ingest 4-D image tensors
INPUT_TYPE="image_tensor"
# path to pipeline config file
PIPELINE_CONFIG_PATH="faster_rcnn_resnet101_coco_baseON_2018_07_13_sag.config"
# path to model.ckpt
TRAINED_CKPT_PREFIX="training/model.ckpt-44712"
# path to folder that will be used for export
EXPORT_DIR="Sag_1-491_Resnet_Jun222020_graph"

python3 export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
