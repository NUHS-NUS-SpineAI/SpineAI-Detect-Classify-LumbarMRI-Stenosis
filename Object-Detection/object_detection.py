#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[ ]:


import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
import pickle

# for download url and extract zip
# import six.moves.urllib as urllib
# import tarfile
# import zipfile

# legacy utils
# from collections import defaultdict
# from io import StringIO

from matplotlib import pyplot as plt
from PIL import Image

# zhulei custom config patch
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

print(tf.__version__)


# ## Env setup

# In[ ]:


myhost = os.uname()[1]
print(">>>>    Hostname: ", myhost)
print("\nCWD: ", os.getcwd())


# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:


# Folder name containing the Trained Obj-det model/graph
MODEL_NAME = 'Axial_1-491_Resnet_Jun142020_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'ak_detection.pbtxt')


# 3 for axial, 1 for sagittal
NUM_CLASSES = 3

# save previews of the overlay images
save_previews = False

if save_previews:
    script_dir = os.getcwd()
    results_dir = os.path.join(script_dir, 'Sag_Resnet_1-491-preview_obj-det-results-Jun222020/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


# zhulei updated with compat.v1 and tf.io
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# zhulei custom modify
#category_index = {1: {'id': 1, 'name': 'left'}, 2: {'id': 2, 'name': 'center'}, 3: {'id': 3, 'name': 'right'}}
print("\ncategory_index: ", category_index)


# ## Helper code

# In[ ]:


# zhulei modify to detect image.mode
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    # 'L' for Grayscale, 'RGB' : for 3 channel images
    channel_dict = {'L':1, 'RGB':3}
    return np.array(image.getdata()).reshape(
        (im_height, im_width, channel_dict[image.mode])).astype(np.uint8)


# # Detection

# In[ ]:


print("current dir: ", os.getcwd())

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'images/test'


generated_pickle = './{}/obj-det.pickle'.format(
    PATH_TO_TEST_IMAGES_DIR)

# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
TEST_IMAGE_PATHS = []
for file in os.listdir(PATH_TO_TEST_IMAGES_DIR):
    if '.jpg' in file or '.png' in file or '.JPG' in file:
        TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, file))

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

print("\nTest images: ", TEST_IMAGE_PATHS[:5])

print("\nNum test images: ", len(TEST_IMAGE_PATHS))


# In[ ]:


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.compat.v1.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.compat.v1.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# In[ ]:

detection_result = {}

count_img = 0

for image_path in TEST_IMAGE_PATHS:
    # zhulei add counting img
    print('process: ', str(count_img), image_path)
    count_img += 1

    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    # zhulei check for image_np
    if image_np.shape[2] != 3:
        # Duplicating the Content
        image_np = np.broadcast_to(image_np, (image_np.shape[0], image_np.shape[1], 3)).copy()
        ## adding Zeros to other Channels
        ## This adds Red Color stuff in background -- not recommended
        # z = np.zeros(image_np.shape[:-1] + (2,), dtype=image_np.dtype)
        # image_np = np.concatenate((image_np, z), axis=-1)
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=2)  # default 8

    # zhulei custom
    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']
    max_boxes_to_draw = 20
    min_score_thresh = 0.5
    box_to_class = {}

    # zhulei custom noting down the box_to_classes
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            box_to_class[box] = classes[i]

    # for export to pickle later
    detection_result[image_path] = box_to_class

    if save_previews:
        plt.figure(figsize=IMAGE_SIZE)
        # need imshow to save the imgs
        plt.imshow(image_np)
        plt.savefig('{}/{}'.format(
            results_dir, os.path.basename(image_path))
        )

# In[ ]:

# export detection_result dict with pickle
with open(generated_pickle, 'wb') as f:
    pickle.dump(detection_result, f, protocol=pickle.HIGHEST_PROTOCOL)


# verify the generated pickle is saved
# generated_pickle = './{}/{}_detection.pickle'.format(PATH_TO_TEST_IMAGES_DIR, labeler)
print(os.path.exists(generated_pickle))
print(generated_pickle, "exists")
print(os.path.getsize(generated_pickle), "byte")
