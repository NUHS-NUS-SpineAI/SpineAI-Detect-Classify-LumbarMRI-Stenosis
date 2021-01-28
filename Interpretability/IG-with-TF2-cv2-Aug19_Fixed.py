#!/usr/bin/env python
# coding: utf-8

# ## Load the model and weights

# In[1]:


### Import packages

import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import pickle
import cv2

print("TF version: ", tf.__version__)
print("cv2 version: ", cv2.__version__)

### File Constants
# Architecture
arch = "Resnet"  # NAS | Resnet

# Mode
mode = "Axial"  # Sag | Axial

# some images are in .JPG some in .jpg
IMG_EXT = "JPG" # or "jpg" or "png" or "JPG"!

# classifier dir
#  CLASSIFIER_ROOT_DIR = "/hdd2/kaiyuan/SpineAI_classifier_postRSNA"
CLASSIFIER_ROOT_DIR = "./Resnet_Best_Classifiers_Jun2020/"

# 9 weights for avg and std
VERSION = "v_3_C"


if mode == "Axial":
    # weights
    best_center_weight = "Axial_center_resnetscale150V3_150x150bat128_6LDropout_Date0618-1158_Ep22_ValAcc0.856_ValLoss10.78.h5"

    best_center_path = "Axial_Center_BestWeights_NewTop3_Jun2020/"

    CENTER_MODEL_WEIGHT = os.path.join(
        CLASSIFIER_ROOT_DIR,
        best_center_path,
        VERSION,
        best_center_weight
    )

    print(os.path.exists(CENTER_MODEL_WEIGHT))
    print(CENTER_MODEL_WEIGHT, "exists")
    print(os.path.getsize(CENTER_MODEL_WEIGHT), "byte")

    # load model
    TRAINED_JSON = os.path.join(
        CLASSIFIER_ROOT_DIR,
        best_center_path,
        "6conv-model.json"
    )
    # Instantiate a model from JSON
    json_file = open(TRAINED_JSON, 'r')
    model_json = json_file.read()
    json_file.close()

    center_model = model_from_json(model_json)

    center_model.load_weights(CENTER_MODEL_WEIGHT)

    print("Loaded center_model from disk")


# ## Setup and dependencies

# In[2]:


import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# model summary check
model = center_model
model.summary()

# ## Load image with keras and tf

# In[3]:


img_path = "./1.3.6.1.4.1.5962.99.1.2380920017.862678823.1535684277457.21756.0.jpg"
# dimensions of our images.
img_width, img_height = 150, 150
img_size = (img_width, img_height)

# labels
grading = np.array(['normal', 'mild', 'moderate', 'severe'])


img_url = {
    'TestImg': img_path,
}

"""
# original tf2 read_image()

def read_image(file_name):
  image = tf.io.read_file(file_name)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
  return image
"""

# new read_image with cv2 resize and interpolation
def read_image(file_name):
    """our pipeline"""
    img_orig = keras.preprocessing.image.load_img(file_name)
    print("-keras load_img: ", img_orig)
    img = keras.preprocessing.image.img_to_array(img_orig)
    print("-keras img_to_array: ", img.shape, img.dtype, img[0][0])

    # change from INTER_CUBIC to INTER_LINEAR
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
    print("-cv2 resize: ", img.shape, img.dtype, img[0][0])
    
    x = 1/255.0 * img
    print("-normalize: ", x.shape, x.dtype, x[0][0])

    image = tf.image.convert_image_dtype(x, tf.float32)
    print("-convert_image_dtype: ", image.shape, image.dtype, image[0][0])

    return image

img_paths = {name: url for (name, url) in img_url.items()}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}


# In[4]:


plt.figure(figsize=(8, 8))
for n, (name, img_tensors) in enumerate(img_name_tensors.items()):
  ax = plt.subplot(1, 2, n+1)
  ax.imshow(img_tensors)
  ax.set_title(name)
  ax.axis('off')
plt.tight_layout()


# ## Classify Images

# In[6]:


def top_k_predictions(img, k=3):
#     print(img)
#     x = np.expand_dims(img, axis=0)
#     image_batch = np.vstack([x])
    
    image_batch = tf.expand_dims(img, 0)
    print("image_batch shape: ", image_batch.shape)
    predictions = model(image_batch)
    print("predictions: ", predictions)
    probs = predictions
    print("probs: ", probs)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    # not using imagenet_labels
    top_labels = grading[tuple(top_idxs)]
    print("tuple(top_idxs): ", tuple(top_idxs))
    print("top_labels: ", top_labels)
    return top_labels, top_probs[0]

for (name, img_tensor) in img_name_tensors.items():
  plt.imshow(img_tensor)
  plt.title(name, fontweight='bold')
  plt.axis('off')
  plt.show()

  pred_label, pred_prob = top_k_predictions(img_tensor)
  for label, prob in zip(pred_label, pred_prob):
    print(f'{label}: {prob:0.1%}')


# ## Using IG

# ### Baseline

# In[7]:


baseline = tf.zeros(shape=(150,150,3))

plt.imshow(baseline)
plt.title("Baseline")
plt.axis('off')
plt.show()

m_steps=50
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.
def interpolate_images(baseline,
                       image,
                       alphas):
  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(image, axis=0)
  delta = input_x - baseline_x
  images = baseline_x +  alphas_x * delta
  return images

interpolated_images = interpolate_images(
    baseline=baseline,
    image=img_name_tensors['TestImg'],
    alphas=alphas)

fig = plt.figure(figsize=(20, 20))

i = 0
for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
  i += 1
  plt.subplot(1, len(alphas[0::10]), i)
  plt.title(f'alpha: {alpha:.1f}')
  plt.imshow(image)
  plt.axis('off')

plt.tight_layout();


# ### Compute Gradients

# In[8]:


target_class_idx = 2 # 2 is moderate for the TestImg


# In[9]:


def compute_gradients(images, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(images)
    logits = model(images)
#     probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
  return tape.gradient(logits, images)

path_gradients = compute_gradients(
    images=interpolated_images,
    target_class_idx=target_class_idx)

# m_steps = 50, so the path_gradients.shape should be (50+1,..)
print("path_gradients.shape", path_gradients.shape)

# Visualize the gradient saturation

pred = model(interpolated_images)
pred_proba = pred[:, target_class_idx]

plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba)
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])

ax2 = plt.subplot(1, 2, 2)
# Average across interpolation steps
average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
# Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average pixel gradients (normalized) over alpha')
ax2.set_ylabel('Average pixel gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1]);


# ### Accumulate gradients (integral approximation)

# In[10]:


def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients

ig = integral_approximation(
    gradients=path_gradients)

print("shape of IG: ", ig.shape)


@tf.function
def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):
  # 1. Generate alphas.
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

  # Initialize TensorArray outside loop to collect gradients.    
  gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)
    
  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    # 2. Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

    # 3. Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx)
    
    # Write batch indices and gradients to extend TensorArray.
    gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    
  
  # Stack path gradients together row-wise into single tensor.
  total_gradients = gradient_batches.stack()

  # 4. Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients)

  # 5. Scale integrated gradients with respect to input.
  integrated_gradients = (image - baseline) * avg_gradients

  return integrated_gradients

ig_attributions = integrated_gradients(baseline=baseline,
                                       image=img_name_tensors['TestImg'],
                                       target_class_idx=target_class_idx,
                                       m_steps=240)

print("IG feature attribution shape: ", ig_attributions.shape)


# ## Visualize Attributions

# In[11]:


def plot_img_attributions(baseline,
                          image,
                          target_class_idx,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4):

  attributions = integrated_gradients(baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      m_steps=m_steps)

  # Sum of the attributions across color channels for visualization.
  # The attribution mask shape is a grayscale image with height and width
  # equal to the original image.
  attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

  fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

  axs[0, 0].set_title('Baseline image')
  axs[0, 0].imshow(baseline)
  axs[0, 0].axis('off')

  axs[0, 1].set_title('Original image')
  axs[0, 1].imshow(image)
  axs[0, 1].axis('off')

  axs[1, 0].set_title('Attribution mask')
  axs[1, 0].imshow(attribution_mask, cmap=cmap)
  axs[1, 0].axis('off')

  axs[1, 1].set_title('Overlay')
  axs[1, 1].imshow(attribution_mask, cmap=cmap)
  axs[1, 1].imshow(image, alpha=overlay_alpha)
  axs[1, 1].axis('off')

  plt.tight_layout()
  plt.savefig("IG-TF2-sample.jpg")
  return fig

_ = plot_img_attributions(image=img_name_tensors['TestImg'],
                          baseline=baseline,
                          target_class_idx=target_class_idx,
                          m_steps=240,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.4)

