### Import packages

import os
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import pickle
import cv2

print("TF version: ", tf.__version__)
print("cv2 version: ", cv2.__version__)

### Configs for TF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras


### File Constants
# Architecture
arch = "Resnet"  # NAS | Resnet

# Mode
mode = "Axial"  # Sag | Axial

# some images are in .JPG some in .jpg
IMG_EXT = "jpg" # or "jpg" or "png" or "JPG"!

# file path and names required
ROOT_DIR = "/hdd1/kaiyuan/SpineAI_with_Zhulei_Dec2019_obj_det/research/object_detection/"
OBJ_DET_ROOT = "/hdd2/kaiyuan/SpineAI_with_Zhulei_Dec2019_obj_det/research/object_detection/"
# classifier dir
#  CLASSIFIER_ROOT_DIR = "/hdd2/kaiyuan/SpineAI_classifier_postRSNA"
CLASSIFIER_ROOT_DIR = "/hdd2/kaiyuan/SpineAI_with_Zhulei_Dec2019_obj_det/research/object_detection/Resnet_Best_Classifiers_Jun2020/"

if arch == "NAS":
    OBJ_DET_IMGS_DIR = "Testset_Axial_NAS"
    OBJ_DET_PICKLE = os.path.join(
        ROOT_DIR,
        OBJ_DET_IMGS_DIR,
        "mar272020_axial_Testset_NAS_detection.pickle"
    )
elif arch == "Resnet":
    OBJ_DET_IMGS_DIR = "AX_jpgs"
    OBJ_DET_PICKLE = os.path.join(
        OBJ_DET_ROOT,
        OBJ_DET_IMGS_DIR,
        "ax.pickle"
    )

# 9 weights for avg and std
VERSION = "v_1_C"


if mode == "Axial":
    # weights
    best_center_weight = "Axial_center_resnetscale150V1_150x150bat128_6LDropout_Date0616-1302_Ep30_ValAcc0.871_ValLoss10.56.h5"
    best_lateral_weight = "Axial_lateral_resnetscale150V1_150x150bat128_6LDropout_Date0616-2058_Ep28_ValAcc0.746_ValLoss11.68.h5"

    best_center_path = "Axial_Center_BestWeights_NewTop3_Jun2020/"
    best_lateral_path = "Axial_Lateral_BestWeights_NewTop3_Jun2020/"

    CENTER_MODEL_WEIGHT = os.path.join(
        CLASSIFIER_ROOT_DIR,
        best_center_path,
        VERSION,
        best_center_weight
    )
    LATERAL_MODEL_WEIGHT = os.path.join(
        CLASSIFIER_ROOT_DIR,
        best_lateral_path,
        VERSION,
        best_lateral_weight
    )
    print(os.path.exists(CENTER_MODEL_WEIGHT))
    print(CENTER_MODEL_WEIGHT, "exists")
    print(os.path.getsize(CENTER_MODEL_WEIGHT), "byte")

    print(os.path.exists(LATERAL_MODEL_WEIGHT))
    print(LATERAL_MODEL_WEIGHT, "exists")
    print(os.path.getsize(LATERAL_MODEL_WEIGHT), "byte")

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

    lateral_model = model_from_json(model_json)

    lateral_model.load_weights(LATERAL_MODEL_WEIGHT)

    print("Loaded lateral_model from disk")

elif mode == "Sag":
    # weights
    best_sag_weight = "Sag_resnetscale150V3_linearf0003_Date0624-1919_Ep20_ValAcc0.809_ValLoss11.80.h5"
    best_sag_path = "Sag_BestWeights_ResnetScale150_Jun2020_Top3/"
    SAG_MODEL_WEIGHT = os.path.join(
        CLASSIFIER_ROOT_DIR,
        best_sag_path,
        VERSION,
        best_sag_weight
    )
    print(os.path.exists(SAG_MODEL_WEIGHT))
    print(SAG_MODEL_WEIGHT, "exists")
    print(os.path.getsize(SAG_MODEL_WEIGHT), "byte")

    # load model
    TRAINED_JSON = os.path.join(
        CLASSIFIER_ROOT_DIR,
        best_sag_path,
        "6conv-model.json"
    )
    # Instantiate a model from JSON
    json_file = open(TRAINED_JSON, 'r')
    model_json = json_file.read()
    json_file.close()

    sag_model = model_from_json(model_json)

    sag_model.load_weights(SAG_MODEL_WEIGHT) # Sets the state of the model.

    print("Loaded sag_model from json and weights")

print(os.path.exists(OBJ_DET_PICKLE))
print(OBJ_DET_PICKLE, "exists")
print(os.path.getsize(OBJ_DET_PICKLE), "byte")

print(os.path.exists(TRAINED_JSON))
print(TRAINED_JSON, "exists")
print(os.path.getsize(TRAINED_JSON), "byte")


with open(OBJ_DET_PICKLE, 'rb') as f:
    roi_detection = pickle.load(f)


print("Num of ROI detections: ", len(roi_detection.keys()))

print("partial view of roi_detection:")
print(dict(list(roi_detection.items())[0:2]))


"""
# see how many of the images do NOT even
# have a bounding box
without_box = [k for k in sag_detection.keys() if len(sag_detection[k]) == 0]
print(len(without_box))
# verify one example
IMG_EXAMPLE1 = OBJ_DET_IMGS_DIR + "/459_val_158--T1W_Sagittal_00000002--00000019.JPG"
print("IMG_EXAMPLE1 in without_box? :", IMG_EXAMPLE1 in without_box)
"""

# ### Config for model to predict

# In[ ]:


# num of classes is 4!
nb_class = 4
# dimensions of our images.
img_width, img_height = 150, 150

prelabel_folder = os.path.join(OBJ_DET_ROOT, OBJ_DET_IMGS_DIR)

grading = ['normal', 'mild', 'moderate', 'severe']

# scale the crop
def scale_crop(xmin, xmax, ymin, ymax, factor, img_shape):
    """
    the imgs top left corner is (0,0),
    x min-max is from point A,
    y min-max is from point B
    """
    cropped_w = xmax - xmin
    cropped_h = ymax - ymin
    xmin -= ((cropped_w * factor) // 2)
    ymin -= ((cropped_h * factor) // 2)

    xmax += ((cropped_w * factor) // 2)
    ymax += ((cropped_h * factor) // 2)

    # cv2 img shape information
    height = img_shape[0]
    width = img_shape[1]

    return (
            int(max(xmin, 0)),
            int(min(xmax, width)),
            int(max(ymin, 0)),
            int(min(ymax, height))
            )


# ### Start the Crop+Label using model.predict and write to xml

count_img = 0
for file in os.listdir(prelabel_folder):
    print(str(count_img) + " process file: " + file)
    if (not file.endswith('jpg')
            and not file.endswith('png')
            and not file.endswith('JPG')):
        print("***[NOT IMAGE]*** ", file, " is not an image")
        continue
    count_img += 1

    img_orig = keras.preprocessing.image.load_img(os.path.join(prelabel_folder, file))
    img = keras.preprocessing.image.img_to_array(img_orig)

    h, w, _ = img.shape
    print("--> img shape: ", img.shape, " <---")

    xml = [
        "<annotation>\n",
        "\t<folder>"+OBJ_DET_IMGS_DIR+"</folder>\n",
        "\t<filename>" + file + "</filename>\n",
        "\t<path>" + os.path.join(prelabel_folder, file) + "</path>\n",
        "\t<source>\n",
        "\t\t<database>Unknown</database>\n",
        "\t</source>\n",
        "\t<size>\n",
        "\t\t<width>"+str(w)+"</width>\n",
        "\t\t<height>"+str(h)+"</height>\n",
        "\t\t<depth>1</depth>\n",
        "\t</size>\n",
        "\t<segmented>0</segmented>\n",
    ]

    # xxx_detection is the loaded pickle dict
    # make this scale150 version
    for k, v in roi_detection[
        os.path.join(OBJ_DET_IMGS_DIR, file)
    ].items():
        ymin, xmin, ymax, xmax = k
        (xmin, xmax, ymin, ymax) = (
            int(xmin * w),
            int(xmax * w),
            int(ymin * h),
            int(ymax * h),
        )

        print(">>>>before scaling, xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)

        scale_factor = 0.5 # 0.5 for scale by 150%
        cv2_shape = [h, w]
        (xmin_for_pred, xmax_for_pred, ymin_for_pred, ymax_for_pred) = scale_crop(
            xmin, xmax, ymin, ymax, scale_factor, cv2_shape)

        print("<<<<after scaling, xmin, xmax, ymin, ymax: ",
              xmin_for_pred, xmax_for_pred, ymin_for_pred, ymax_for_pred)

        cropped_img = img[ymin_for_pred:ymax_for_pred, xmin_for_pred:xmax_for_pred, :]
        cropped_img = cv2.resize(
            cropped_img,
            (img_width, img_height),
            # change this to inter_linear
            interpolation=cv2.INTER_LINEAR
        )

        x = 1 / 255.0 * cropped_img

        if v == 3:
            print("\nv is 3, flip", v)
            x = cv2.flip(x, 1)

        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])

        prediction = []
        predicted_class = -1
        if mode == "Sag" and v == 1:  # sag only 1 label == 1
            prediction = sag_model.predict(images)
            predicted_class = np.argmax(prediction[0])
        elif mode == "Axial":
            if v == 1 or v == 3:  # lateral
                prediction = lateral_model.predict(images)
                predicted_class = np.argmax(prediction[0])
            elif v == 2:  # center
                prediction = center_model.predict(images)
                predicted_class = np.argmax(prediction[0])
        else:
            continue

        obj = [
            "\t<object>\n",
            "\t\t<name>" + grading[predicted_class] + "</name>\n",
            "\t\t<pose>Unspecified</pose>\n",
            "\t\t<truncated>0</truncated>\n",
            "\t\t<difficult>0</difficult>\n",
            "\t\t<bndbox>\n",
            "\t\t\t<xmin>" + str(xmin) + "</xmin>\n",
            "\t\t\t<ymin>" + str(ymin) + "</ymin>\n",
            "\t\t\t<xmax>" + str(xmax) + "</xmax>\n",
            "\t\t\t<ymax>" + str(ymax) + "</ymax>\n",
            "\t\t</bndbox>\n",
            "\t</object>\n",
        ]

        for e in obj:
            xml.append(e)

    xml.append("</annotation>\n") # close off the xml tag

    with open(
        os.path.join(prelabel_folder, file.replace(IMG_EXT, "xml")),
        "w"
    ) as f:
        for e in xml:
            f.write(e)

# verify the outcome
num_files_combined = len(os.listdir(prelabel_folder))

print("num_files_combined in ", prelabel_folder, " = ", num_files_combined)

files_xml = [f for f in os.listdir(prelabel_folder) if "xml" in f]
print("num of xmls: ", len(files_xml))
