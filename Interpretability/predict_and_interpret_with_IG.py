import os
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing import image
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pickle
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from heatmap_IG_utils import main_ig


print("TF version: ", tf.__version__)
print("cv2 version: ", cv2.__version__)

### File Constants
# Architecture
arch = "Resnet"  # NAS | Resnet

# Mode
mode = "Axial"  # Sag | Axial

OBJ_DET_ROOT = "./"

# classifier dir
#  CLASSIFIER_ROOT_DIR = "/hdd2/kaiyuan/SpineAI_classifier_postRSNA"
CLASSIFIER_ROOT_DIR = os.path.join(OBJ_DET_ROOT, "Resnet_Best_Classifiers_Jun2020/")

if arch == "Resnet":
    if mode == "Axial":
        SAVE_DIR = "Aug24_IG_SAset_Axial"  # no "/" at the end!
        VERSION = "v_1_C"
        OBJ_DET_IMGS_DIR = "SA_validation_set"
        OBJ_DET_PICKLE = os.path.join(
            OBJ_DET_ROOT,
            OBJ_DET_IMGS_DIR,
            "detect.pickle"
        )
    elif mode == "Sag":
        SAVE_DIR = "Aug24_IG_SAset_Sag"  # no "/" at the end!
        VERSION = "v_2_B"
        OBJ_DET_IMGS_DIR = "SA_validation_set_sagittal"
        OBJ_DET_PICKLE = os.path.join(
            OBJ_DET_ROOT,
            OBJ_DET_IMGS_DIR,
            "detect_sag.pickle"
        )


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
    best_sag_weight = "Sag_resnetscale150V2_linearf0003_Date0624-1814_Ep36_ValAcc0.779_ValLoss12.13.h5"
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


# num of classes is 4!
nb_class = 4
# dimensions of our images.
img_width, img_height = 150, 150
img_size = (img_width, img_height)

prelabel_folder = os.path.join(OBJ_DET_ROOT, OBJ_DET_IMGS_DIR)

# for IG, the grading needs to be numpy array
grading = np.array(['normal', 'mild', 'moderate', 'severe'])

# prediction stats for verification
center_matrix = np.zeros(nb_class)
lateral_matrix = np.zeros(nb_class)
sag_matrix = np.zeros(nb_class)


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
    print("\n" + str(count_img) + " process file: " + file)
    if (not file.endswith('jpg')
            and not file.endswith('png')
            and not file.endswith('JPG')):
        print("***[NOT IMAGE]*** ", file, " is not an image")
        continue
    count_img += 1

    # get image file path
    img_path = os.path.join(prelabel_folder, file)
    # print("img_path: ", img_path)

    img_orig = keras.preprocessing.image.load_img(img_path)
    # print("-keras load_img: ", img_orig)
    # preview the original image for debug
    # plt.imshow(img_orig)
    # plt.show()
    # load img as array
    img = keras.preprocessing.image.img_to_array(img_orig)
    # print("-keras img_to_array: ", img.shape, img.dtype, img[0][0])

    h, w, _ = img.shape
    # print("--> img shape: ", img.shape, " <---")

    # xxx_detection is the loaded pickle dict
    # make this scale150 version

    # for Sag sort the detection by ymin first
    detection_items = roi_detection[
        os.path.join(OBJ_DET_IMGS_DIR, file)
    ].items()

    if mode == "Sag":
        # NOTE: ymin in k is the 0th item
        # sort by ymin
        detection_items = sorted(
            detection_items,
            key=lambda item: item[0][0]
        )

    for (count, (k, v)) in enumerate(detection_items):
        ymin, xmin, ymax, xmax = k
        (xmin, xmax, ymin, ymax) = (
            int(xmin * w),
            int(xmax * w),
            int(ymin * h),
            int(ymax * h),
        )

        # print(">>>>before scaling, xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)

        scale_factor = 0.5 # 0.5 for scale by 150%
        cv2_shape = [h, w]
        (xmin_for_pred, xmax_for_pred, ymin_for_pred, ymax_for_pred) = scale_crop(
            xmin, xmax, ymin, ymax, scale_factor, cv2_shape)

        # print("<<<<after scaling, xmin, xmax, ymin, ymax: ",
            #   xmin_for_pred, xmax_for_pred, ymin_for_pred, ymax_for_pred)

        cropped_img = img[ymin_for_pred:ymax_for_pred, xmin_for_pred:xmax_for_pred, :]
        cropped_img = cv2.resize(
            cropped_img,
            (img_width, img_height),
            # change this to inter_linear
            interpolation=cv2.INTER_LINEAR
        )
        # print("-cv2 resize: ", cropped_img.shape, cropped_img.dtype, cropped_img[0][0])

        x = 1 / 255.0 * cropped_img
        # print("-normalize: ", x.shape, x.dtype, x[0][0])

        if v == 3:
            # print("\nv is 3, flip", v)
            x = cv2.flip(x, 1)

        # digress to a Tensor object for IG workflow
        img_tensor = tf.image.convert_image_dtype(x, tf.float32)
        # print("-convert_image_dtype: ", img_tensor.shape, img_tensor.dtype, img_tensor[0][0])

        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        # print("images (batch) shape: ", images.shape)

        prediction = []
        predicted_class = -1
        meta = {}
        meta["file_name"] = file
        meta["v"] = v
        meta["mode"] = mode
        # keep track of the sag ROI from top to down
        meta["position_index"] = (count + 1)
        # save dir for jpegs, no "/" at the end!
        meta["save_dir"] = os.path.join(OBJ_DET_ROOT, SAVE_DIR)
        if mode == "Sag" and v == 1:  # sag only 1 label == 1
            prediction = sag_model.predict(images)
            predicted_class = np.argmax(prediction[0])
            sag_matrix[predicted_class] += 1
            # call the IG main function
            main_ig(
                sag_model,
                img_tensor,
                predicted_class,
                prediction,
                meta)
        elif mode == "Axial":
            if v == 1 or v == 3:  # lateral
                prediction = lateral_model.predict(images)
                predicted_class = np.argmax(prediction[0])
                lateral_matrix[predicted_class] += 1
                # call the IG main function
                main_ig(
                    lateral_model,
                    img_tensor,
                    predicted_class,
                    prediction,
                    meta)
            elif v == 2:  # center
                prediction = center_model.predict(images)
                predicted_class = np.argmax(prediction[0])
                center_matrix[predicted_class] += 1
                # call the IG main function
                main_ig(
                    center_model,
                    img_tensor,
                    predicted_class,
                    prediction,
                    meta)
        else:
            continue

        print(f"\n\t>>>> Mode: {mode}, v: {v}")
        print("\tprediction: ", prediction)
        print("\tpredicted_class: ", predicted_class, grading[predicted_class])


# verify the outcome
print("\n\n === IG generation finished ====")
print("center_matrix: ", center_matrix)
print("lateral_matrix: ", lateral_matrix)
print("sag_matrix: ", sag_matrix)
