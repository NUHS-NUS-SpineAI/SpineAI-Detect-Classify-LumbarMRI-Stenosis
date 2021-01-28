import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import cv2


# import visualization_utils as vis_util
from utils import visualization_utils as vis_util


convert_to_number = {'normal':1, 'mild': 2, 'moderate':3, 'severe':4, '1':1, '2':2, '3':3, '4':4}


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


# dimensions of our images.
# img_width, img_height = 84, 84
img_width, img_height = 150, 150


# load detected bounding box for axial and sagittal images
# for Resnet
with open('./Axial_Testset_Resnet_detection.pickle', 'rb') as f:
    axial_detection = pickle.load(f)
    # for pickled dict iteration
    axial_detection_item_predix = "Testset_Axial"

with open('./Sag_Testset_Resnet_detection.pickle', 'rb') as f:
    sag_detection = pickle.load(f)
    # for pickled dict iteration
    sag_detection_item_predix = "Testset_Sag"

print("Num of axial detections: ", len(axial_detection.keys()))

print("partial view of axial_detection:")
print(dict(list(axial_detection.items())[0:2]))

print("Num of sag detections: ", len(sag_detection.keys()))

print("partial view of sag_detection:")
print(dict(list(sag_detection.items())[0:2]))


# load classification models
CLASSIFIER_ROOT_DIR = "./Resnet_Best_Classifiers_Jun2020/"

AX_VERSION = "v_1_C"
SAG_VERSION = "v_3_A"

# weights
if AX_VERSION == "v_1_A":
    best_center_weight = "Axial_center_resnetscale150V1_150x150bat128_6LDropout_Date0615-2011_Ep25_ValAcc0.844_ValLoss10.41.h5"
    best_lateral_weight = "Axial_lateral_resnetscale150V1_150x150bat128_6LDropout_Date0616-1846_Ep26_ValAcc0.806_ValLoss11.16.h5"

    best_center_path = "Axial_Center_BestWeights_NewTop3_Jun2020/"
    best_lateral_path = "Axial_Lateral_BestWeights_NewTop3_Jun2020/"
elif AX_VERSION == "v_1_C":
    best_center_weight = "Axial_center_resnetscale150V1_150x150bat128_6LDropout_Date0616-1302_Ep30_ValAcc0.871_ValLoss10.56.h5"
    best_lateral_weight = "Axial_lateral_resnetscale150V1_150x150bat128_6LDropout_Date0616-2058_Ep28_ValAcc0.746_ValLoss11.68.h5"

    best_center_path = "Axial_Center_BestWeights_NewTop3_Jun2020/"
    best_lateral_path = "Axial_Lateral_BestWeights_NewTop3_Jun2020/"

CENTER_MODEL_WEIGHT = os.path.join(
    CLASSIFIER_ROOT_DIR,
    best_center_path,
    AX_VERSION,
    best_center_weight
)
LATERAL_MODEL_WEIGHT = os.path.join(
    CLASSIFIER_ROOT_DIR,
    best_lateral_path,
    AX_VERSION,
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

# weights
best_sag_weight = "Sag_resnetscale150V3_linearf0003_Date0624-1917_Ep50_ValAcc0.802_ValLoss13.19.h5"
best_sag_path = "Sag_BestWeights_ResnetScale150_Jun2020_Top3/"
SAG_MODEL_WEIGHT = os.path.join(
    CLASSIFIER_ROOT_DIR,
    best_sag_path,
    SAG_VERSION,
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


nb_class = 4

category_index = {1: {'id': 1, 'name': 'normal'}, 2: {'id': 2, 'name': 'mild'}, 3: {'id': 3, 'name': 'moderate'}, 4: {'id': 4, 'name': 'severe'}}


cwd = './result-analysis-package-jan212021version/spine-test'
for folder in os.listdir(cwd):
    print("process folder " + folder)
    if folder == 'axial-test' or folder == 'sag-test':
        pass
    else:
        continue

    for file in os.listdir(os.path.join(cwd, folder)):
        if '.xml' in file:
            print('process file ' + file)

            img_orig = image.load_img(os.path.join(cwd, folder, file.replace('xml', 'JPG')))
            img = image.img_to_array(img_orig)

            h, w, _ = img.shape

            data = ET.parse(os.path.join(cwd, folder, file))
            root = data.getroot()

            if 'axial' in folder:

                visualize_test_box = []
                visualize_test_classes = []
                for o in root.findall('object'):
                    name = o.find('name').text
                    xmin = int(o.find('bndbox').find('xmin').text)
                    xmax = int(o.find('bndbox').find('xmax').text)
                    ymin = int(o.find('bndbox').find('ymin').text)
                    ymax = int(o.find('bndbox').find('ymax').text)

                    visualize_test_box.append([ymin, xmin, ymax, xmax])
                    visualize_test_classes.append(convert_to_number[name])

                test_img = np.copy(img_orig)

                visualize_test_box = np.array(visualize_test_box)
                visualize_test_classes = np.array(visualize_test_classes, dtype=np.uint8)

                print(visualize_test_classes.shape[0])
                print(visualize_test_box.shape[0])

                vis_util.visualize_boxes_and_labels_on_image_array(
                    test_img,
                    visualize_test_box,
                    visualize_test_classes,
                    np.ones(visualize_test_classes.shape[0], dtype=np.uint8),
                    category_index,
                    use_normalized_coordinates=False,
                    max_boxes_to_draw=20,
                    min_score_thresh=.0,
                    line_thickness=4)


                cv2.imwrite(os.path.join(cwd, 'axial-test-with-prob', file.replace('xml', 'JPG')), test_img)


                nb_prediction = len(axial_detection[axial_detection_item_predix+'/'+file.replace('xml','JPG')].keys())
                visualize_predict_box = np.zeros((nb_prediction, 4))
                visualize_predict_classes = np.zeros(nb_prediction, dtype=np.uint8)
                visualize_predict_scores = np.zeros(nb_prediction)

                count_prediction = 0
                for k,v in axial_detection[axial_detection_item_predix+'/'+file.replace('xml','JPG')].items():

                    ymin, xmin, ymax, xmax = k
                    (xmin, xmax, ymin, ymax) = (int(xmin*w), int(xmax*w), int(ymin*h), int(ymax*h))
                    visualize_predict_box[count_prediction] = np.array([ymin, xmin, ymax, xmax])

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
                    if v == 1 or v == 3:  # lateral
                        prediction = lateral_model.predict(images)
                        predicted_class = np.argmax(prediction[0])

                    elif v == 2: # center
                        prediction = center_model.predict(images)
                        predicted_class = np.argmax(prediction[0])

                    else:
                        count_prediction += 1
                        continue

                    visualize_predict_classes[count_prediction] = predicted_class+1
                    visualize_predict_scores[count_prediction] = prediction[0][predicted_class]

                    count_prediction += 1

                predict_img = np.copy(img_orig)
                print("\n##### axial-predict #####\n")
                vis_util.visualize_boxes_and_labels_on_image_array(
                    predict_img,
                    visualize_predict_box,
                    visualize_predict_classes,
                    visualize_predict_scores,
                    category_index,
                    use_normalized_coordinates=False,
                    max_boxes_to_draw=20,
                    min_score_thresh=.0,
                    line_thickness=1)

                cv2.imwrite(os.path.join(cwd, 'axial-predict', file.replace('xml', 'JPG')), predict_img)


            elif 'sag' in folder:

                boxes = []
                gradings = []
                for o in root.findall('object'):
                    gradings.append(convert_to_number[o.find('name').text])
                    xmin = int(o.find('bndbox').find('xmin').text)
                    xmax = int(o.find('bndbox').find('xmax').text)
                    ymin = int(o.find('bndbox').find('ymin').text)
                    ymax = int(o.find('bndbox').find('ymax').text)

                    boxes.append((xmin, xmax, ymin, ymax))

                # boxes = sorted(boxes, key=lambda x: x[2])

                visualize_test_box = np.zeros((len(boxes), 4))
                visualize_test_classes = np.zeros((len(boxes)), dtype=np.uint8)

                labels = []

                for i in range(0, len(boxes)):
                    visualize_test_box[i] = np.array([boxes[i][2], boxes[i][0], boxes[i][3], boxes[i][1]])
                    # visualize_test_classes[i] = int(sag_label[file][i])
                    # labels.append([boxes[i], int(sag_label[file][i])-1])

                    visualize_test_classes[i] = gradings[i]
                    labels.append([boxes[i], gradings[i]-1])

                test_img = np.copy(img_orig)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    test_img,
                    visualize_test_box,
                    visualize_test_classes,
                    np.ones(visualize_test_classes.shape[0], dtype=np.uint8),
                    category_index,
                    use_normalized_coordinates=False,
                    max_boxes_to_draw=20,
                    line_thickness=4)


                cv2.imwrite(os.path.join(cwd, 'sag-test-with-prob', file.replace('xml', 'JPG')), test_img)


                nb_prediction = len(sag_detection[sag_detection_item_predix+'/'+file.replace('xml','JPG')].keys())
                visualize_predict_box = np.zeros((nb_prediction, 4))
                visualize_predict_classes = np.zeros(nb_prediction, dtype=np.uint8)
                visualize_predict_scores = np.zeros(nb_prediction)

                count_prediction = 0
                for k,v in sag_detection[sag_detection_item_predix+'/'+file.replace('xml','JPG')].items():
                    ymin, xmin, ymax, xmax = k
                    (xmin, xmax, ymin, ymax) = (int(xmin*w), int(xmax*w), int(ymin*h), int(ymax*h))
                    box = (xmin, xmax, ymin, ymax)

                    visualize_predict_box[count_prediction] = np.array([ymin, xmin, ymax, xmax])

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
                    x = np.expand_dims(x, axis=0)

                    images = np.vstack([x])

                    prediction = sag_model.predict(images)
                    predicted_class = np.argmax(prediction[0])

                    visualize_predict_classes[count_prediction] = predicted_class+1
                    visualize_predict_scores[count_prediction] = prediction[0][predicted_class]

                    count_prediction += 1


                predict_img = np.copy(img_orig)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    predict_img,
                    visualize_predict_box,
                    visualize_predict_classes,
                    visualize_predict_scores,
                    category_index,
                    use_normalized_coordinates=False,
                    max_boxes_to_draw=20,
                    min_score_thresh=.0,
                    line_thickness=1)

                cv2.imwrite(os.path.join(cwd, 'sag-predict', file.replace('xml', 'JPG')), predict_img)

            else:
                continue

