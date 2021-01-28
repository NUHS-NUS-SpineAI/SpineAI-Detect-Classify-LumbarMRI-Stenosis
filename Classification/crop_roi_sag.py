import os
import xml.etree.ElementTree as ET
import cv2
# import numpy as np


data_dir = "Sag_1-491_5380x2jpgXmls_ResnetRecrop_Jun223020/"
sag_dir = data_dir[:-1] + "-scaledby150-cropped-version3/"

scale_factor = 0.5

grading = ['1', '2', '3', '4']

grading_LUT = {
            'normal': '1',
            'mild': '2',
            'moderate': '3',
            'severe': '4',
        }

sag_label_file = data_dir[:-1] + "-Jun242020-sagittal-label.txt"

sag_label = {}

with open(sag_label_file, 'r') as f:
    lines = f.readlines()
    for e in lines:
        tmp = e.strip().split(' ')
        tmp2 = []
        for i in range(1, len(tmp)):
            tmp2.append(tmp[i])

        sag_label[tmp[0]] = tmp2

print("sag_label: ", sag_label)

if not os.path.exists(sag_dir):
    os.makedirs(sag_dir)

for e in grading:
    if not os.path.exists(os.path.join(sag_dir, e)):
        os.makedirs(os.path.join(sag_dir, e))

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

for file in os.listdir(data_dir):
    if '.xml' in file:
        print("\nXML file: ", file)
        # Load an color image in grayscale, flag=0
        img = cv2.imread(
            os.path.join(data_dir, file.replace('xml', 'jpg')),
            0)
        # NOTE: some imgs are with ext of JPG uppercase
        # JPG == jpg == jpeg == JPEG in storage
        if img is None:
            img = cv2.imread(
                os.path.join(
                    data_dir,
                    file.replace('xml', 'JPG')
                ), 0)
        print("img.shape: ", img.shape)

        data = ET.parse(os.path.join(data_dir, file))
        root = data.getroot()

        crops = []
        for o in root.findall('object'):
            xmin = int(o.find('bndbox').find('xmin').text)
            xmax = int(o.find('bndbox').find('xmax').text)
            ymin = int(o.find('bndbox').find('ymin').text)
            ymax = int(o.find('bndbox').find('ymax').text)

            print(">>>>before scaling, xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)

            (xmin, xmax, ymin, ymax) = scale_crop(xmin, xmax, ymin, ymax, scale_factor, img.shape)

            print("<<<<after scaling, xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)

            cropped_img = img[ymin:ymax, xmin:xmax]
            crops.append((ymin, cropped_img))

        crops = sorted(crops, key=lambda x: x[0])

        for e in crops:
            print("=======crops item:=======")
            print(e[0])
            #  print("cropped img: ", e[1])
            print("img.shape: ", e[1].shape)

        original_labels = sag_label.get(file, '')
        print("original_labels: ", original_labels)
        labels = [grading_LUT.get(i, i) for i in original_labels]
        print("labels: ", labels)

        file_index = 0
        for i in range(0, len(labels)):
            print('labels[i]: ', labels[i])
            #  print('crops[i][1]: ', crops[i][1])
            file_name = os.path.join(
                    sag_dir,
                    labels[i],
                    file[:-3])
            unique_file_name = file_name + \
                    'file-' + str(file_index) + \
                    '-label-' + str(labels[i]) + \
                    ".png"
            print("file_name: ", file_name)
            print("unique_file_name: ", unique_file_name)
            # write to destination folder with the filename:
            # <original_filename>.file-<index>-label-<label>.png
            if not cv2.imwrite(unique_file_name, crops[i][1]):
                raise Exception("Could not write image")
            file_index += 1
