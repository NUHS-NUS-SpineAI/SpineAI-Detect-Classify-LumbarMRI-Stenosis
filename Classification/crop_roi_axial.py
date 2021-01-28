import os
import xml.etree.ElementTree as ET
import cv2

cwd = './Axial_1-491_10751x2jpgXmls_ResnetRecrop_Jun152020'
center_dir = 'center-crop'
left_dir = 'left-crop'
right_dir = 'right-crop'

out_dirs = [center_dir, left_dir, right_dir]
grading = ['1', '2', '3', '4']

center_label_file = 'Axial_1-491_10751x2_Jun142020-axial-center-label.txt'
right_label_file = 'Axial_1-491_10751x2_Jun142020-axial-right-label.txt'
left_label_file = 'Axial_1-491_10751x2_Jun142020-axial-left-label.txt'

# 150% scale, use scale_factor = 0.5
scale_factor = 0.5

for e in out_dirs:
    if not os.path.exists(e):
        os.makedirs(e)

    for ee in grading:
        if not os.path.exists(os.path.join(e, ee)):
            os.makedirs(os.path.join(e, ee))

center_label = {}
right_label = {}
left_label = {}

with open(center_label_file, 'r') as f:
    lines = f.readlines()
    for e in lines:
        tmp = e.strip().split(' ')
        center_label[tmp[0]] = tmp[1]

with open(right_label_file, 'r') as f:
    lines = f.readlines()
    for e in lines:
        tmp = e.strip().split(' ')
        right_label[tmp[0]] = tmp[1]

with open(left_label_file, 'r') as f:
    lines = f.readlines()
    for e in lines:
        tmp = e.strip().split(' ')
        left_label[tmp[0]] = tmp[1]

# print(center_label)
# print(right_label)
# print(left_label)

# exit(0)


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


for file in os.listdir(cwd):
    if '.xml' in file:
        print(file)
        img = cv2.imread(os.path.join(cwd, file.replace('xml', 'JPG')), 0)
        print(img.shape)

        # print(np.array_equal(img[:,:,0], img[:,:,1]))
        # print(np.array_equal(img[:,:,0], img[:,:,2]))
        # print(np.array_equal(img[:,:,1], img[:,:,2]))

        # for k in range(0, 3):
        #   for i in range(0, 10):
        #       tmp = ''
        #       for j in range(0, 10):
        #           tmp += str(img[i,j,k]) + ' '
        #       print(tmp + '\n')

        #   print('\n\n')

        data = ET.parse(os.path.join(cwd, file))
        root = data.getroot()
        for o in root.findall('object'):
            name = o.find('name').text
            xmin = int(o.find('bndbox').find('xmin').text)
            xmax = int(o.find('bndbox').find('xmax').text)
            ymin = int(o.find('bndbox').find('ymin').text)
            ymax = int(o.find('bndbox').find('ymax').text)

            print(">>>>before scaling, xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)

            (xmin, xmax, ymin, ymax) = scale_crop(xmin, xmax, ymin, ymax, scale_factor, img.shape)

            print("<<<<after scaling, xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)

            cropped_img = img[ymin:ymax, xmin:xmax]

            # cv2.imshow('image', cropped_img)
            # cv2.waitKey(0)
            # exit(0)

            # print(file)
            # exit(0)

            if name == 'center':
                tmp = center_label[file]
                if not cv2.imwrite(os.path.join(center_dir, tmp, file.replace('xml', 'png')), cropped_img):
                    print(os.path.join(center_dir, tmp, file.replace('xml', 'jpg')))
                    raise Exception("Could not write image")
            elif name == 'right':
                tmp = right_label[file]
                if not cv2.imwrite(os.path.join(right_dir, tmp, file.replace('xml', 'png')), cropped_img):
                    raise Exception("Could not write image")
            else:
                tmp = left_label[file]
                if not cv2.imwrite(os.path.join(left_dir, tmp, file.replace('xml', 'png')), cropped_img):
                    raise Exception("Could not write image")
