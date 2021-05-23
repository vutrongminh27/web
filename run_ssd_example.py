from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import cv2
import sys
import numpy as np

label_path = './models/labels.txt'
image_path = ''
class_names = [name.strip() for name in open(label_path).readlines()]
model_path = './models/mb1-ssd-Epoch-7-Loss-0.7908091119357518.pth'

net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)


def import_image(image_path):
    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        box1 = int(box[0])
        box2 = int(box[1])
        box3 = int(box[2])
        box4 = int(box[3])
        cor = []
        cv2.rectangle(orig_image, (box1, box2), (box3, box4), (255, 255, 0), 1)
        # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(orig_image, label,
                    (box1 - 20, box2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)
    return orig_image

# crop = orig_image[box2:box4, box1:box3]  # [box1:box3, box2:box4]
# imgSize = np.shape(crop)
# gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# blockSize = int(1 / 8 * imgSize[0] / 2 * 2 + 1)
# if blockSize <= 1:
#     blockSize = int(imgSize[0] / 2 * 2 + 1)
# const = 10
# thresh = cv2.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
#                                thresholdType=cv2.THRESH_BINARY_INV, blockSize=blockSize, C=const)
#
#
# def sort_contours(cnts, method="left-to-right"):
#     # initialize the reverse flag and sort index
#     reverse = False
#     i = 0
#     # handle if we need to sort in reverse
#     if method == "right-to-left" or method == "bottom-to-top":
#         reverse = True
#     # handle if we are sorting against the y-coordinate rather than
#     # the x-coordinate of the bounding box
#     if method == "top-to-bottom" or method == "bottom-to-top":
#         i = 1
#     # construct the list of bounding boxes and sort them from top to
#     # bottom
#     boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#     (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
#                                         key=lambda b: b[1][i], reverse=reverse))
#     # return the list of sorted contours and bounding boxes
#     return (cnts, boundingBoxes)
#
#
# contoursX, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # contoursX, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# (contoursX, boundingBoxes) = sort_contours(contoursX, method="left-to-right")
# for contour in contoursX:
#     print(contour)
#     [x, y, w, h] = cv2.boundingRect(contour)
#     cv2.rectangle(crop, (x, y), (x + w, y + h), (90, 0, 255), 2)
# cv2.imshow("crop", crop)
# if cv2.waitKey(0):
#     pass
# # cv2.imwrite(path, orig_image)
# print(f"Found {len(probs)} objects. The output image is {path}")
