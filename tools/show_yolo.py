import imghdr
import os
import cv2
import numpy as np

root = 'G:\coco\yolo_pose'
type = 'val2017' 

img_root = os.path.join(root, 'images', type)
imgs_path = os.listdir(img_root)

for img_name in imgs_path:

    img_p = os.path.join(img_root, img_name)

    img = cv2.imread(img_p)
    img_h, img_w, _ = img.shape

    label_p = img_p.replace('images', 'labels').replace(os.path.splitext(img_p)[-1], '.txt')
    lables = np.loadtxt(label_p).reshape(-1, 56)

    lables[:, [1, 3]] *= img_w
    lables[:, [2, 4]] *= img_h
    lables[:, [1, 2]] -= lables[:, [3, 4]]/2.0
    lables[:, [3, 4]] += lables[:, [1, 2]]
    lables[:, 5::3] *= img_w
    lables[:, 6::3] *= img_h
    for label in lables:
        bbox = label[1:5]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3) 
        point = label[5:].reshape(-1, 3)
        cv2.circle(img, (int(point[15, 0]), int(point[15, 1])), 3, (0, 0, 255), 2)

    cv2.imshow('result', img)
    cv2.waitKey(0)