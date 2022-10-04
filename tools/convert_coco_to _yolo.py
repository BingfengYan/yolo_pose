import os
import shutil
from types import new_class
import cv2
import numpy as np
from pycocotools.coco import COCO

root = 'G:\coco'
type = 'train2017' 
json_path = os.path.join(root, 'annotations\person_keypoints_%s.json'%type)
coco = COCO(json_path)

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

for imgid in imgIds:

    img = coco.loadImgs(imgid)[0]

    img_path = os.path.join(root, 'images', type, img['file_name'])
    new_img_path = img_path.replace('images', 'yolo_pose/images')
    os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
    shutil.copy(img_path, new_img_path)

    img_h, img_w = img['height'], img['width']
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    label_p = new_img_path.replace('images', 'labels').replace(os.path.splitext(new_img_path)[-1], '.txt')
    os.makedirs(os.path.dirname(label_p), exist_ok=True)
    with open(label_p, 'w') as fp:
        for ann in anns:
            bbox = np.array(ann['bbox']).astype(np.float32)
            bbox[[0,1]] += bbox[[2,3]]/2.0
            bbox[[0,2]] /= img_w
            bbox[[1,3]] /= img_h

            points = np.array(ann['keypoints']).astype(np.float32).reshape(-1, 3)
            points[:, 0] /= img_w
            points[:, 1] /= img_h

            fp.write('0 %f %f %f %f'%(bbox[0], bbox[1], bbox[2], bbox[3]))
            for p in points:
                fp.write(' %f %f %f'%(p[0], p[1], p[2]))
            fp.write('\n')



