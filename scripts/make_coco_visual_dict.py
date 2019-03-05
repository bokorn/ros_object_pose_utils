#!/usr/bin/env python

import os
import numpy as np
import cv2

from pycocotools.coco import COCO
#Based in part on https://github.com/bikz05/bag-of-words/blob/master/findFeatures.py

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

def rotateAndCrop(img, bbox, 
                  boarder_width = 10,
                  angle = 0):

    rows, cols = img.shape[:2]
    x,y,w,h = bbox
    y0 = min(max(y - boarder_width, 0), rows)
    x0 = min(max(x - boarder_width, 0), cols)
    y1 = min(max(y + h + boarder_width, 0), rows)
    x1 = min(max(x + w + boarder_width, 0), cols)

    if(angle != 0):
        corners = np.array([[x0,y0,1],[x0,y1,1],[x1,y0,1],[x1,y1,1]]).T
        center = (cols/2, rows/2)
        M = cv2.getRotationMatrix2D(center,angle,1.0)
        img_rot = cv2.warpAffine(img, M, (cols,rows))
        corners_rot = M.dot(corners)
        x0, y0 = np.min(corners_rot.astype(int), axis = 1)
        x1, y1 = np.max(corners_rot.astype(int), axis = 1)
        img_crop = img_rot[y0:y1,x0:x1]
    else:
        img_crop = img[y0:y1,x0:x1]

    return img_crop

def buildVisualDictionary(ann_path, dict_path,
                          category_names = [],
                          boarder_width = 10,
                          num_rotations = 0,
                          mask_images = False,
                          background_folder = None,
                          ): 
    coco = COCO(ann_path)
    cat_ids = coco.getCatIds(catNms=category_names)
    cats = coco.loadCats(cat_ids)
    class_names = {cat['id']:cat['name'] for cat in cats}
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    detector = cv2.xfeatures2d.SIFT_create()
    des_list = []
    cls_list = []

    for i_id in img_ids:
        img_info = coco.loadImgs(i_id)[0]
        img = cv2.imread(img_info['path'])
        for ann in coco.loadAnns(coco.getAnnIds(img_info['id'])):
            for _ in range(max(1,num_rotations)):
                if(background_folder is not None):
                    # Add random background
                    background = 255
                else:
                    background = 255
                if(mask_images):
                    mask = np.stack([coco.annToMask(ann)]*3, axis=2)
                    img_masked = img*mask + background*(1-mask)
                else:
                    img_masked = img

                if(num_rotations > 0):
                    angle = np.random.rand()*360
                else:
                    angle = 0
                img_crop =rotateAndCrop(img_masked, ann['bbox'], 
                                        boarder_width = boarder_width, 
                                        angle=angle)
                if(mask_images):
                    mask_crop = rotateAndCrop(mask, ann['bbox'],
                                              boarder_width = boarder_width,
                                              angle=angle)
                else:
                    mask_crop = None

                kpts, des = detector.detectAndCompute(img_crop, mask_crop)
                des_list.append(des)
                cls_list.append(ann['category_id'])

    descriptors = np.vstack(des_list)
    k = 100
    voc, variance = kmeans(descriptors, k, 1) 

    # Calculate the histogram of features
    num_des = len(des_list)
    im_features = np.zeros((num_des, k), "float32")
    for i in range(num_des):
        words, distance = vq(des_list[i],voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*num_des+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Train the Linear SVM
    clf = LinearSVC()
    clf.fit(im_features, np.array(cls_list))

    # Save the SVM
    joblib.dump((clf, class_names, stdSlr, k, voc), dict_path, compress=3) 

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('annotation_path', type=str)
    parser.add_argument('dictionary_path', type=str, default=None)
    parser.add_argument('--category_names', type=str, nargs='*', default=[])

    parser.add_argument('--boarder_width', type=int, default=10)
    parser.add_argument('--num_rotations', type=int, default=0)
    parser.add_argument('--mask_images', dest='mask_images', action='store_true')
    parser.add_argument('--background_folder', type=str, default=None)
    
    args = parser.parse_args()

    buildVisualDictionary(ann_path = args.annotation_path, 
                          dict_path = args.dictionary_path, 
                          category_names = args.category_names,
                          boarder_width = args.boarder_width,
                          num_rotations = args.num_rotations,
                          mask_images = args.mask_images,
                          background_folder = args.background_folder)

if __name__=='__main__':
    main()
