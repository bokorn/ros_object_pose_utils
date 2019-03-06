#!/usr/bin/env python

import os
import numpy as np
import cv2

#Based in part on https://github.com/bikz05/bag-of-words/blob/master/getClass.py
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

def cropBBox(img, bbox, boarder_width = 10):
    rows, cols = img.shape[:2]
    x0,y0,x1,y1 = bbox
    y0 = min(max(y0 - boarder_width, 0), rows)
    x0 = min(max(x0 - boarder_width, 0), cols)
    y1 = min(max(y1 + boarder_width, 0), rows)
    x1 = min(max(x1 + boarder_width, 0), cols)
    img_crop = img[y0:y1,x0:x1]

    return img_crop

def classificationFunction(img, anns, classifier):
    cls_dict = {}
    for k,v in anns.items():
        bbox = v['bbox']
        img_crop = cropBBox(img, bbox.bbox())
        cls = classifier.classify(img_crop)[0] 
        cls_dict[k] = cls-1
    return cls_dict

class FeatureClassifier(object):
    def __init__(self, voc_file):
        self.clf, self.class_names, self.stdSlr, self.k, self.voc = joblib.load(voc_file)
        self.detector = cv2.xfeatures2d.SIFT_create()

    def featurize(self, img, mask = None):
        kpts, des = self.detector.detectAndCompute(img, mask)
        feature = np.zeros([1,self.k], "float32")
        success = False
        if(des is not None):
            words, distance = vq(des, self.voc)
            for w in words:
                feature[0][w] += 1
            feature = self.stdSlr.transform(feature)
            success = True

        return feature, success

    def predict(self, feature):
        cls_id = self.clf.predict(feature)[0]
        cls_name = self.class_names[cls_id]
        return cls_id, cls_name


    def classify(self, img, mask = None):
        features, success = self.featurize(img, mask)
        if(success):
            return self.predict(features)
        else:
            return -1, 'Featurization Failure'

