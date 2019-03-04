#!/usr/bin/env python

import os
import numpy as np
import cv2

#Based in part on https://github.com/bikz05/bag-of-words/blob/master/getClass.py
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

class FeatureClassifier(object):
    def __init__(self, voc_file):
        self.clf, self.class_names, self.stdSlr, self.k, self.voc = joblib.load(voc_file)
        self.detector = cv2.xfeatures2d.SIFT_create()

    def featurize(self, img, mask = None):
        kpts, des = self.detector.detectAndCompute(img, mask)
        words, distance = vq(des, self.voc)
        feature = np.zeros(self.k, "float32")
        for w in words:
            feature[w] += 1
        feature = self.stdSlr.transform([feature])
        return feature

    def predict(self, feature):
        cls_id = self.clf.predict(feature)[0]
        cls_name = self.class_names[cls_id]
        return cls_id, cls_name


    def classify(self, img, mask = None):
        return self.predict(self.featurize(img, mask))
