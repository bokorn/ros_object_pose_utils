#!/usr/bin/env python
from ast import literal_eval
from multiprocessing import Lock

import numpy as np
import cv2
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import message_filters

from functools import partial
from object_masker import ObjectMasker
from feature_classification import FeatureClassifier, classificationFunction, cropBBox

roslib.load_manifest("rosparam")

class ObjectMaskerNode(object):

    def __init__(self):
        visual_dict = rospy.get_param('~visual_dict')
        self.image_roi = rospy.get_param('~image_roi',  default=None)

        if self.image_roi is not None:
            self.image_roi = literal_eval(self.image_roi)

        self.filter_size = rospy.get_param('~filter_size', default=41)
        self.filter_const = rospy.get_param('~filter_const', default=20)

        self.info_mutex = Lock()
        self.bridge = CvBridge()
        
        self.image_sub = message_filters.Subscriber('in_image', Image)
        self.info_sub = message_filters.Subscriber('in_camera_info', CameraInfo)
        self.image_pub = rospy.Publisher('out_image', Image, queue_size = 1)

        self.masker = ObjectMasker(self.filter_size, self.filter_const, self.image_roi)
        self.classifier = FeatureClassifier(visual_dict)
        self.category_func = partial(classificationFunction, classifier=self.classifier)
        self.category_names = [v for _, v in sorted(self.classifier.class_names.items())] 
        self.obj_idxs = {k:0 for k in self.classifier.class_names.keys()} 
        
        self.output_folder = rospy.get_param('~output_folder', default=None)
        if(self.output_folder is not None):
            for cat_name in self.category_names:
                if not os.path.exists(os.path.join(self.output_folder, cat_name)):
                    os.makedirs(os.path.join(self.output_folder, cat_name))

        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], queue_size = 100)
        self.ts.registerCallback(self.imageCallback)

    def imageCallback(self, img_msg, info_msg):
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            header = img_msg.header
        except CvBridgeError as err:
            rospy.logerr(err)
            return

        masks, mask_idxs = self.masker.getMasks(img)

        ann_img = self.masker.getAnnotations(img, masks, mask_idxs, 
                                             category_names = self.category_names,
                                             category_func = self.category_func)
        
        display_img = ann_img.draw(thickness=1, color_by_category=True)
       
        if(self.output_folder is not None):
            for ann in ann_img.annotations.values():
                img_crop = cropBBox(img, ann.mask.bbox())
                mask_crop = cropBBox(ann.mask.array, ann.mask.bbox())
                obj_img = np.concatenate([img_crop, np.expand_dims(mask_crop,2)*255], axis=2)
                cat_id = ann.category.id
                cv2.imwrite(os.path.join(self.output_folder, 
                    self.classifier.class_names[cat_id], 
                    "{:06}.png".format(self.obj_idxs[cat_id])), 
                    obj_img)
                self.obj_idxs[cat_id] += 1

        try:
            display_msg = self.bridge.cv2_to_imgmsg(display_img.astype(np.uint8), encoding="bgr8")
        except CvBridgeError as err:
            rospy.logerr(err)
            return

        display_msg.header = img_msg.header
        self.image_pub.publish(display_msg)