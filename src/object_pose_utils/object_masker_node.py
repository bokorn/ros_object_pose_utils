#!/usr/bin/env python
from ast import literal_eval
from multiprocessing import Lock
import os

import numpy as np
import cv2
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import message_filters

from object_masker import ObjectMasker
from feature_classification import FeatureClassifier, ClassificationMapper, cropBBox

from dynamic_reconfigure.server import Server
from object_pose_utils.cfg import ObjectMaskerConfig

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

        self.info_msg = None
        self.img_msg = None
        self.new_camera_data = False

        self.image_sub = message_filters.Subscriber('in_image', Image)
        self.info_sub = message_filters.Subscriber('in_camera_info', CameraInfo)
        self.object_select_sub = rospy.Subscriber('object_select', String, self.object_select_cb)
        self.image_pub = rospy.Publisher('out_image', Image, queue_size = 1)
        self.mask_pub = rospy.Publisher('out_mask', Image, queue_size=1)

        self.debug = rospy.get_param('~debug', default=False)

        self.masker = ObjectMasker(self.filter_size, self.filter_const, self.image_roi, self.debug)
        self.classifier = FeatureClassifier(visual_dict)
        self.mapper = ClassificationMapper(self.classifier)
        self.categories = self.classifier.class_names
        self.obj_idxs = {k: 0 for k in self.classifier.class_names.keys()}

        self.object_select = rospy.get_param('~object_select', default=None)
        self.output_folder = rospy.get_param('~output_folder', default=None)


        if self.output_folder is not None:
            for cat_name in self.category_names:
                if not os.path.exists(os.path.join(self.output_folder, cat_name)):
                    os.makedirs(os.path.join(self.output_folder, cat_name))

        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], queue_size = 5)
        self.ts.registerCallback(self.imageCallback)

        self.dynamic_reconfigure_server = Server(ObjectMaskerConfig, self.dynamic_reconfigure_cb)
        self.loop_timer = rospy.Timer(rospy.Duration(0.05), self.process_image)

    def dynamic_reconfigure_cb(self, config, level):

        filter_size = (config.filter_size // 2) * 2 + 1
        filter_const = config.filter_const
        morph_kernel_size = (config.morph_kernel_size // 2) * 2 + 1

        self.masker.setThreshBlockSize(filter_size)
        self.masker.setThreshConst(filter_const)
        self.masker.setVisualizationDebug(config.debug)
        self.masker.setMorphKernelSize(morph_kernel_size)

        config.filter_size = filter_size
        config.morph_kernel_size = morph_kernel_size
        return config

    def object_select_cb(self, msg):
        self.object_select = msg.data

    def imageCallback(self, img_msg, info_msg):
        if self.info_mutex.acquire(False):
            try:
                self.new_camera_data = True
                self.img_msg = img_msg
                self.info_msg = info_msg
            finally:
                self.info_mutex.release()

    def process_image(self, event):

        if not self.new_camera_data:
            return
        
        with self.info_mutex:
            self.new_camera_data = False
            img_msg = self.img_msg
            info_msg = self.info_msg
            
            try:
                img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                header = img_msg.header
            except CvBridgeError as err:
                rospy.logerr(err)
                return

            masks, mask_idxs = self.masker.getMasks(img)
            ann_img, category_map = self.masker.getAnnotations(img, masks, mask_idxs,categories=self.categories, mapper=self.mapper)
            
            mask_img = np.zeros_like(masks, dtype=np.uint8)
            name_to_label = {}
            output_object = None

            if self.object_select is not None:
                if self.object_select in self.categories.values():
                    output_object = self.object_select

            match_count = 0
            # build the mask image
            for mask_id, cat_id in sorted(category_map.items(), key=lambda x: x[1]):
                try:
                    cat_id = np.uint8(cat_id)

                    # if there is a valid output object name, set the mask to 1.  This allows for a single object mask
                    if self.categories[cat_id] == output_object:
                        mask_img[masks == mask_id] = 1
                        match_count += 1
                    # else, set the mask to the category id
                    if output_object is None:
                        mask_img[masks == mask_id] = cat_id
                    name_to_label[self.categories[cat_id]] = cat_id
                except KeyError:
                    pass

            if match_count > 0:
                rospy.loginfo('found {} instances of {}'.format(match_count, output_object))

            display_img = ann_img.draw(thickness=1, color_by_category=True)
        
            if self.output_folder is not None:
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
                mask_msg = self.bridge.cv2_to_imgmsg(cv2.normalize(mask_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX), encoding="mono8")
            except CvBridgeError as err:
                rospy.logerr(err)
                return

            if self.debug:
                cv2.imshow("display_msg", display_img)


            if match_count > 0:
                display_msg.header = img_msg.header
                mask_msg.header = img_msg.header
                self.image_pub.publish(display_msg)
                self.mask_pub.publish(mask_msg)
