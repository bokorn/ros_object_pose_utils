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

from ObjectMasker import ObjectMasker

roslib.load_manifest("rosparam")


def isTopLevel(hier):
    return hier[3] == -1


class ObjectMaskerNode(object):
    def __init__(self):
        rospy.init_node("object_masker")    
        
        self.image_roi = rospy.get_param('~image_roi',  default=None)

        if(self.image_roi is not None):
            self.image_roi = literal_eval(self.image_roi)

        self.filter_size = rospy.get_param('~filter_size', default=41)
        self.filter_const = rospy.get_param('~filter_const', default=20)

        self.info_mutex = Lock()
        self.bridge = CvBridge()
        
        self.image_sub = message_filters.Subscriber('in_image', Image)
        self.info_sub = message_filters.Subscriber('in_camera_info', CameraInfo)
        self.image_pub = rospy.Publisher('out_image', Image, queue_size = 1)

        self.masker = ObjectMasker(self.filter_size, self.filter_const)

        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], queue_size = 100)
        self.ts.registerCallback(self.imageCallback)

    def imageCallback(self, img_msg, info_msg):
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            header = img_msg.header
        except CvBridgeError as err:
            rospy.logerr(err)
            return

        contours, hierarchy = self.masker.getMasks(img)

        display_img = img.copy()
        for cnt, hier in zip(contours, hierarchy[0]):
            cv2.drawContours(display_img, [cnt], 0, (0,0,255), 1)

            if isTopLevel(hier):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(display_img,(x,y),(x+w,y+h),(0,255,0), 1)

        try:
            display_msg = self.bridge.cv2_to_imgmsg(display_img.astype(np.uint8), encoding="bgr8")
        except CvBridgeError as err:
            rospy.logerr(err)
            return

        display_msg.header = img_msg.header
        self.image_pub.publish(display_msg)

