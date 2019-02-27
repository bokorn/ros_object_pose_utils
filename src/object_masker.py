#!/usr/bin/env python

import rospy
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo
import message_filters

from ast import literal_eval

import roslib
roslib.load_manifest("rosparam")
import rosparam

import imantics

def filterMarkers(markers, thresh_min, thresh_max):
    idx, counts = np.unique(markers, return_counts=True)
    idx_out = []
    counts_out = []
    for i, c in zip(idx, counts):
        if thresh_min < c and c < thresh_max:
            idx_out.append(i)
            counts_out.append(c)
    return idx_out, counts_out

def segmentImage(img, block_size=11, C=10):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thr = cv2.adaptiveThreshold(img_bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, C)

    _, markers = cv2.connectedComponents(img_thr)
    return markers

def getCMapImage(image, cmap='gray'):
    img_cmap = plt.imshow(image, cmap=cmap)._rgba_cache
    return img_cmap

class ObjectMasker(object):
    def __init__(self,
                 ):
        rospy.init_node("object_masker")    
        
        self.image_roi = rospy.get_param('~image_roi',  default=None)
        if(self.image_roi is not None):
            self.image_roi = literal_eval(self.image_roi)
        self.filter_size = rospy.get_param('~filter_size', default=41)
        self.filter_const = rospy.get_param('~filter_const', default=20)
	        
        self.bridge = CvBridge()
        
        self.image_sub = message_filters.Subscriber('in_image', Image)
        self.info_sub = message_filters.Subscriber('in_camera_info', CameraInfo)
        self.image_pub = rospy.Publisher('out_image', Image, queue_size = 1)

        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], queue_size = 100)
        self.ts.registerCallback(self.imageCallback)

    def imageCallback(self, img_msg, info_msg):
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            header = img_msg.header
        except CvBridgeError as err:
            rospy.logerr(err)
            return
       
        board_mask = np.zeros_like(img[:,:,0]);
        if(self.image_roi is None):
	        board_mask[:,:] = 1;
        else:
            board_mask[self.image_roi[0]:self.image_roi[1], self.image_roi[2]:self.image_roi[3]] = 1
        board_size = board_mask.sum() 
        all_markers = segmentImage(img, self.filter_size, self.filter_const) 
        markers_masked = (all_markers + 1)*board_mask

        filtered_idxs, filtered_counts = filterMarkers(markers_masked, board_size/250, board_size/3)
        markers_masked[np.isin(markers_masked, filtered_idxs, invert=True)] = 0
        ann_img = imantics.Image(image_array = img)
        colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0)]
        for j, idx in enumerate(filtered_idxs):
            mask = imantics.Mask((markers_masked == idx).astype(np.uint8))
            ann = imantics.Annotation(image = ann_img, 
                                      category = imantics.Category('obj_{}'.format(j), 
                                                                   color=imantics.Color(rgb=colors[j])),
                                      mask=mask, bbox=imantics.BBox.from_mask(mask))
            ann_img.add(ann)
        display_img = ann_img.draw(thickness=1, color_by_category=True)
        
        try:
            display_msg = self.bridge.cv2_to_imgmsg(display_img.astype(np.uint8), encoding="bgr8")
        except CvBridgeError as err:
            rospy.logerr(err)
            return

        display_msg.header = img_msg.header
        self.image_pub.publish(display_msg)

def main():
    rospy.init_node("object_masker") 
    obj_masker = ObjectMasker()
  
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down pose_labeler module")

if __name__=='__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
   
