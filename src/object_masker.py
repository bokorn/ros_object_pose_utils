#!/usr/bin/env python

import rospy
import glob
import numpy as np
import os
import yaml

import cv2
from cv_bridge import CvBridge, CvBridgeError
from multiprocessing import Lock

import tf2_ros
from tf2_geometry_msgs import PointStamped

from sensor_msgs.msg import Image, CameraInfo
import image_geometry
import message_filters

from tf.transformations import quaternion_matrix


import matplotlib as mpl
import matplotlib.pyplot as plt

import roslib
roslib.load_manifest("rosparam")
import rosparam

def filterMarkers(markers, thresh_min, thresh_max):
    idx, counts = np.unique(markers, return_counts=True)
    idx_out = []
    counts_out = []
    for i, c in zip(idx, counts):
        if thresh_min < c and c < thresh_max:
            idx_out.append(i)
            counts_out.append(c)
    return idx_out, counts_out

def thresholdImage(img, invert=False):
    if(invert):
        tresh_flags = cv2.THRESH_BINARY_INV
    else:
        tresh_flags = cv2.THRESH_BINARY

    #img = cv2.GaussianBlur(img,(3,3),0)
    #threshval, img_out = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_out = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    tresh_flags,41,2)
    return img_out

def segmentImage(image):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image_bw = cv2.GaussianBlur(image_bw,(7,7),0)
    image_thr = thresholdImage(image_bw, invert=True)
    #idx, counts = np.unique(markers, return_counts=True)
    #idx_filt, counts_out = filterMarkers(idx, counts, 2500, 10000)
     
    #kernel = np.ones((7,7), np.uint8)
    #image_thr = cv2.morphologyEx(image_thr, cv2.MORPH_CLOSE, kernel, iterations = 1)
    _, markers = cv2.connectedComponents(image_thr)
    return markers

def getCMapImage(image, cmap='gray'):
    img_cmap = plt.imshow(image, cmap=cmap)._rgba_cache
    return img_cmap

class ObjectMasker(object):
    def __init__(self, obj_filename = '/home/bokorn/data/surgical/models/scalpel.obj',# scissor.obj',
                 ):
        rospy.init_node("object_masker")    

        #paramlist=rosparam.load_file("/path/to/myfile",default_namespace="my_namespace")
        #for params, ns in paramlist:
        #    rosparam.upload_params(ns,params)
        
        self.model = image_geometry.PinholeCameraModel()

        self.frame_id = rospy.get_param('~board_frame', 'ar_marker_0')
        self.info_mutex = Lock()
         
        ll_corner = PointStamped()
        ll_corner.point.x = 0.00
        ll_corner.point.y = 0.05
        ll_corner.point.z = 0.0
        ll_corner.header.frame_id = self.frame_id

        lr_corner = PointStamped()
        lr_corner.point.x = 0.40
        lr_corner.point.y = 0.05
        lr_corner.point.z = 0.0
        lr_corner.header.frame_id = self.frame_id

        ur_corner = PointStamped()
        ur_corner.point.x = 0.40
        ur_corner.point.y = 0.25
        ur_corner.point.z = 0.0
        ur_corner.header.frame_id = self.frame_id

        ul_corner = PointStamped()
        ul_corner.point.x = 0.00
        ul_corner.point.y = 0.25
        ul_corner.point.z = 0.0
        ul_corner.header.frame_id = self.frame_id

        self.board_corners = [ll_corner, lr_corner, ur_corner, ul_corner]; 

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_interface = tf2_ros.BufferInterface()
        self.bridge = CvBridge()
        
        self.image_sub = message_filters.Subscriber('in_image', Image)
        self.info_sub = message_filters.Subscriber('in_camera_info', CameraInfo)
        self.image_pub = rospy.Publisher('out_image', Image, queue_size = 1)

        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], queue_size = 100)
        self.ts.registerCallback(self.imageCallback)
        
        #self.rot = quaternion_from_euler(config['angle_x']*np.pi/180,
        #                                 config['angle_y']*np.pi/180,
        #                                 config['angle_z']*np.pi/180,
        #                                 axes='rxyz')
        #self.trans = np.array([config['trans_x'], config['trans_y'], config['trans_z']])
        #self.trans_mat = quaternion_matrix(self.rot)
        #self.trans_mat[:3, 3] = self.trans


    def imageCallback(self, img_msg, info_msg):
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            trans_marker = self.tf_buffer.lookup_transform(img_msg.header.frame_id[1:],
                                                           self.frame_id, 
                                                           img_msg.header.stamp,
                                                           rospy.Duration(0.01))
            self.board_trans = np.array([trans_marker.transform.translation.x,
                                         trans_marker.transform.translation.y,
                                         trans_marker.transform.translation.z])
            self.board_rot = np.array([trans_marker.transform.rotation.x,
                                       trans_marker.transform.rotation.y,
                                       trans_marker.transform.rotation.z,
                                       trans_marker.transform.rotation.w])
            self.board_mat = quaternion_matrix(self.board_rot)
            self.board_mat[:3, 3] = self.board_trans
            header = img_msg.header
        except CvBridgeError as err:
            rospy.logerr(err)
            return
       
        self.model.fromCameraInfo(info_msg)
        trans_corners = []
        frame_id = header.frame_id
        if(frame_id[0] == '/'):
            frame_id = frame_id[1:]
        try:
            for pt in self.board_corners:
                pt.header.stamp = header.stamp
                if(self.tf_buffer.can_transform(frame_id, pt.header.frame_id, 
                    pt.header.stamp, rospy.Duration(0.1))):
                    corner = self.tf_buffer.transform(pt, frame_id, rospy.Duration(0.5))
                    trans_corners.append(np.array([corner.point.x, corner.point.y, corner.point.z]))
        except tf2_ros.ExtrapolationException as err:
            rospy.logwarn(type(err))
            rospy.logwarn(err)
            return
        pixel_corners = []
        for j, pt in enumerate(trans_corners):
            pixel_pt = np.array(self.model.project3dToPixel(pt)).astype(int) 
            pixel_corners.append(pixel_pt)
        pixel_corners = np.array(pixel_corners)
        
        board_mask = np.zeros_like(img);
        cv2.fillConvexPoly(board_mask, pixel_corners, 1)
        
        #trans_mat = self.board_mat.dot(self.trans_mat)
        board_size = board_mask[:,:,0].sum() 
        all_markers = segmentImage(img) 
        markers_masked = (all_markers + 1)*board_mask[:,:,0]

        filtered_idxs, filtered_counts = filterMarkers(markers_masked, board_size/500, board_size/3)
        filtered_idxs = np.array(filtered_idxs)[np.argsort(filtered_counts)[-3:]] - 1
        markers = np.zeros_like(all_markers)
        dists = []
        for idx in filtered_idxs:
            M = cv2.moments((all_markers == idx).astype(np.uint8)) 
            x_c = M["m10"] / M["m00"]
            y_c = M["m01"] / M["m00"]
            dists.append(np.linalg.norm(pixel_corners[0,:]-np.array([y_c,x_c])))

        d_idxs = np.argsort(dists)
        
        for j, idx in enumerate(d_idxs):
        #for j, idx in enumerate(filtered_idxs):
            markers[all_markers == (filtered_idxs[idx])] = j+1
            #rospy.loginfo(np.sum(markers_masked == idx))
            #markers[markers_masked == idx] = j

        display_img = cv2.applyColorMap(markers.astype(np.uint8)*85, cv2.COLORMAP_JET) 
        #display_img = cv2.applyColorMap(markers_masked.astype(np.uint8), cv2.COLORMAP_JET) 
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
   
