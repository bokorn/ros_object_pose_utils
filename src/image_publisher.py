#!/usr/bin/env python

import rospy
import glob
import numpy as np
import os
import yaml

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo

def parseCameraInfoYaml(filename):
    with file(filename, 'r') as f:
        calib_data = yaml.load(f)
        camera_info = CameraInfo()
        camera_info.width = calib_data['image_width']
        camera_info.height = calib_data['image_height']
        camera_info.K = calib_data['camera_matrix']['data']
        camera_info.D = calib_data['distortion_coefficients']['data']
        camera_info.R = calib_data['rectification_matrix']['data']
        camera_info.P = calib_data['projection_matrix']['data']
        camera_info.distortion_model = calib_data['distortion_model']
    return camera_info

def resizeCameraInfo(camera_info, new_size):
    y_scale = new_size[0]/camera_info.height
    x_scale = new_size[1]/camera_info.width
    camera_info.height = new_size[0]
    camera_info.width = new_size[1]
    # Camera Matrix
    camera_info.K[0] *= x_scale
    camera_info.K[2] *= x_scale
    camera_info.K[4] *= y_scale
    camera_info.K[5] *= y_scale
    # Projection Matrix
    camera_info.P[0] *= x_scale
    camera_info.P[2] *= x_scale
    camera_info.P[5] *= y_scale
    camera_info.P[6] *= y_scale

def main():
    rospy.init_node("image_publisher")    
    camera_info_file = rospy.get_param('~camera_info_file')
    image_folder = rospy.get_param('~image_folder')
    loop = rospy.get_param('~loop', False)
    image_ext = rospy.get_param('~image_ext', 'png')
    frame_id = rospy.get_param('~frame_id', 'camera')
    publish_rate = rospy.get_param('~publish_rate', 100)

    bridge = CvBridge()
    info_msg = parseCameraInfoYaml(camera_info_file)

    image_pub = rospy.Publisher('out_image', Image, queue_size = 5) 
    info_pub = rospy.Publisher('out_camera_info', CameraInfo, queue_size = 10)    

    image_filenames = sorted(glob.glob(os.path.join(image_folder, '*.'+image_ext)))
    num_images = len(image_filenames)
    rate = rospy.Rate(publish_rate)
    first_loop = True
    while(loop or first_loop):
        first_loop = False
        for j, fn in enumerate(image_filenames):
            if(rospy.is_shutdown()):
                break
            img = cv2.imread(fn)
            if(img.shape[:2] != (info_msg.height, info_msg.width)):
                resizeCameraInfo(info_msg, img.shape[:2])
            try:
                img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
            except CvBridgeError as err:
                rospy.logerr(err)
                continue

            timestamp = rospy.Time.now()
            info_msg.header.stamp = timestamp 
            img_msg.header.stamp = timestamp 
            info_msg.header.seq = j 
            img_msg.header.seq = j
            info_msg.header.frame_id = frame_id 
            img_msg.header.frame_id = frame_id
            
            rospy.loginfo('Publishing image {} ({} of {})'.format(fn, j, num_images))
            info_pub.publish(info_msg)
            image_pub.publish(img_msg)
            rate.sleep()

if __name__=='__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
