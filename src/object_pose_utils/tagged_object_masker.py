#!/usr/bin/env python

import rospy
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError
from multiprocessing import Lock

import tf2_ros
from tf2_geometry_msgs import PointStamped
import tf

from sensor_msgs.msg import Image, CameraInfo
import image_geometry
import message_filters

from tf.transformations import quaternion_matrix
import xml.etree.ElementTree as ET

from object_pose_utils.object_masker import ObjectMasker
from object_pose_utils.object_masker import AnnotationMapper


import roslib
roslib.load_manifest("rosparam")


def parseConfigXml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    obj_filenames = {}
    pcd_filenames = {}
    poses = {}
    for obj in root:
        obj_filenames[obj.get('name')] = obj.find('obj_file').text
        pcd_filenames[obj.get('name')] = obj.find('pcd_file').text
        t = np.fromstring(obj.find('translation').text, sep=',')
        q = np.fromstring(obj.find('orientation').text, sep=',')
        trans_mat = tf.transformations.quaternion_matrix(q)
        trans_mat[:3,3] = t
        poses[obj.get('name')] = trans_mat
    return poses, pcd_filenames, obj_filenames


class TaggedAnnotationMapper(AnnotationMapper):

    def __init__(self, object_centers):
        """

        Args:
            object_centers: dictionary of annotation_idx to object centers
        """
        self.object_centers = object_centers

    def sort(self, image, masks, mask_idx, annotations, annotation_idx):
        pass


class TaggedObjectMasker(object):
    def __init__(self):

        config_filename = rospy.get_param('~config_file')
        self.obj_poses, _, _ = parseConfigXml(config_filename)
        self.model = image_geometry.PinholeCameraModel()
        self.frame_id = rospy.get_param('~board_frame', 'marker_bundle')
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

        self.object_centers = {}
        for name, pose in self.obj_poses.items():
            center = PointStamped()
            center.point.x = pose[0, 3]
            center.point.y = pose[1, 3]
            center.point.z = pose[2, 3]
            center.header.frame_id = self.frame_id
            self.object_centers[name] = center

        thresh_block_size = rospy.get_param("~thresh_block_size", 30)
        thresh_const = rospy.get_param("~thresh_const", 5)
        self.object_masker = ObjectMasker(thresh_block_size, thresh_const)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_interface = tf2_ros.BufferInterface()
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
       
        self.model.fromCameraInfo(info_msg)

        frame_id = header.frame_id
        if(frame_id[0] == '/'):
            frame_id = frame_id[1:]
        
        proj_img = np.zeros(img.shape[:2])

        # try:
        #     if self.tf_buffer.can_transform(self.frame_id, frame_id, header.stamp, rospy.Duration(5)):
        #         trans = self.tf_buffer.lookup_transform(frame_id, self.frame_id, header.stamp, rospy.Duration(0.1))
        #         trans_mat = tf.transformations.quaternion_matrix([trans.transform.rotation.x,
        #                                                           trans.transform.rotation.y,
        #                                                           trans.transform.rotation.z,
        #                                                           trans.transform.rotation.w])
        #         trans_mat[:3, 3] = [trans.transform.translation.x,
        #                             trans.transform.translation.y,
        #                             trans.transform.translation.z]
        #
        #         K = np.array([[self.model.fx(), 0, self.model.Tx(), 0],
        #                       [0, self.model.fy(), self.model.Ty(), 0],
        #                       [0, 0, 1, 0]])
        #         P = K.dot(trans_mat)
        #         c_img = np.array([[self.model.cx(), self.model.cy(), 0]]).T
        #         for idx, pts in enumerate(self.obj_pts.values()):
        #             pts_img = P.dot(pts)
        #             pts_img = pts_img/pts_img[2, :] + c_img
        #             u = pts_img[0, :]
        #             v = pts_img[1, :]
        #             mask = np.logical_and.reduce((u >= 0, u < img.shape[1], v >= 0, v < img.shape[0]))
        #             u = u[mask]
        #             v = v[mask]
        #             proj_img[v.astype(int), u.astype(int)] = idx + 1
        # except tf2_ros.ExtrapolationException as err:
        #     rospy.logwarn(err)
        #     return

        # Transform the object centers into the image frame
        object_centers_projected = {}
        try:
            for name, pt in self.object_centers.items():
                pt.header.stamp = header.stamp
                if self.tf_buffer.can_transform(frame_id, pt.header.frame_id, pt.header.stamp, rospy.Duration(0.1)):
                    center_transformed = self.tf_buffer.transform(pt, frame_id, rospy.Duration(0.5)).point
                    x = center_transformed.x
                    y = center_transformed.y
                    z = center_transformed.z
                    center_projected = self.model.project3dToPixel([x,y,z])
                    object_centers_projected[name] = (int(center_projected[0]), int(center_projected[1]))
                else:
                    rospy.logwarn('TF not available')
        except tf2_ros.ExtrapolationException as err:
            rospy.logwarn(err)


        for name, center_projected in object_centers_projected.items():
            cv2.putText(img, name, center_projected, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(img, center_projected, 10, (0, 255, 0), -1)


        cv2.imshow("object_centers", img)
        cv2.waitKey(0)

        # Transform the board corners into the image frame
        trans_corners = []
        try:
            for pt in self.board_corners:
                pt.header.stamp = header.stamp
                if(self.tf_buffer.can_transform(frame_id, pt.header.frame_id, 
                    pt.header.stamp, rospy.Duration(0.1))):
                    corner = self.tf_buffer.transform(pt, frame_id, rospy.Duration(0.5))
                    trans_corners.append(np.array([corner.point.x, corner.point.y, corner.point.z]))
                else:
                    rospy.logwarn('TF not availble')
                    return
        except tf2_ros.ExtrapolationException as err:
            rospy.logwarn(err)
            return


        pixel_corners = []
        for j, pt in enumerate(trans_corners):
            pixel_pt = np.array(self.model.project3dToPixel(pt)).astype(int) 
            pixel_corners.append(pixel_pt)
        pixel_corners = np.array(pixel_corners)
        
        board_mask = np.zeros_like(img[:, :, 0])
        cv2.fillConvexPoly(board_mask, pixel_corners, 1)

        board_size = board_mask[:, :].sum()
        markers, marker_idx = self.object_masker.getMasks(img, roi_mask=board_mask)

        marker_centers = []
        dists = []
        for idx in marker_idx:
            M = cv2.moments((markers == idx).astype(np.uint8))
            x_c = M["m10"] / M["m00"]
            y_c = M["m01"] / M["m00"]
            center = np.array([y_c, x_c])
            marker_centers.append(center)
            dists.append(np.linalg.norm(pixel_corners[0, :]-np.array([y_c, x_c])))

        annotated_image = self.object_masker.getAnnotations(img, markers, marker_idx)

        d_idxs = np.argsort(dists)

        markers_remapped = np.zeros(markers.shape)
        for j, idx in enumerate(d_idxs):
            markers_remapped[markers == marker_idx[idx]] = j + 1

        combined_img = proj_img + markers_remapped
        display_img = cv2.applyColorMap(combined_img.astype(np.uint8)*42, cv2.COLORMAP_JET)
        try:
            display_msg = self.bridge.cv2_to_imgmsg(display_img.astype(np.uint8), encoding="bgr8")
        except CvBridgeError as err:
            rospy.logerr(err)
            return

        display_msg.header = img_msg.header
        self.image_pub.publish(display_msg)

        cv2.imshow("display_img", display_img)
        cv2.waitKey(0)
