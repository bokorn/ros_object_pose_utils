#!/usr/bin/env python

import rospy
import numpy as np
import os

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

from collections import namedtuple
import re

ImageItem = namedtuple("ImageItem", ["name", "pose", "obj_file", "pcd_file"])


# regex to break file into <stub><integer><.extension>
rg = re.compile("(.*?)(\\d+)(\..*?)$", re.IGNORECASE)


def find_largest_index(folder, stub, extension):
    """ Get the largest file index of a file in a folder

    Args:
        folder: folder to search
        stub: "main" part of the filename to match
        extension: the ".xyz" file extension to match

    Returns:
        Largest index matched, or -1 if none found
    """
    index = -1
    for root, dirnames, filenames in os.walk(folder):
        for file in sorted(filenames):
            m = rg.search(file)
            if m:
                st, idx, ext = m.groups()
                if st == stub and ext == extension:
                    if idx > index:
                        index = idx
        break
    return index


def parseConfigXml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    objects = []
    for obj in root:
        name = obj.get('name')
        obj_file = obj.find('obj_file').text
        pcd_file = obj.find('pcd_file').text
        trans = np.fromstring(obj.find('translation').text, sep=',')
        rot = np.fromstring(obj.find('orientation').text, sep=',')
        trans_mat = tf.transformations.quaternion_matrix(rot)
        trans_mat[:3,3] = trans
        item = ImageItem(name, trans_mat, obj_file, pcd_file)
        objects.append(item)
    return objects


class TaggedAnnotationMapper(AnnotationMapper):

    def __init__(self, object_centers):
        """

        Args:
            object_centers: dictionary of category name to object centers
        """
        self.object_centers = object_centers

    def sort(self, image, masks, categories, annotations):
        """
        Args:
            image: input image as a BGR ndarray
            masks: connected component mask
            categories: dictionary of names to category IDs  (string : int)
            annotations: dictionary of mask_ids to imantics.Annotation objects

        Returns:
            Mapping of mask_idx to annotation_idx as a python dictionary
        """

        category_to_idx = {}
        img = image.copy()

        for mask_id, annotation in annotations.items():
            min_dist = np.inf
            bbox = annotation.bbox
            width = abs(bbox.top_left[0] - bbox.top_right[0])
            height = abs(bbox.top_left[1] - bbox.bottom_left[1])
            annotation_center = np.array(bbox.top_left) + np.array([width / 2, height / 2])
            #cv2.circle(img, (annotation_center[0], annotation_center[1]), 5, (0,0,255), -1)

            for object_category, object_center in self.object_centers.items():
                dist = np.linalg.norm(object_center - annotation_center)
                if dist < min_dist:
                    min_dist = dist
                    category_to_idx[mask_id] = object_category

        # put labeled circles into the image
        # for category_id, center_projected in self.object_centers.items():
        #     cv2.putText(img, categories[category_id], center_projected, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     cv2.circle(img, center_projected, 5, (0, 255, 0), -1)
        # cv2.imshow("Tagged object masker", img)
        # cv2.waitKey(0)

        return category_to_idx


class TaggedObjectMasker(object):
    def __init__(self, config_filename, board_frame='board_frame', thresh_block_size=30, thresh_const=5, output_folder="./output"):

        self.objects = parseConfigXml(config_filename)
        self.model = image_geometry.PinholeCameraModel()
        self.board_frame = board_frame
        self.info_mutex = Lock()

        # inner dimensions of the board are x=48cm, y=24cm
        # add another 1 cm of padding around the border

        ll_corner = PointStamped()
        ll_corner.point.x = 0.02
        ll_corner.point.y = 0.02
        ll_corner.point.z = 0.0
        ll_corner.header.frame_id = self.board_frame

        lr_corner = PointStamped()
        lr_corner.point.x = 0.42
        lr_corner.point.y = 0.02
        lr_corner.point.z = 0.0
        lr_corner.header.frame_id = self.board_frame

        ur_corner = PointStamped()
        ur_corner.point.x = 0.42
        ur_corner.point.y = 0.20
        ur_corner.point.z = 0.0
        ur_corner.header.frame_id = self.board_frame

        ul_corner = PointStamped()
        ul_corner.point.x = 0.02
        ul_corner.point.y = 0.20
        ul_corner.point.z = 0.0
        ul_corner.header.frame_id = self.board_frame

        self.board_corners = [ll_corner, lr_corner, ur_corner, ul_corner]

        self.object_centers = {}
        self.categories = {}
        for j, object in enumerate(self.objects):
            center = PointStamped()
            center.point.x = object.pose[0, 3]
            center.point.y = object.pose[1, 3]
            center.point.z = object.pose[2, 3]
            center.header.frame_id = self.board_frame
            self.object_centers[j] = center
            self.categories[j] = object.name

        self.object_masker = ObjectMasker(thresh_block_size, thresh_const)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_interface = tf2_ros.BufferInterface()
        self.bridge = CvBridge()
        
        self.image_sub = message_filters.Subscriber('in_image', Image)
        self.info_sub = message_filters.Subscriber('in_camera_info', CameraInfo)
        self.image_pub = rospy.Publisher('out_image', Image, queue_size = 1)
        self.image_pub_debug = rospy.Publisher('out_image_debug', Image, queue_size=1)

        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], queue_size = 100)
        self.ts.registerCallback(self.imageCallback)
        self.new_data = False

        self.image_folder = output_folder
        self.annotation_folder = output_folder
        self.image_stub = "img_"
        self.image_ext = "jpg"
        self.image_id = find_largest_index(self.image_folder, self.image_stub, self.image_ext) + 1
        self.annotation_ext = 'json'


    def imageCallback(self, img_msg, info_msg):

        # convert the incoming images into a BGR image for opencv
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            img_debug = img.copy()
            header = img_msg.header
        except CvBridgeError as err:
            rospy.logerr(err)
            return

        # Create the pinhole camera model using the camera info message
        self.model.fromCameraInfo(info_msg)

        frame_id = header.frame_id
        if frame_id[0] == '/':
            frame_id = frame_id[1:]

        # Transform the object centers into the image frame
        object_centers_projected = {}
        try:
            for name, pt in self.object_centers.items():
                pt.header.stamp = header.stamp
                if self.tf_buffer.can_transform(frame_id, pt.header.frame_id, pt.header.stamp, rospy.Duration(0.5)):
                    center_transformed = self.tf_buffer.transform(pt, frame_id, rospy.Duration(0.5)).point
                    x = center_transformed.x
                    y = center_transformed.y
                    z = center_transformed.z
                    center_projected = self.model.project3dToPixel((x, y, z))
                    object_centers_projected[name] = (int(center_projected[0]), int(center_projected[1]))
                else:
                    rospy.logwarn('TF not available between {} and {}'.format(frame_id, pt.header.frame_id))
        except tf2_ros.ExtrapolationException as err:
            rospy.logwarn(err)
            return

        except Exception as err:
            rospy.logerr(err)
            return

        # put labeled circles into the image
        for name, center_projected in object_centers_projected.items():
            cv2.putText(img_debug, self.categories[name], center_projected, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(img_debug, center_projected, 10, (0, 255, 0), -1)


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
                    rospy.logwarn('TF not availble between {} and {}'.format(frame_id, pt.header.frame_id))
                    return
        except tf2_ros.ExtrapolationException as err:
            rospy.logwarn(err)
            return


        # transform the pixel corners into the image frame
        pixel_corners = []
        for j, pt in enumerate(trans_corners):
            pixel_pt = np.array(self.model.project3dToPixel(pt)).astype(int) 
            pixel_corners.append(pixel_pt)
        pixel_corners = np.array(pixel_corners)

        # Create and fill in the board mask
        board_mask = np.zeros_like(img[:, :, 0])
        cv2.fillConvexPoly(board_mask, pixel_corners, 1)

        # Mask the debug image
        img_debug = cv2.bitwise_and(img_debug, img_debug, mask=board_mask)

        # Get the connected component markers and their ids
        board_size = board_mask[:, :].sum()
        markers, marker_idx = self.object_masker.getMasks(img, roi_mask=board_mask)


        mapper = TaggedAnnotationMapper(object_centers_projected)
        annotated_image = self.object_masker.getAnnotations(img,
                                                            markers,
                                                            marker_idx,
                                                            categories=self.categories,
                                                            mapper=mapper)

        mask_image = annotated_image.draw(color_by_category=True)

        try:
            display_msg = self.bridge.cv2_to_imgmsg(mask_image.astype(np.uint8), encoding="bgr8")
        except CvBridgeError as err:
            rospy.logerr(err)
            return

        display_msg.header = img_msg.header
        self.image_pub.publish(display_msg)

        display_msg_debug = self.bridge.cv2_to_imgmsg(img_debug.astype(np.uint8), encoding="bgr8")
        display_msg_debug.header = img_msg.header
        self.image_pub_debug.publish(display_msg_debug)
        self.export_annotated_image(img, mask_image, annotated_image)

    def export_annotated_image(self, image, display_image, annotated_image):
        if not os.path.exists(self.image_folder):
            rospy.loginfo("creating directory:{}".format(self.image_folder))
            os.makedirs(self.image_folder)
        if not os.path.exists(self.annotation_folder):
            rospy.loginfo("creating directory:{}".format(self.annotation_folder))
            os.makedirs(self.annotation_folder)

        image_id = "{:04d}".format(self.image_id)
        image_filename = self.image_stub + image_id + '.' + self.image_ext
        image_path = os.path.join(self.image_folder, image_filename)

        display_filename = self.image_stub + image_id + '.disp.' + self.image_ext
        display_path = os.path.join(self.image_folder, display_filename)

        annotation_filename = self.image_stub + image_id + '.' + self.annotation_ext
        annotation_path = os.path.join(self.annotation_folder, annotation_filename)

        annotated_image.id = image_id
        annotated_image.path = image_path
        annotated_image.file_name = image_filename

        rospy.loginfo('saving {}'.format(image_path))
        cv2.imwrite(image_path, image)
        cv2.imwrite(display_path, display_image)

        annotated_image.save(annotation_path)
        self.image_id += 1

