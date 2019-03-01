import unittest
import rospy
import os
import cv2
import yaml
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from object_pose_utils.tagged_object_masker import TaggedObjectMasker
import tf
import time
import threading


TEST_FOLDER = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_PATH = os.path.join(TEST_FOLDER, "images", "image_0585.png")
TEST_CAMERA_INFO_PATH = os.path.join(TEST_FOLDER, "config", "test_camera_info.yaml")
TEST_CONFIG_PATH = os.path.join(TEST_FOLDER, "config", "test_config.xml")


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
    y_scale = float(new_size[0])/camera_info.height
    x_scale = float(new_size[1])/camera_info.width
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


def create_image_msg(image_path, camera_info, frame_id, t):
    bridge = CvBridge()
    img = cv2.imread(image_path)
    if (img.shape[:2] != (camera_info.height, camera_info.width)):
        resizeCameraInfo(camera_info, img.shape[:2])

    img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")

    timestamp = t
    camera_info.header.stamp = timestamp
    img_msg.header.stamp = timestamp
    camera_info.header.frame_id = frame_id
    img_msg.header.frame_id = frame_id
    return img_msg, camera_info


class TF_Thread(threading.Thread):

    def __init__(self, translation, rotation, t, child_frame, parent_frame):
        threading.Thread.__init__(self)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.translation = translation
        self.rotation = rotation
        self.t = t
        self.child_frame = child_frame
        self.parent_frame = parent_frame
        self.running = False

    def run(self):
        self.running = True

        while self.running :
            self.tf_broadcaster.sendTransform(self.translation,
                                              self.rotation,
                                              self.t,
                                              self.child_frame,
                                              self.parent_frame)
            rospy.sleep(rospy.Duration(0.10))

    def stop(self):
        self.running = False


class TaggedObjectMasterTest(unittest.TestCase):

    def setUp(self):
        rospy.init_node("tagged_object_masker_test")
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.camera_info_msg = parseCameraInfoYaml(TEST_CAMERA_INFO_PATH)
        self.bridge = CvBridge()

    def tearDown(self):
        pass

    def test_imageCallback(self):

        camera_frame = "camera"
        board_frame = "test_board_frame"

        rospy.set_param('~config_file', TEST_CONFIG_PATH)
        rospy.set_param('~board_frame', board_frame)
        rospy.set_param('~thresh_block_size', 31)
        rospy.set_param('~thresh_const', 10)

        #image #585
        translation = [0.349684424343, 0.0956428247035, 0.548789059364]
        rotation = [-0.461545442226, 0.881830968253, 0.095355690736, -0.0160387167094]

        #rotation = tf.transformations.quaternion_from_euler(0, 0, 0)
        t = rospy.Time.now()
        child_frame  = board_frame
        parent_frame = camera_frame

        tf_thread = TF_Thread(translation, rotation, t, child_frame, parent_frame)
        tf_thread.start()
        time.sleep(3)

        taggedObjectMasker = TaggedObjectMasker()

        img_msg, camera_info_msg = create_image_msg(TEST_IMAGE_PATH, self.camera_info_msg, camera_frame, t)
        taggedObjectMasker.imageCallback(img_msg, camera_info_msg)

        # tf_thread.stop()
        # tf_thread.join(5)



if __name__ == "__main__":
    unittest.main()
