import rospy
from sensor_msgs.msg import Image
import message_filters
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import message_filters


class MaskImageNode(object):

    def __init__(self, image_topic, mask_topic, output="/output", debug=False):
        self.bridge = CvBridge()

        self.debug = debug
        self.input_image_sub = message_filters.Subscriber(image_topic, Image)
        self.input_mask_sub = message_filters.Subscriber(mask_topic, Image)
        self.publisher = rospy.Publisher(output, Image, queue_size=1)

        self.synchronizer = message_filters.ApproximateTimeSynchronizer([self.input_image_sub, self.input_mask_sub],
                                                                        queue_size=20, slop=0.50)
        self.synchronizer.registerCallback(self.image_cb)

    def image_cb(self, image_msg, mask_msg):
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, image_msg.encoding)
            mask = self.bridge.imgmsg_to_cv2(mask_msg, mask_msg.encoding)
            header = image_msg.header
        except CvBridgeError as err:
            rospy.logerr(err)
            return

        output = cv2.bitwise_and(image, image, mask=cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX))
        if self.debug:
            cv2.imshow("image", image)
            cv2.imshow("mask", mask)
            cv2.imshow("output", output * 255)
            cv2.waitKey(50)

        output_msg = self.bridge.cv2_to_imgmsg(output, image_msg.encoding)
        output_msg.header = image_msg.header
        self.publisher.publish(output_msg)
