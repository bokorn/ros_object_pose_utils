import rospy
from object_pose_utils.object_masker_node import ObjectMaskerNode


def main():
    rospy.init_node("object_masker")
    obj_masker = ObjectMaskerNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down pose_labeler module")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
