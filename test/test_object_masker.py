import unittest
import os
import cv2

from pprint import pprint
from object_pose_utils.object_masker import ObjectMasker

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class ObjectMaskerTest(unittest.TestCase):

    def setUp(self):
        print TEST_DIR
        pass

    def tearDown(self):
        pass

    def test_createCocoData(self):
        image_path = os.path.join(TEST_DIR, "images", "image_0001.png")
        image = cv2.imread(image_path)

        masker = ObjectMasker(31, 5, image_roi=[228, 226, 921, 537])

        categories = {0: "scalpel", 1: "scissors", 2: "hemostat"}

        markers, marker_ids = masker.getMasks(image)
        annotated_image = masker.getAnnotations(image, markers, marker_ids, categories=categories)
        mask_image = annotated_image.draw(color_by_category=True)

        coco_data = annotated_image.coco(True)
        coco_data["images"][0]["path"] = image_path

        pprint(coco_data, indent=4, depth=3)

        cv2.imshow("masked_image", mask_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    unittest.main()