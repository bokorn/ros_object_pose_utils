#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt

import cv2
from collections import namedtuple


def isTopLevel(hier):
    return hier[3] == -1


def filterMarkers(markers, thresh_min, thresh_max):
    """ Filter component indices within a set of bounds

    Args:
        markers: connected components "image" as 2D ndarray
        thresh_min: min number of component counts
        thresh_max: max number of component counts

    Returns:
        Two lists.  First list is the component idx, the second list is the corresponding component count (# of pixels)
    """
    idx, counts = np.unique(markers, return_counts=True)
    idx_out = []
    counts_out = []
    for i, c in zip(idx, counts):
        if thresh_min < c and c < thresh_max:
            idx_out.append(i)
            counts_out.append(c)
    return idx_out, counts_out


def segmentImage(img, block_size=11, C=10):
    """ Segment an image

    Args:
        img: input BGR-format image an an ndarray
        block_size: block size for adaptive thresholding
        C: constant using by adaptive thresholding

    Returns:
        Output ndarray of connected components.  Same width & height as input image.  Cells with the same value
        are part of the same component.
    """
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thr = cv2.adaptiveThreshold(img_bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, C)

    _, markers = cv2.connectedComponents(img_thr)
    return markers


def getCMapImage(image, cmap='gray'):
    img_cmap = plt.imshow(image, cmap=cmap)._rgba_cache
    return img_cmap


class ObjectMasker(object):
    def __init__(self, thresh_block_size, thresh_const, image_roi=None):
        """

        Args:
            thresh_block_size: block size used for adaptive thesholding
            thresh_const: constant used in adaptive thresholding
            image_roi: region of interest in [x1, y1, x2, y2] format
        """
        rospy.init_node("object_masker")

        self.image_roi = None
        self.thresh_block_size = thresh_block_size
        self.thresh_constant = thresh_const

    def setImageRoi(self, x1, y1, x2, y2):
        self.image_roi = [x1, y1, x2, y2]

    def setThreshBlockSize(self, value):
        self.thresh_block_size = value

    def setThreshConst(self, value):
        self.thresh_constant = value

    def getMasks(self, image):
        """ Get objects masks from the image

        Args:
            image: input BGR image as an ndarray

        Returns:
            List of masks, where each mask is an opencv contour
        """
        # create a mask for the ROI
        board_mask = np.zeros_like(image[:, :, 0])

        if self.image_roi is None:
            board_mask[:, :] = 1
        else:
            board_mask[self.image_roi[1]:self.image_roi[3], self.image_roi[0]:self.image_roi[2]] = 1

        board_size = board_mask.sum()

        # Calculate the connected components and mask them with the ROI mask
        all_markers = segmentImage(image, self.thresh_block_size, self.thresh_block_size)
        markers_masked = (all_markers + 1) * board_mask

        # Filter out really small and reall large components (noise and background)
        filtered_idxs, filtereda_counts = filterMarkers(markers_masked, board_size / 500, board_size / 3)

        # Set markers that are outside our filter box to zero
        markers_masked[np.isin(markers_masked, filtered_idxs, invert=True)] = 0

        # Get the contours from the connected components
        img_cnts, contours, hierarchy = cv2.findContours((markers_masked > 0).astype(np.uint8), cv2.RETR_TREE,
                                                         cv2.CHAIN_APPROX_SIMPLE)

        return contours, hierarchy
