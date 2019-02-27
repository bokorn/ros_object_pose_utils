#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imantics

import cv2
from collections import namedtuple
from functools import partial


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


def centerSortX(image, anns, num_categories = np.inf):
    """

    Args:
        image: input image
        anns: dictionary of mask index in the masked image to a dictionary of imantics.bbox and imantics.mask
            ann : {
                int : {
                    "bbox" :
                    "mask":
                    },
                int : {
                    "bbox" :
                    "mask":
                    },
                ...
        num_categories:

    Returns:
        Dictionary of mask index(int) to category(int)
    """
    if(num_categories < len(anns)):
        anns_prunned = {}
        areas = [v['bbox'].area for _,v in anns.items()]
        top_idxs = np.argsort(areas)[-num_categories:]
        for j, (k, v) in enumerate(anns.items()):
            if(j in top_idxs):
                anns_prunned[k] = v
        anns = anns_prunned

    centers = []
    keys = []
    for k,v in anns.items():
        centers.append(0.5*(v['bbox'].min_point[0] + v['bbox'].max_point[0]))
 
    return dict(zip(np.array(anns.keys())[np.argsort(centers)], range(len(anns))))


class ObjectMasker(object):
    def __init__(self, thresh_block_size, thresh_const, image_roi=None):
        """

        Args:
            thresh_block_size: block size used for adaptive thesholding
            thresh_const: constant used in adaptive thresholding
            image_roi: region of interest in [x1, y1, x2, y2] format
        """
        self.image_roi = image_roi
        self.thresh_block_size = thresh_block_size
        self.thresh_constant = thresh_const

    def setImageRoi(self, x1, y1, x2, y2):
        self.image_roi = [x1, y1, x2, y2]

    def setThreshBlockSize(self, value):
        self.thresh_block_size = value

    def setThreshConst(self, value):
        self.thresh_constant = value

    def getMasks(self, image, roi_mask = None):
        """ Get objects masks from the image

        The output masks are stored in a single "image" as an ndarray of connected components.  It has the same width &
        height as the input image where the value of each "pixel" corresponds to a connected component.  Pixels with the
        same value are considered to be in the same component.

        Args:
            image: input BGR image as an ndarray

        Returns:
            Tuple of a mask "image" and list of corresponding filter IDs.
        """
        # create a mask for the ROI
        if(roi_mask is None):
            roi_mask = np.zeros_like(image[:, :, 0])

            if self.image_roi is None:
                roi_mask[:, :] = 1
            else:
                roi_mask[self.image_roi[1]:self.image_roi[3], self.image_roi[0]:self.image_roi[2]] = 1

        roi_size = roi_mask.sum()

        # Calculate the connected components and mask them with the ROI mask
        all_markers = segmentImage(image, self.thresh_block_size, self.thresh_constant)
        markers_masked = (all_markers + 1) * roi_mask

        # Filter out really small and really large components (noise and background)
        filtered_idxs, filtered_counts = filterMarkers(markers_masked, roi_size / 250, roi_size / 3)

        # Set markers that are outside our filter box to zero
        markers_masked[np.isin(markers_masked, filtered_idxs, invert=True)] = 0
        
        return markers_masked, filtered_idxs

    def getAnnotations(self, image, masks, mask_ids = None, 
                       category_names = None, category_colors = None, 
                       category_func = None):
        """

        Args:
            image: input BGR image as and ndarray
            masks: connected component "mask" image from getMasks as an ndarray
            mask_ids: list of component values
            category_names: list of strings
            category_colors:
            category_func:

        Returns:

        """

        ann_img = imantics.Image(image_array = image)
         
        if(mask_ids is None):
            mask_ids = np.unique(masks)
        
        if(category_names is None):
            category_names = ['obj_{}'.format(j) for j in range(len(mask_ids))]

        if(category_colors is None):
            if(len(mask_ids) < 8):
                category_colors = 255*np.array(sns.color_palette())
            else:
                category_colors = 255*np.array(sns.color_palette("hls", len(mask_ids)))
        
        _, unique_idx, unique_inv = np.unique(category_names, return_index=True, return_inverse=True)
        for j, u_id in enumerate(unique_inv):
            category_colors[j] = category_colors[unique_idx[u_id]]

        if(category_func is None):
            category_func = partial(centerSortX, num_categories = len(category_names))
        
        anns = {}
        for m_id in mask_ids:
            mask = imantics.Mask((masks == m_id).astype(np.uint8))
            bbox = imantics.BBox.from_mask(mask) 
            anns[m_id] = {'mask':mask, 'bbox':bbox}
        
        cat_ids = category_func(image, anns)
        
        for m_id, c_id in cat_ids.items():
            category = imantics.Category(category_names[c_id],
                                         color=imantics.Color(rgb=tuple(category_colors[c_id])))
            ann = imantics.Annotation(image = ann_img, category = category, 
                                      mask=anns[m_id]['mask'], bbox=anns[m_id]['bbox'])
            ann_img.add(ann)

        return ann_img
 
