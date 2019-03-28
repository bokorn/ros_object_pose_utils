import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imantics

import cv2
from functools import partial

class AnnotationMapper(object):
    """
    Class that will return a mapping of marker ids to annotation ids

    """
    def sort(self, image, masks, categories, annotations):
        """

        Args:
            image: input image as a BGR ndarray
            masks: connected component mask
            categories: names of the categories as a list of strings
            annotations: dictionary of mask_ids to imantics.Annotation objects

        Returns:
            Maping of mask_idx to annotation_idx as a python dictionary
        """
        raise NotImplementedError("This method must be implemented")


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

def getCMapImage(image, cmap='gray'):
    img_cmap = plt.imshow(image, cmap=cmap)._rgba_cache
    return img_cmap


class CenterSortMapper(AnnotationMapper):
    """
    This mapper will label the i
    """

    def sort(self, image, masks, categories, annotations):

        num_annotations = len(annotations)
        num_categories = len(categories)

        if num_categories < num_annotations:
            annotations_prunned = {}

            mask_ids, annotations = annotations.items()

            areas = [annotation.area for annotation in annotations]
            # take the top num_categories areas
            top_idx = np.argsort(areas)[-num_categories:]

            for idx in top_idx:
                annotations_prunned[mask_ids[idx]] = annotations[idx]

            annotations = annotations_prunned

        centers = []
        keys = []
        for mask_id, annotation in annotations.items():
            centers.append(0.5 * (annotation.bbox.min_point[0] + annotation.bbox.max_point[0]))

        return dict(zip(np.array(annotations.keys())[np.argsort(centers)], range(len(annotations))))


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

    def __init__(self, thresh_block_size, thresh_const, image_roi=None, debug=False):
        """

        Args:
            thresh_block_size: block size used for adaptive thesholding
            thresh_const: constant used in adaptive thresholding
            image_roi: region of interest in [x1, y1, x2, y2] format
        """
        self.image_roi = image_roi
        self.thresh_block_size = thresh_block_size
        self.thresh_constant = thresh_const
        self.debug = debug
        self.morph_kernel_size = 0
        self.min_cluster_width = 1
        self.max_cluster_width = 10000
        self.min_cluster_height = 1
        self.max_cluster_height = 10000
        self.min_cluster_area = self.min_cluster_height * self.min_cluster_width
        self.max_cluster_area = self.max_cluster_height * self.max_cluster_width

    def setImageRoi(self, x1, y1, x2, y2):
        self.image_roi = [x1, y1, x2, y2]

    def setThreshBlockSize(self, value):
        self.thresh_block_size = value

    def setThreshConst(self, value):
        self.thresh_constant = value

    def setMorphKernelSize(self, value):
        self.morph_kernel_size = value

    def setVisualizationDebug(self, value):
        self.debug = value

    def setClusterWidth(self, min_val, max_val):
        self.min_cluster_width = min_val
        self.max_cluster_width = max_val

    def setClusterHeight(self, min_val, max_val):
        self.min_cluster_height = min_val
        self.max_cluster_height = max_val

    def setClusterArea(self, min_val, max_val):
        self.min_cluster_area = min_val
        self.max_cluster_area = max_val


    def getMasks(self, image, roi_mask = None):
        """ Get objects masks from the image

        The output masks are stored in a single "image" as an ndarray of connected components.  It has the same width &
        height as the input image where the value of each "pixel" corresponds to a connected component.  Pixels with the
        same value are considered to be in the same component.

        Args:
            image: input BGR image as an ndarray
            roi_mask: 2D binary mask as an ndarray

        Returns:
            Tuple of a mask "image" and list of corresponding filter IDs.
        """
        # create a mask for the ROI
        if roi_mask is None:
            roi_mask = np.zeros_like(image[:, :, 0])

            if self.image_roi is None:
                roi_mask[:, :] = 1
            else:
                roi_mask[self.image_roi[1]:self.image_roi[3], self.image_roi[0]:self.image_roi[2]] = 1

        roi_size = roi_mask.sum()

        # Calculate the connected components and mask them with the ROI mask
        all_markers = self.segmentImage(image, self.thresh_block_size, self.thresh_constant)
        markers_masked = (all_markers + 1) * roi_mask

        # Filter out really small and really large components (noise and background)
        filtered_idxs, filtered_counts = filterMarkers(markers_masked, roi_size / 250, roi_size / 3)

        # Set markers that are outside our filter box to zero
        markers_masked[np.isin(markers_masked, filtered_idxs, invert=True)] = 0
        
        return markers_masked, filtered_idxs

    def getAnnotations(self, image, masks, mask_ids=None, categories=None, colors=None, mapper=CenterSortMapper()):
        """ Get annotations from the image

        Args:
            image: input BGR image as and ndarray
            masks: connected component "mask" image from getMasks as an ndarray
            mask_ids: list of component values
            categories: dictionary of category_id to string
            category_colors: dictionary of category_id to tuple of BGR values
                ex: { 0: (255, 0, 0),
                      1: (0, 255, 0),
                      2: (0, 0, 255)}
            mapper: Instance of an AnnotationMapper

        Returns:
            Annotated image as an imantics.Image
        """

        ann_img = imantics.Image(image_array=image)
         
        if mask_ids is None:
            mask_ids = np.unique(masks)
        
        if categories is None:
            category_names = ['obj_{}'.format(j) for j in range(len(mask_ids))]
            categories = dict(zip(range(len(mask_ids)), category_names))

        if colors is None:
            category_colors = (np.array(sns.color_palette(n_colors=(len(categories)))) * 255).astype(np.uint8).tolist()
            colors = dict(zip(sorted(categories.keys()), category_colors))

        annotations = {}
        for m_id in mask_ids:
            mask = imantics.Mask((masks == m_id).astype(np.uint8))
            bbox = imantics.BBox.from_mask(mask)
            annotation = imantics.Annotation(bbox=bbox, mask=mask)
            annotations[m_id] = annotation

        category_map = mapper.sort(image, masks, categories, annotations)

        for mask_id, category_id in category_map.items():
            try:
                category = imantics.Category(categories[category_id],
                                             color=(colors[category_id]),
                                             id=category_id)
                ann = imantics.Annotation(image=ann_img, category=category,
                                          mask=annotations[mask_id].mask, bbox=annotations[mask_id].bbox)
                ann_img.add(ann)
            except KeyError:
                pass

        return ann_img, category_map
 
    def segmentImage(self, img, block_size=11, C=10, visualize=False):
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

        if self.morph_kernel_size > 0:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size),np.uint8)
            img_thr = cv2.morphologyEx(img_thr, cv2.MORPH_CLOSE, kernel)

        num_labels, markers, stats, centroids = cv2.connectedComponentsWithStats(img_thr)

        for label in range(num_labels):
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            if width < self.min_cluster_width or width > self.max_cluster_width:
                markers[markers == label] = 0
            elif height < self.min_cluster_height or height > self.max_cluster_height:
                markers[markers == label] = 0
            elif area < self.min_cluster_area or area > self.max_cluster_area:
                markers[markers == label] = 0

        if self.debug:
            im = cv2.applyColorMap(markers.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("segment image", im)
            cv2.waitKey(3)
        return markers
