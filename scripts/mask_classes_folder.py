#!/usr/bin/env python

import os
import numpy as np
import cv2
import glob

from functools import partial

from object_pose_utils.object_masker import ObjectMasker
from object_pose_utils.feature_classification import FeatureClassifier, classificationFunction, cropBBox

def maskImageFolder(input_folder, visual_dict_path, 
                    output_folder = None, image_ext = 'jpg',
                    image_roi = None, filter_size = 41, filter_const = 30,
                    save_disp = False): 

    masker = ObjectMasker(filter_size, filter_const, image_roi)
    classifier = FeatureClassifier(visual_dict_path)
    category_func = partial(classificationFunction, classifier=classifier)
    category_names = [v for _, v in sorted(classifier.class_names.items())] 
       
    if(output_folder is not None):
        for cat_name in category_names:
            if not os.path.exists(os.path.join(output_folder, cat_name)):
                os.makedirs(os.path.join(output_folder, cat_name))

    filenames = glob.glob(input_folder + '*.' + image_ext) 
       
    for fn in filenames:
        if('.disp.' in fn):
            continue
        img = cv2.imread(fn)
        masks, mask_idxs = masker.getMasks(img)
        ann_img = masker.getAnnotations(img, masks, mask_idxs, 
                                        category_names = category_names,
                                        category_func = category_func)
        
        if(save_disp):
            display_img = ann_img.draw(thickness=1, color_by_category=True)
            cv2.imwrite('.'.join(fn.split('.')[:-1] + ['disp.' + image_ext,]), display_img)
        
        if(output_folder is not None):
            base_name = os.path.splitext(os.path.basename(fn))[0]
            obj_idxs = {k:0 for k in classifier.class_names.keys()} 
            for ann in ann_img.annotations.values():
                img_crop = cropBBox(img, ann.mask.bbox())
                mask_crop = cropBBox(ann.mask.array, ann.mask.bbox())
                obj_img = np.concatenate([img_crop, np.expand_dims(mask_crop,2)*255], axis=2)
                cat_id = ann.category.id
                cls_name = classifier.class_names[cat_id]
                cv2.imwrite(os.path.join(output_folder, 
                    cls_name, 
                    "{}_{}_{}.{}".format(base_name, cls_name, obj_idxs[cat_id], image_ext)), 
                    obj_img)
                obj_idxs[cat_id] += 1

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input_folder', type=str)
    parser.add_argument('visual_dict_path', type=str)
    parser.add_argument('--output_folder', type=str, default = None)
    parser.add_argument('--image_ext', type=str, default='jpg')

    parser.add_argument('--image_roi', type=int, nargs=4, default=None)
    parser.add_argument('--filter_size', type=int, default=41)
    parser.add_argument('--filter_const', type=int, default=30)
    parser.add_argument('--save_display', dest='save_display', action='store_true') 
    
    args = parser.parse_args()

    maskImageFolder(args.input_folder, 
                    visual_dict_path = args.visual_dict_path, 
                    output_folder = args.output_folder, 
                    image_ext = args.image_ext,
                    image_roi = args.image_roi, 
                    filter_size = args.filter_size, 
                    filter_const = args.filter_const, 
                    save_disp = args.save_display,
                    )
if __name__=='__main__':
    main()
