#!/usr/bin/env python

import os
import numpy as np
import cv2
import glob
import json
import imantics

from object_pose_utils.object_masker import ObjectMasker

def annotateImageFolder(image_path, annotation_path, category_names, image_ext = 'jpg',
                        image_roi = None, filter_size = 41, filter_const = 20,
                        category_func = None, 
                        save_disp = False, save_indv = False): 
    dataset = imantics.Dataset('surgical_tools')
    masker = ObjectMasker(filter_size, filter_const, image_roi)
    filenames = glob.glob(image_path + '*.' + image_ext) 
    img_id = 0
    for fn in filenames:
        if('.disp.' in fn):
            continue
        img = cv2.imread(fn)
        masks, mask_idxs = masker.getMasks(img)
        ann_img = masker.getAnnotations(img, masks, mask_idxs, category_names = category_names)
        ann_img.id = img_id
        ann_img.path = fn
        ann_img.file_name = os.path.basename(fn)
        if(save_indv):
            ann_img.save('.'.join(fn.split('.')[:-1] + ['json',]))
        
        dataset.add(ann_img)
        if(save_disp):
            display_img = ann_img.draw(thickness=1, color_by_category=True)
            cv2.imwrite('.'.join(fn.split('.')[:-1] + ['disp.' + image_ext,]), display_img)
        img_id += 1
    if(not save_indv):
        with open(annotation_path, 'w') as f:                   
            json.dump(dataset.coco(), f)

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('image_folder', type=str)
    parser.add_argument('--category_names', type=str, nargs='+')
    parser.add_argument('--annotation_path', type=str, default=None)
    parser.add_argument('--image_ext', type=str, default='jpg')

    parser.add_argument('--image_roi', type=int, nargs=4, default=None)
    parser.add_argument('--filter_size', type=int, default=41)
    parser.add_argument('--filter_const', type=int, default=20)
    parser.add_argument('--save_display', dest='save_display', action='store_true') 
    parser.add_argument('--save_individual', dest='save_individual', action='store_true') 
    
    args = parser.parse_args()
    if(args.annotation_path is None):
        args.annotation_path = os.path.join(args.image_folder, '../annotations/coco.json')

    annotateImageFolder(args.image_folder, 
                        args.annotation_path, 
                        args.category_names,
                        image_ext = args.image_ext,
                        image_roi = args.image_roi, 
                        filter_size = args.filter_size, 
                        filter_const = args.filter_const, 
                        save_disp = args.save_display,
                        save_indv = args.save_individual)

if __name__=='__main__':
    main()
