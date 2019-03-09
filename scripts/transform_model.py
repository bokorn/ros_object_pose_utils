# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018
@author: bokorn
"""

import cv2
import os
import time
import numpy as np
import quat_math as qm

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input_filename', type=str)
    parser.add_argument('output_filename', type=str)
    parser.add_argument('--scale', nargs='+', type=float, default=None)
    parser.add_argument('--quat', nargs=4, type=float, default=None)
    parser.add_argument('--trans', nargs=3, type=float, default=None)
    parser.add_argument('--mtl_Ka', nargs=3, type=float, default=None)
    parser.add_argument('--mtl_Kd', nargs=3, type=float, default=[0,0,0])
    parser.add_argument('--mtl_Ks', nargs=3, type=float, default=[0.25,0.25,0.25])
    parser.add_argument('--mtl_Ke', nargs=3, type=float, default=[0,0,0])
    parser.add_argument('--mtl_Ns', type=float, default=18)
    parser.add_argument('--mtl_Ni', type=float, default=1)
    parser.add_argument('--mtl_d', type=float, default=1)
    parser.add_argument('--mtl_illum', type=float, default=2)

    args = parser.parse_args()
    
    transform_verts = not (args.scale is None and args.quat is None and args.trans is None) 

    if(args.scale is not None):
        s = np.array(args.scale)
    else:
        s = 1.0  
    if(args.quat is not None):
        R = qm.quaternion_matrix(args.quat)[:3,:3]
    else:
        R = np.eye(3)
    if(args.trans is not None):
        t = np.array(args.trans)
    else:
        t = np.zeros(3)

    if(args.mtl_Ka is not None):
        with open(args.output_filename[:-3]+'mtl', 'w') as f:
            f.write('newmtl obj_mtl\n')
            f.write('Ns {}\n'.format(args.mtl_Ns))
            f.write('Ka {} {} {}\n'.format(*args.mtl_Ka))
            f.write('Kd {} {} {}\n'.format(*args.mtl_Kd))
            f.write('Ks {} {} {}\n'.format(*args.mtl_Ks))
            f.write('Ke {} {} {}\n'.format(*args.mtl_Ke))
            f.write('Ni {}\n'.format(args.mtl_Ni))
            f.write('d {}\n'.format(args.mtl_d))
            f.write('illum {}\n'.format(args.mtl_illum))

        set_mtl = True
    else:
        set_mtl = False

    with open(args.input_filename, 'r') as in_file, open(args.output_filename, 'w') as out_file:
        for line in in_file:
            if(line[:2] == 'v ' and transform_verts):
                verts = np.array(line.split()[1:], dtype=float)
                verts = R.dot(s*verts) + t
                out_file.write('v {} {} {} \n'.format(*verts))
            elif(set_mtl and line[:2] == 'f '):
                out_file.write('usemtl obj_mtl\n')
                out_file.write(line)
                set_mtl = False
            else:
                out_file.write(line)
        

