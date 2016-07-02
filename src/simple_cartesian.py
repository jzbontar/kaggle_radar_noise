#! /usr/bin/env python2

import argparse
import os

import cv2
import numpy as np
import pyart

import util

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../data/test', help='location of test data')
parser.add_argument('--predictions-dir', default='simple_cartesian', help='output destination')
args = parser.parse_args()

try:
    os.mkdir(args.predictions_dir)
except OSError:
    pass

for fname in sorted(os.listdir(args.data_dir)):
    if fname.endswith('.gz'):
        print fname
        radar = pyart.io.read(os.path.join(args.data_dir, fname), scans=[0])
        ref_polar = radar.get_field(0, 'reflectivity')
        x, y, _ = radar.get_gate_x_y_z(0)
        ref_cartesian256 = util.togrid(ref_polar, x, y, gridsize=256)
        prediction_cartesian256 = ref_cartesian256.filled(0)
        prediction_cartesian1024 = cv2.resize(prediction_cartesian256, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        np.savetxt(os.path.join(args.predictions_dir, fname + '.txt'), prediction_cartesian1024, fmt='%.1f')
