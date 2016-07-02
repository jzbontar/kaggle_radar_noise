#! /usr/bin/env python2

import argparse
import os

import cv2
import numpy as np
import pyart

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../data/test', help='location of test data')
parser.add_argument('--predictions-dir', default='simple_polar', help='location of predictions')
args = parser.parse_args()

try:
    os.mkdir(args.predictions_dir)
except OSError:
    pass

for fname in sorted(os.listdir(args.data_dir)):
    if fname.endswith('.gz'):
        print fname
        radar = pyart.io.read(os.path.join(args.data_dir, fname), scans=[0])
        ref = radar.get_field(0, 'reflectivity')
        prediction_polar = ref.filled(0)
        np.savetxt(os.path.join(args.predictions_dir, fname + '.txt'), prediction_polar, fmt='%.1f')
