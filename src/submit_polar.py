#! /usr/bin/env python2

import argparse
import gzip
import os

import cv2
import numpy as np
import scipy.spatial
import pyart

import util

parser = argparse.ArgumentParser(description='Generate submission from predictions in polar coordiantes.')
parser.add_argument('--data-dir', default='../data/test', help='location of test data')
parser.add_argument('--predictions-dir', default='simple_polar', help='location of predictions')
args = parser.parse_args()

file = gzip.open('submission_{}.csv.gz'.format(args.predictions_dir), 'w')
file.write('Id,Prediction\n')
for fname in sorted(os.listdir(args.data_dir)):
    if fname.endswith('.gz'):
        print fname
        radar = pyart.io.read(os.path.join(args.data_dir, fname), scans=[0])
        x, y, _ = radar.get_gate_x_y_z(0)
        prediction_polar = np.loadtxt(os.path.join(args.predictions_dir, fname + '.txt'))
        assert prediction_polar.shape == x.shape
        prediction_cartesian = util.togrid(prediction_polar, x, y)
        mask = cv2.imread(os.path.join(args.data_dir, fname.replace('.gz', '_mask.png')), 0)
        for i, j in zip(*np.nonzero(mask)):
            id = '{}_{}_{}'.format(fname, i, j)
            file.write('{},{}\n'.format(id, prediction_cartesian[i, j]))
