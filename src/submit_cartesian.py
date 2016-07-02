#! /usr/bin/env python2

import argparse
import gzip
import os

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../data/test', help='location of test data')
parser.add_argument('--predictions-dir', default='simple_cartesian', help='location of predictions')
args = parser.parse_args()

file = gzip.open('submission_{}.csv.gz'.format(args.predictions_dir), 'w')
file.write('Id,Prediction\n')
for fname in sorted(os.listdir(args.data_dir)):
    if fname.endswith('.gz'):
        print fname
        prediction_cartesian = np.loadtxt(os.path.join(args.predictions_dir, fname + '.txt'))
        assert prediction_cartesian.shape == (1024, 1024)
        mask = cv2.imread(os.path.join(args.data_dir, fname.replace('.gz', '_mask.png')), 0)
        for i, j in zip(*np.nonzero(mask)):
            id = '{}_{}_{}'.format(fname, i, j)
            file.write('{},{}\n'.format(id, prediction_cartesian[i, j]))
