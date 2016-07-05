#! /usr/bin/env python2

import argparse
import os
import sys

import cv2
import numpy as np
import scipy.signal
import scipy.ndimage.morphology
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

def spinchange(data):
    spin = np.zeros_like(data)
    for i in range(data.shape[0]):
        signs = np.sign(np.trunc(np.diff(data[i]) / 2))
        nnz = np.flatnonzero(signs)
        spin[i,nnz[1:]] = np.diff(signs[nnz]) != 0
    kernel_h = np.ones((1, 21))
    kernel_v = np.ones((11, 1))
    spin = scipy.signal.convolve(spin, kernel_h, 'same')
    spin = scipy.signal.convolve(spin, kernel_v, 'same')
    possible = np.ones_like(spin)
    possible[:,[0,-1]] = 0
    possible = scipy.signal.convolve(possible, kernel_h, 'same')
    possible = scipy.signal.convolve(possible, kernel_v, 'same')
    return spin / possible

def steiner_smith(radar, refl_thresh=5, spin_thresh_a=8, spin_thresh_b=40, spin_thresh_c=15, grad_thresh=20):
    data0 = radar[0].get_field(0, 'reflectivity')
    x0, y0, _ = radar[0].get_gate_x_y_z(0)
    data2 = radar[2].get_field(0, 'reflectivity')
    x2, y2, _ = radar[2].get_gate_x_y_z(0)
    src = np.column_stack((x2.ravel(), y2.ravel()))
    trg = np.column_stack((x0.ravel(), y0.ravel()))
    data2 = util.ipol_nearest(src, trg, data2.ravel()).reshape(data0.shape)

    spin_thresh = (spin_thresh_a - (data0.filled(0) - spin_thresh_b) / spin_thresh_c) * 0.01
    zpixel = data0.filled(0) >= refl_thresh
    echotop = data2.filled(0) >= refl_thresh
    echotop = scipy.ndimage.morphology.binary_dilation(echotop)
    spin = spinchange(data0.filled(0)) >= spin_thresh
    vgrad = np.abs(data0 - data2).filled(0) > grad_thresh

    r1 = ~zpixel
    r2 =  zpixel & ~echotop
    r3 =  zpixel &  echotop & ~spin
    r4 =  zpixel &  echotop &  spin & ~vgrad
    r5 =  zpixel &  echotop &  spin &  vgrad

    return r1 | r2 | r5
