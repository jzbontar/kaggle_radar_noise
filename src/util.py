import numpy as np
import scipy.spatial

# Inspired by wradlib library's ipol and togrid functions.
def ipol_nearest(src, trg, data):
    tree = scipy.spatial.cKDTree(src)
    dists, ix = tree.query(trg, k=1)
    return data[ix]

def togrid(polar, x, y, gridsize=1024, lim=460):
    src = np.column_stack((x.ravel(), y.ravel()))
    grid = np.linspace(-lim * 1000, lim * 1000, gridsize)
    mgrid = np.meshgrid(grid, grid[::-1])
    trg = np.column_stack((mgrid[0].ravel(), mgrid[1].ravel()))
    grid = ipol_nearest(src, trg, polar.ravel())
    return grid.reshape(gridsize, gridsize)
