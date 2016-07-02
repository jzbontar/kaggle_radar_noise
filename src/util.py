import numpy as np
import scipy.spatial


# Inspired by wradlib library's togrid function
def togrid(polar, x, y, gridsize=1024, lim=460):
    src = np.column_stack((x.ravel(), y.ravel()))
    grid = np.linspace(-lim * 1000, lim * 1000, gridsize)
    mgrid = np.meshgrid(grid, grid[::-1])
    trg = np.column_stack((mgrid[0].ravel(), mgrid[1].ravel()))
    tree = scipy.spatial.cKDTree(src)
    dists, ix = tree.query(trg, k=1)
    grid = polar.ravel()[ix]
    return grid.reshape(gridsize, gridsize)
