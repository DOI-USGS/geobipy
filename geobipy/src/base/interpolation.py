import subprocess

from numpy import argwhere, ceil, column_stack, diff, floor, hstack, inf, int32, linspace, meshgrid
from numpy import minimum, maximum, nan, nanmax, nanmin, tile, zeros

from .fileIO import deleteFile
from ..classes.statistics import StatArray
from . import utilities as cf
from scipy import interpolate
from scipy.interpolate import CloughTocher2DInterpolator
#from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
try:
    from netCDF4 import Dataset
except:
    Warning('For minimum curvature plotting, the netcdf4 package must be installed')
    pass

from numba_kdtree import KDTree
from numba import jit
_numba_settings = {'nopython': True, 'nogil': False, 'fastmath': True, 'cache': False}

def sibson(x, y, values, grid_x, grid_y, z=None, grid_z = None, max_distance=inf):

    if grid_z is None:
        return __sibson_2d(x, y, values, grid_x, grid_y, max_distance)
    else:
        return __sibson_3d(x, y, z, values, grid_x, grid_y, grid_z)

def __sibson_2d(x, y, values, grid_x, grid_y, max_distance=inf):

    import numpy as np
    points = np.vstack([x, y]).T

    # Integerize the points to the grid.
    dx = (grid_x[1] - grid_x[0])
    points[:, 0] -= grid_x[0]
    points[:, 0] = points[:, 0] / dx

    dy = (grid_y[1] - grid_y[0])
    points[:, 1] -= grid_y[0]
    points[:, 1] = points[:, 1] / dy

    if not max_distance:
        max_distance = inf

    max_distance = max_distance / (dx*dy)

    # KdTree the points
    kdtree = KDTree(points, leafsize=16)

    out = __sibson_2d_inner(kdtree, values, grid_x, grid_y, max_distance)

    return out

@jit(**_numba_settings)
def __sibson_2d_inner(kdtree, values, grid_x, grid_y, max_distance=inf):
    nx = grid_x.size - 1
    ny = grid_y.size - 1

    c = zeros((ny, nx))
    n = zeros((ny, nx), dtype=int32)

    # For each raster pixel
    for i in range(ny):
        for j in range(nx):
            # Find the nearest point to the pixel
            distance, index, _ = kdtree.query([j, i], k=1)
            distance = ceil(distance)

            distance = int32(distance.item())
            index = index.item()

            d2 = (distance**2.0) + 0.25

            # if d2 < max_distance:
            for i_s in range(maximum(0, i-distance), minimum(ny, i+distance)):
                in_i = (i_s-i)**2.0
                for j_s in range(maximum(0, j-distance), minimum(nx, j+distance)):
                    in_j = (j_s-j)**2.0
                    if (in_i + in_j) <= d2:
                        c[i_s, j_s] += values[index]
                        n[i_s, j_s] += 1

            if d2 > max_distance:
                c[i, j] = nan

    return c / n


def __sibson_3d(x, y, z, values, grid_x, grid_y, grid_z, plot=False, max_distance=inf):

    points = hstack([x, y, z])

    # Integerize the points to the grid.
    dx = (grid_x[1] - grid_x[0])
    points[:, 0] -= grid_x[0]
    points[:, 0] = points[:, 0] / dx

    dy = (grid_y[1] - grid_y[0])
    points[:, 1] -= grid_y[0]
    points[:, 1] = points[:, 1] / dy

    dz = (grid_z[1] - grid_z[0])
    points[:, 2] -= grid_z[0]
    points[:, 2] = points[:, 2] / dz

    max_distance = max_distance / (dx*dy*dz)

    # KdTree the points
    kdtree = KDTree(points, leafsize=16)

    return __sibson_3d_inner(kdtree, values, grid_x, grid_y, grid_z, max_distance)


@jit(**_numba_settings)
def __sibson_3d_inner(kdtree, values, grid_x, grid_y, grid_z, max_distance=inf):

    nx = grid_x.size
    ny = grid_y.size
    nz = grid_z.size

    c = zeros((nz, ny, nx))
    n = zeros((nz, ny, nx), dtype=int32)

    # For each raster pixel
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                # Find the nearest point to the pixel
                distance, index, _ = kdtree.query([k, j, i], k=1)
                distance = ceil(distance)

                distance = int32(distance.item())
                index = index.item()

                d2 = (distance**2.0) + 0.125

                for i_s in range(maximum(0, i-distance), minimum(nz, i+distance)):
                    in_i = (i_s-i)**2.0
                    for j_s in range(maximum(0, j-distance), minimum(ny, j+distance)):
                        in_j = (j_s-j)**2.0
                        tmp = in_i + in_j
                        for k_s in range(maximum(0, k-distance), minimum(nx, k+distance)):
                            in_k = (k_s-k)**2.0
                            if (tmp + in_k) <= d2:
                                c[i_s, j_s, k_s] += values[index]
                                n[i_s, j_s, k_s] += 1

                if d2 > max_distance:
                    c[i, j, k] = nan

    return c / n