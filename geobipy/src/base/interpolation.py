import subprocess

from numpy import argwhere, column_stack, diff, floor, inf, linspace, meshgrid
from numpy import nan, nanmax, nanmin, tile, zeros

from .fileIO import deleteFile
from ..classes.core import StatArray
from . import utilities as cf
from scipy import interpolate
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
#from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
try:
    from netCDF4 import Dataset
except:
    Warning('For minimum curvature plotting, the netcdf4 package must be installed')
    pass

# def CT(XY, values, x_grid=None, y_grid=None, dx=None, dy=None,
#        bounds=None, mask = False, kdtree = None, clip = False, extrapolate=None):
#     """Use Scipy's CloughTocher C1 continuous interpolation using unstructured meshes to interpolate arbitrary locations to a grid

#     Parameters
#     ----------
#     XY : 2D ndarray of floats
#         Two columns, each column contains the co-ordinate in a dimension
#     values : ndarray
#         The values to interpolate to the grid
#     x_grid : array_like, optional
#         Grid node locations in x, regularly spaced.
#     y_grid : array_like, optional
#         Grid node locations in y, regularly spaced.
#     dx : float, optional
#         The required spacing in x between grid nodes
#     dy : float, optional
#         The required spacing in y between grid nodes
#     bounds : array_like, optional
#         Length 4 array with the minimum and maximum in two directions. [Xmin, Xmax, Ymin, Ymax]
#     mask : float, optional
#         Force interpolated values that are greater than a distance of mask from any known point to be NaN
#     kdtree : scipy.spatial.ckdtree.cKDTree, optional
#         If no kdtree is given for the set of points, one is created.  To speed up multiple interpolations, the user can pass their own fixed kdtree and prevent the generation of one every time.
#     clip : bool, optional
#         Interpolation can overshoot the known value. clip = True ensures that the min  max of the grid is the same as the known data points.
#     extrapolate : bool, optional
#         Extrapolate the grid past the convex hull of the known points using nearest neighbour interpolation.

#     Returns
#     -------
#     x : array of floats
#         The unique grid node along the first dimension
#     y : array of floats
#         The unique grid node along the second dimension
#     vals : array of floats
#         The interpolated values on a grid, represented by a 2D array

#     """

#     if x_grid is None:
#         x_grid = centred_grid_nodes(bounds[:2], dx)
#     if y_grid is None:
#         y_grid = centred_grid_nodes(bounds[2:], dy)

#     values[values == inf] = nan
#     mn = nanmin(values)
#     mx = nanmax(values)

#     values -= mn
#     if (mx - mn) != 0.0:
#         values = values / (mx - mn)

#     # Create the CT function for interpolation
#     f = CloughTocher2DInterpolator(XY, values)

#     xc, yc, intPoints = getGridLocations2D(bounds, dx, dy, x_grid, y_grid)

#     # Interpolate to the grid
#     vals = f(intPoints).reshape(yc.size, xc.size)

#     # Reshape to a 2D array
#     vals = StatArray.StatArray(vals, name=cf.getName(values), units = cf.getUnits(values))

#     if mask or extrapolate:
#         if (kdtree is None):
#             kdt = cKDTree(XY)
#         else:
#             kdt = kdtree

#     # Use distance masking
#     if mask:
#         g = meshgrid(xc, yc)
#         xi = _ndim_coords_from_arrays(tuple(g), ndim=XY.shape[1])
#         dists, indexes = kdt.query(xi)
#         vals[dists > mask] = nan

#     # Truncate values to the observed values
#     if (clip):
#         minV = nanmin(values)
#         maxV = nanmax(values)
#         mask = ~isnan(vals)
#         mask[mask] &= vals[mask] < minV
#         vals[mask] = minV
#         mask = ~isnan(vals)
#         mask[mask] &= vals[mask] > maxV
#         vals[mask] = maxV

#     if (not extrapolate is None):
#         assert isinstance(extrapolate,str), 'extrapolate must be a string. Choose [Nearest]'
#         extrapolate = extrapolate.lower()
#         if (extrapolate == 'nearest'):
#             # Get the indices of the nans
#             iNan = argwhere(isnan(vals))
#             # Create Query locations from the nans
#             xi =  zeros([iNan.shape[0],2])
#             xi[:,0]=x[iNan[:,1]]
#             xi[:,1]=y[iNan[:,0]]
#             # Get the nearest neighbours
#             dists, indexes = kdt.query(xi)
#             # Assign the nearest value to the Nan locations
#             vals[iNan[:,0],iNan[:,1]] = values[indexes]
#         else:
#             assert False, 'Extrapolation option not available. Choose [Nearest]'

#     if (mx - mn) != 0.0:
#         vals = vals * (mx - mn)
#     vals += mn

#     return xc, yc, vals

# def minimumCurvature(x, y, values, x_grid=None, y_grid=None, dx=None, dy=None,
#        bounds=None, mask=False, clip=False, iterations=2000, tension=0.25, accuracy=0.01, verbose=False, **kwargs):

#     from pygmt import surface

#     values[values == inf] = nan
#     mn = nanmin(values)
#     mx = nanmax(values)

#     values -= mn
#     if (mx - mn) != 0.0:
#         values = values / (mx - mn)

#     if x_grid is None:
#         x_grid = centred_grid_nodes(bounds[:2], dx)
#     if y_grid is None:
#         y_grid = centred_grid_nodes(bounds[2:], dy)

#     xc = StatArray.StatArray(x_grid, name=cf.getName(x), units=cf.getUnits(x))
#     yc = StatArray.StatArray(y_grid, name=cf.getName(y), units=cf.getUnits(y))

#     bounds = r_[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
#     dx = abs(x_grid[1] - x_grid[0])
#     dy = abs(y_grid[1] - y_grid[0])

#     if clip:
#         clip_min = kwargs.pop('clip_min', nanmin(values))
#         clip_max = kwargs.pop('clip_max', nanmax(values))
#         xr = surface(x=x, y=y, z=values, spacing=(dx, dy), region=bounds, N=iterations, T=tension, C=accuracy, Ll=[clip_min], Lu=[clip_max])
#     else:
#         xr = surface(x=x, y=y, z=values, spacing=(dx, dy), region=bounds, N=iterations, T=tension, C=accuracy)

#     vals = xr.values

#     if (mx - mn) != 0.0:
#         vals = vals * (mx - mn)
#     vals += mn

#     # Use distance masking
#     if mask:
#         kdt = cKDTree(column_stack((x, y)))
#         xi = _ndim_coords_from_arrays(tuple(meshgrid(xc, yc)), ndim=2)
#         dists, indexes = kdt.query(xi)
#         vals[dists > mask] = nan

#     vals = StatArray.StatArray(vals, name=cf.getName(values), units = cf.getUnits(values))
#     return xc, yc, vals

# # def centred_grid_nodes(bounds, spacing):
# #     """Generates grid nodes centred over bounds

# #     Parameters
# #     ----------
# #     bounds : array_like
# #         bounds of the dimension
#     spacing : float
#         distance between nodes

#     """
#     # Get the discretization
#     assert spacing > 0.0, ValueError("spacing must be positive!")
#     nx = int32(floor((bounds[1] - bounds[0])/spacing) + 1)
#     mid = 0.5 * (bounds[0] + bounds[1])
#     sx = 0.5 * nx * spacing
#     return linspace(mid - sx, mid + sx, nx)

# def getGridLocations2D(bounds, dx=None, dy=None, x_grid=None, y_grid=None):
#     """Discretize a 2D bounding box by increments of dx and return the grid node locations

#     Parameters
#     ----------
#     bounds : array of floats
#         Length 4 array with the minimum and maximum in two directions. [Xmin, Xmax, Ymin, Ymax]
#     dx : float
#         The spacing between grid nodes

#     Returns
#     -------
#     x : array of floats
#         The unique grid node along the first dimension
#     y : array of floats
#         The unique grid node along the second dimension
#     intPoints : array of floats
#         2D array containing all grid node locations

#     """
#     # Create the cell centres from the bounds
#     if x_grid is None:
#         x_grid = centred_grid_nodes(bounds[:2], dx)
#     if y_grid is None:
#         y_grid = centred_grid_nodes(bounds[2:], dy)

#     x = x_grid[:-1] + 0.5 * diff(x_grid)
#     y = y_grid[:-1] + 0.5 * diff(y_grid)
#     # Create the unpacked grid locations
#     intPoints = zeros([x.size * y.size, 2], order = 'F')
#     intPoints[:, 0] = tile(x, y.size)
#     intPoints[:, 1] = y.repeat(x.size)
#     return x_grid, y_grid, intPoints
