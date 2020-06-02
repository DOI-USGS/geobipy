import subprocess
import numpy as np
from .fileIO import deleteFile
from ..classes.core import StatArray
from . import customFunctions as cf
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

def CT(dx, dy, bounds, XY, values, mask = False, kdtree = None, clip = False, extrapolate=None):
    """Use Scipy's CloughTocher C1 continuous interpolation using unstructured meshes to interpolate arbitrary locations to a grid

    Parameters
    ----------
    dx : float
        The required spacing between grid nodes
    bounds : ndarray of floats
        Length 4 array with the minimum and maximum in two directions. [Xmin, Xmax, Ymin, Ymax]
    XY : 2D ndarray of floats
        Two columns, each column contains the co-ordinate in a dimension
    values : ndarray
        The values to interpolate to the grid
    mask : float, optional
        Force interpolated values that are greater than a distance of mask from any known point to be NaN
    kdtree : scipy.spatial.ckdtree.cKDTree, optional
        If no kdtree is given for the set of points, one is created.  To speed up multiple interpolations, the user can pass their own fixed kdtree and prevent the generation of one every time.
    clip : bool, optional
        Interpolation can overshoot the known value. clip = True ensures that the min  max of the grid is the same as the known data points.
    extrapolate : bool, optional
        Extrapolate the grid past the convex hull of the known points using nearest neighbour interpolation.

    Returns
    -------
    x : array of floats
        The unique grid node along the first dimension
    y : array of floats
        The unique grid node along the second dimension
    vals : array of floats
        The interpolated values on a grid, represented by a 2D array

    """

    values[values == np.inf] = np.nan
    mn = np.nanmin(values)
    mx = np.nanmax(values)

    if (mx - mn) == 0.0:
        values = values - mn
    else:
        values = (values - mn) / (mx - mn)

    # Create the CT function for interpolation
    f = CloughTocher2DInterpolator(XY, values)

    xc, yc, intPoints = getGridLocations2D(bounds, dx, dy)

    # Interpolate to the grid
    vals = f(intPoints)

    # Reshape to a 2D array
    vals = StatArray.StatArray(vals.reshape(yc.size, xc.size),name=cf.getName(values), units = cf.getUnits(values))


    if mask or extrapolate:
        if (kdtree is None):
            kdt = cKDTree(XY)
        else:
            kdt = kdtree

    # Use distance masking
    if mask:
        g = np.meshgrid(xc,yc)
        xi = _ndim_coords_from_arrays(tuple(g), ndim=XY.shape[1])
        dists, indexes = kdt.query(xi)
        vals[dists > mask] = np.nan

    # Truncate values to the observed values
    if (clip):
        minV = np.nanmin(values)
        maxV = np.nanmax(values)
        mask = ~np.isnan(vals)
        mask[mask] &= vals[mask] < minV
        vals[mask] = minV
        mask = ~np.isnan(vals)
        mask[mask] &= vals[mask] > maxV
        vals[mask] = maxV

    if (not extrapolate is None):
        assert isinstance(extrapolate,str), 'extrapolate must be a string. Choose [Nearest]'
        extrapolate = extrapolate.lower()
        if (extrapolate == 'nearest'):
            # Get the indices of the nans
            iNan = np.argwhere(np.isnan(vals))
            # Create Query locations from the nans
            xi =  np.zeros([iNan.shape[0],2])
            xi[:,0]=x[iNan[:,1]]
            xi[:,1]=y[iNan[:,0]]
            # Get the nearest neighbours
            dists, indexes = kdt.query(xi)
            # Assign the nearest value to the Nan locations
            vals[iNan[:,0],iNan[:,1]] = values[indexes]
        else:
            assert False, 'Extrapolation option not available. Choose [Nearest]'

    if (mx - mn) == 0.0:
        values = values + mn
    else:
        vals = (vals * (mx - mn)) + mn

    return xc, yc, vals


def minimumCurvature(x, y, values, bounds, dx, dy, mask=False, clip=False, iterations=2000, tension=0.25, accuracy=0.01):

    values[values == np.inf] = np.nan
    mn = np.nanmin(values)
    mx = np.nanmax(values)

    if (mx - mn) == 0.0:
        values = values - mn
    else:
        values = (values - mn) / (mx - mn)

    T = np.column_stack([x, y, values])
    np.savetxt('tmp.txt', T)

    bnds = bounds.copy()
    nx = np.int(np.ceil((bnds[1]-bnds[0])/dx))
    ny = np.int(np.ceil((bnds[3]-bnds[2])/dy))

    bnds[0] -= 0.5*dx
    bnds[2] -= 0.5*dy

    bnds[1] = bnds[0] + (nx+1)*dx
    bnds[3] = bnds[2] + (ny+1)*dy

    # Create the grid axes
    x = np.linspace(bnds[0], bnds[0]+nx*dx, nx+1)
    y = np.linspace(bnds[2], bnds[2]+ny*dx, ny+1)

    increments = "-I%g/%g"%(dx,dy)
    region = "-R%g/%g/%g/%g"%(bnds[0], bnds[1], bnds[2], bnds[3])

    if clip:
        subcall = ["gmt", "surface", "tmp.txt", increments, region, "-N%d"%(iterations), "-T%g"%(tension), "-C%g"%(accuracy), "-Gtmp.grd", "-Ll%g"%(np.nanmin(values)), "-Lu%g"%(np.nanmax(values))]
        subprocess.call(subcall)
    else:
        subcall = ["gmt", "surface", "tmp.txt", increments, region, "-N%d"%(iterations), "-T%g"%(tension), "-C%g"%(accuracy), "-Gtmp.grd"]
        subprocess.call(subcall)

    print('Interpolating with {}'.format(' '.join(subcall)))

    with Dataset("tmp.grd", "r") as f:
        xT = np.asarray(f['x'])
        yT = np.asarray(f['y'])
        vals = np.asarray(f['z'])
    deleteFile("tmp.grd")

    if (mx - mn) == 0.0:
        values = values + mn
    else:
        vals = (vals * (mx - mn)) + mn

    if mask:
        masked = "-S%g"%(mask)
        subprocess.call(["gmt", "grdmask", "tmp.txt", increments, region, masked, "-Gmask.grd"])

        with Dataset("mask.grd", 'r') as f:
            msk = np.asarray(f['z'])
        deleteFile("mask.grd")

        msk[msk == 0.0] = np.nan
        vals *= msk
        deleteFile("mask.grd")

#    # Truncate values to the observed values
#    if (clip):
#        minV = np.nanmin(values)
#        maxV = np.nanmax(values)
#        mask = ~np.isnan(vals)
#        mask[mask] &= vals[mask] < minV
#        vals[mask] = minV
#        mask = ~np.isnan(vals)
#        mask[mask] &= vals[mask] > maxV
#        vals[mask] = maxV

    deleteFile('tmp.txt')

    xT = StatArray.StatArray(x, name=cf.getName(x), units=cf.getUnits(x))
    yT = StatArray.StatArray(y, name=cf.getName(y), units=cf.getUnits(y))

    vals = StatArray.StatArray(vals, name=cf.getName(values), units = cf.getUnits(values))
    return xT, yT, vals

def getGridLocations2D(bounds, dx, dy):
    """Discretize a 2D bounding box by increments of dx and return the grid node locations

    Parameters
    ----------
    bounds : array of floats
        Length 4 array with the minimum and maximum in two directions. [Xmin, Xmax, Ymin, Ymax]
    dx : float
        The spacing between grid nodes

    Returns
    -------
    x : array of floats
        The unique grid node along the first dimension
    y : array of floats
        The unique grid node along the second dimension
    intPoints : array of floats
        2D array containing all grid node locations

    """
    # Create the cell centres from the bounds
    centres = np.asarray([bounds[0]+0.5*dx, bounds[1]-0.5*dx, bounds[2]+0.5*dy, bounds[3]-0.5*dy])
    nx = np.int((centres[1]-centres[0])/dx)
    ny = np.int((centres[3]-centres[2])/dy)
    x = np.linspace(centres[0], centres[1], nx+1)
    y = np.linspace(centres[2], centres[3], ny+1)
    # Create the unpacked grid locations
    intPoints = np.zeros([x.size*y.size,2], order = 'F')
    intPoints[:,0] = np.tile(x,y.size)
    intPoints[:,1] = y.repeat(x.size)
    return x,y,intPoints
