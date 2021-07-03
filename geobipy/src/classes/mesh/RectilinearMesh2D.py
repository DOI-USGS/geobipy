""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import deepcopy
from .Mesh import Mesh
from ...classes.core import StatArray
from ..model.Model import Model
from . import RectilinearMesh1D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import binned_statistic
from ...base import plotting as cP
from ...base import utilities as cF
from scipy.sparse import (kron, diags)
from scipy import interpolate

try:
    from pyvtk import VtkData, CellData, Scalars, PolyData
except:
    pass


class RectilinearMesh2D(Mesh):
    """Class defining a 2D rectilinear mesh with cell centres and edges.

    Contains a simple 2D mesh with cell edges, widths, and centre locations.
    There are two ways of instantiating the RectilinearMesh2D.
    The first is by specifying the x and y cell centres or edges. In this case,
    the abscissa is the standard x axis, and y is the ordinate. The z co-ordinates are None.
    The second is by specifyin the x, y, and z cell centres or edges. In this case,
    The mesh is a 2D plane with the ordinate parallel to z, and the "horizontal" locations
    have co-ordinates (x, y).
    This allows you to, for example, create a vertical 2D mesh that is not parallel to either the
    x or y axis, like a typical line of data.
    If x, y, and z are specified, plots can be made against distance which calculated cumulatively between points.

    RectilinearMesh2D([xCentres or xEdges], [yCentres or yEdges], [zCentres or zEdges])

    Parameters
    ----------
    xCentres : geobipy.StatArray, optional
        The locations of the centre of each cell in the "x" direction. Only xCentres or xEdges can be given.
    xEdges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges, in the "x" direction. Only xCentres or xEdges can be given.
    yCentres : geobipy.StatArray, optional
        The locations of the centre of each cell in the "y" direction. Only yCentres or yEdges can be given.
    yEdges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges, in the "y" direction. Only yCentres or yEdges can be given.
    zCentres : geobipy.StatArray, optional
        The locations of the centre of each cell in the "z" direction. Only zCentres or zEdges can be given.
    zEdges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges, in the "z" direction. Only zCentres or zEdges can be given.

    Other Parameters
    ----------------
    [x, y, z]edgesMin : float, optional
        See geobipy.RectilinearMesh1D for edgesMin description.
    [x, y, z]edgesMax : float, optional
        See geobipy.RectilinearMesh1D for edgesMax description.
    [x, y, z]log : 'e' or float, optional
        See geobipy.RectilinearMesh1D for log description.
    [x, y, z]relativeTo : float, optional
        See geobipy.RectilinearMesh1D for relativeTo description.

    Returns
    -------
    out : RectilinearMesh2D
        The 2D mesh.

    """

    def __init__(self, xCentres=None, xEdges=None, yCentres=None, yEdges=None, zCentres=None, zEdges=None, heightCentres=None, heightEdges=None, **kwargs):
        """ Initialize a 2D Rectilinear Mesh"""

        self._x = None
        self._y = None
        self._z = None
        self._distance = None
        self.xyz = None
        self._height = None

        if (all(x is None for x in [xCentres, yCentres, zCentres, xEdges, yEdges, zEdges])):
            return

        xExtras = dict((k[1:], kwargs.pop(k, None)) for k in ['xlog'])
        self.x = RectilinearMesh1D.RectilinearMesh1D(centres=xCentres, edges=xEdges, relativeTo=kwargs.pop('xrelativeTo', 0.0), **xExtras)

        yExtras = dict((k[1:], kwargs.pop(k, None)) for k in ['ylog'])
        self.y = RectilinearMesh1D.RectilinearMesh1D(centres=yCentres, edges=yEdges, relativeTo=kwargs.pop('yrelativeTo', 0.0),  **yExtras)

        self.xyz = False
        self._z = self._y

        if (not zCentres is None or not zEdges is None):
            zExtras = dict((k[1:], kwargs.pop(k, None)) for k in ['zlog'])
            self.z = RectilinearMesh1D.RectilinearMesh1D(centres=zCentres, edges=zEdges, relativeTo=kwargs.pop('zrelativeTo', 0.0), **zExtras)
            self.xyz = True

        if not ((heightCentres is None) and (heightEdges is None)):
            # mesh of the z axis values
            self._height = RectilinearMesh1D.RectilinearMesh1D(centres=heightCentres, edges=heightEdges)

            assert self.height.nCells == self.x.nCells, Exception("heights must have enough values for {} cells or {} edges.\nInstead got {}".format(self.x.nCells, self.x.nEdges, self.height.nCells))

    def __getitem__(self, slic):
        """Allow slicing of the histogram.

        """
        assert np.shape(slic) == (2,), ValueError("slic must be over 2 dimensions.")

        slic0 = slic

        slic = []
        axis = -1
        for i, x in enumerate(slic0):
            if isinstance(x, (int, np.integer)):
                tmp = x
                axis = i
            else:
                tmp = x
                if isinstance(x.stop, (int, np.integer)):
                    # If a slice, add one to the end for bins.
                    tmp = slice(x.start, x.stop+1, x.step)

            slic.append(tmp)
        slic = tuple(slic)

        if axis == -1:
            height = self.height.edges[slic[1]] if not self.height is None else None
            if self.xyz:
                out = type(self)(xEdges=self._x.edges[slic[1]], yEdges=self._y.edges[slic[1]], zEdges=self._z.edges[slic[0]], heightEdges=height)
                return out
            else:
                out = type(self)(xEdges=self._x.edges[slic[1]], yEdges=self._z.edges[slic[0]], heightEdges=height)
            return out

        out = RectilinearMesh1D.RectilinearMesh1D(edges=self.axis(1-axis).edges[slic[1-axis]])
        return out

    @property
    def distance(self):
        """The distance along the top of the mesh using the x and y co-ordinates. """

        assert self.xyz, Exception("To set the distance, the mesh must be instantiated with three co-ordinates")

        if self._distance is None:

            dx = np.diff(self.x.edges)
            dy = np.diff(self.y.edges)

            distance = StatArray.StatArray(np.zeros(self.x.nEdges), 'Distance', self.x.centres.units)
            distance[1:] = np.cumsum(np.sqrt(dx**2.0 + dy**2.0))

            self._distance = RectilinearMesh1D.RectilinearMesh1D(edges = distance)
        return self._distance

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, values):
        self._height = values

    @property
    def nCells(self):
        """The number of cells in the mesh.

        Returns
        -------
        out : int
            Number of cells

        """

        return self.x.nCells * self.z.nCells


    @property
    def nNodes(self):
        """The number of nodes in the mesh.

        Returns
        -------
        out : int
            Number of nodes

        """

        return self.x.nEdges * self.z.nEdges


    @property
    def shape(self):
        """The dimensions of the mesh

        Returns
        -------
        out : array_like
            Array of integers

        """

        return (self.z.nCells.value, self.x.nCells.value)


    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, values):
        assert isinstance(values, RectilinearMesh1D.RectilinearMesh1D)
        self._x = values

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, values):
        assert isinstance(values, RectilinearMesh1D.RectilinearMesh1D)
        self._y = values

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, values):
        assert self.x.nCells == self.y.nCells, Exception("x and y axes must have the same number of cells.")
        assert isinstance(values, RectilinearMesh1D.RectilinearMesh1D)
        self._z = values


    # def _mean(self, arr, log=None, axis=0):
    #     a = self.axis(axis)
    #     b = self.other_axis(axis)

    #     t = np.sum(np.repeat(np.expand_dims(a.centres, axis), b.nCells, axis) * self.counts, 1-axis)
    #     s = self._counts.sum(axis = 1 - axis)

    #     i = np.where(s > 0.0)[0]
    #     out = np.zeros(t.size)
    #     out[i] = t[i] / s[i]

    #     if log:
    #         out, dum = cF._log(out, log=log)

    #     return out


    def _percent_interval(self, values, percent=95.0, log=None, axis=0):
        """Gets the percent interval along axis.

        Get the statistical interval, e.g. median is 50%.

        Parameters
        ----------
        values : array_like
            Valus used to compute interval like histogram counts.
        percent : float
            Interval percentage.  0.0 < percent < 100.0
        log : 'e' or float, optional
            Take the log of the interval to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the interval locations.

        Returns
        -------
        interval : array_like
            Contains the interval along the specified axis. Has size equal to self.shape[axis].

        """

        percent *= 0.01

        # total of the counts
        total = values.sum(axis=1-axis)
        # Cumulative sum
        cs = np.cumsum(values, axis=1-axis)
        # Cumulative "probability"
        tmp = np.divide(cs, np.expand_dims(total, 1-axis))
        # Find the interval
        i = np.apply_along_axis(np.searchsorted, 1-axis, tmp, percent)
        # Obtain the values at those locations
        out = self.axis(axis).centres[i]

        if (not log is None):
            out, dum = cF._log(out, log=log)

        return out


    def _credibleIntervals(self, values, percent=90.0, log=None, axis=0):
        """Gets the median and the credible intervals for the specified axis.

        Parameters
        ----------
        values : array_like
        Values to use to compute the intervals.
        percent : float
        Confidence percentage.
        log : 'e' or float, optional
        Take the log of the credible intervals to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
        Along which axis to obtain the interval locations.

        Returns
        -------
        med : array_like
        Contains the medians along the specified axis. Has size equal to arr.shape[axis].
        low : array_like
        Contains the lower interval along the specified axis. Has size equal to arr.shape[axis].
        high : array_like
        Contains the upper interval along the specified axis. Has size equal to arr.shape[axis].

        """

        percent = 0.5 * np.minimum(percent, 100.0 - percent)

        lower = self._percent_interval(values, percent, log, axis)
        median = self._percent_interval(values, 50.0, log, axis)
        upper = self._percent_interval(values, 100.0 - percent, log, axis)

        return (median, lower, upper)


    def _credibleRange(self, values, percent=90.0, log=None, axis=0):
        """ Get the range of credibility

        Parameters
        ----------
        values : array_like
            Values to use to compute the range.
        percent : float
            Percent of the credible intervals
        log : 'e' or float, optional
            If None: The range is the difference in linear space of the credible intervals
            If 'e' or float: The range is the difference in log space, or ratio in linear space.
        axis : int
            Axis along which to get the marginal histogram.


        """
        sMed, sLow, sHigh = self._credibleIntervals(values, percent, log=log, axis=axis)

        return sHigh - sLow



    def _median(self, values, log=None, axis=0):
        """Gets the median for the specified axis.

        Parameters
        ----------
        values : array_like
            2D array to get the median.
        log : 'e' or float, optional
            Take the log of the median to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the median.

        Returns
        -------
        med : array_like
            Contains the medians along the specified axis. Has size equal to arr.shape[axis].

        """
        return self._percent_interval(values=values, percent=50.0, log=log, axis=axis)


    def __deepcopy__(self, memo):
        """ Define the deepcopy for the StatArray """
        height = self.height.edges if not self.height is None else None
        if self.xyz:
            return RectilinearMesh2D(xEdges=self.x.edges, yEdges=self.y.edges, zEdges=self.z.edges, heightEdges=height)
        else:
            return RectilinearMesh2D(xEdges=self.x.edges, yEdges=self.z.edges, heightEdges=height)


    def edges(self, axis):
        """ Gets the cell edges in the given dimension """
        return self.axis(axis).edges


    def other_axis(self, axis):
        if axis == 0:
            return self.x
        else:
            return self.z


    def axis(self, axis):
        if isinstance(axis, str):
            assert axis in ['x', 'y', 'z', 'r'], Exception("axis must be either 'x', 'y', 'z', or 'r'")

            if axis == 'x':
                return self.x
            elif axis == 'y':
                assert self.xyz, Exception("To plot against 'y' the mesh must be instantiated with three co-ordinates")
                return self.y
            elif axis == 'z':
                return self.z
            elif axis == 'r':
                assert self.xyz, Exception("To plot against 'r' the mesh must be instantiated with three co-ordinates")
                return self.distance

        if axis == 0:
            return self.z
        elif axis == 1:
            return self.x


    def xGradientMatrix(self):
        tmp = self.x.gradientMatrix()
        return kron(diags(np.sqrt(self.z.widths)), tmp)


    def zGradientMatrix(self):
        nx = self.x.nCells.value
        nz = self.z.nCells.value
        tmp = 1.0 / np.sqrt(self.x.centreTocentre)
        a = np.repeat(tmp, nz) * np.tile(np.sqrt(self.z.widths), nx-1)
        return diags([a, -a], [0, nx], shape=(nz * (nx-1), nz*nz))


    def hasSameSize(self, other):
        """ Determines if the meshes have the same dimension sizes """
        # if self.arr.shape != other.arr.shape:
        #     return False
        if self.x.nCells != other.x.nCells:
            return False
        if self.z.nCells != other.z.nCells:
            return False
        return True

    # def intersect(self, x, y, axis=0):
    #     """Intersect coordinates with

    #     [extended_summary]

    #     Parameters
    #     ----------
    #     values : [type]
    #         [description]
    #     axis : int, optional
    #         [description].
    #         Defaults to 0.

    #     Returns
    #     -------
    #     [type] : [description]
    #     """
    #     return out

    def resample(self, dx, dy, values, kind='cubic'):
        x = np.arange(self.x.edges[0], self.x.edges[-1]+dx, dx)
        y = np.arange(self.y.edges[0], self.y.edges[-1]+dy, dy)

        height = None
        if not self.height is None:
            height = self.height.resample(dx).values
        mesh = RectilinearMesh2D(xEdges=x, yEdges=y, heightCentres = height)

        f = interpolate.interp2d(self.x.centres, self.y.centres, values, kind=kind)
        return mesh, f(mesh.x.centres, mesh.y.centres)

    def interpolate_centres_to_nodes(self, values, kind='cubic'):
        if self.x.nCells <= 3 or self.y.nCells <= 3:
            kind = 'linear'
        f = interpolate.interp2d(self.x.centres, self.y.centres, values, kind=kind)
        return f(self.x.edges, self.y.edges)

    def intervalStatistic(self, arr, intervals, axis=0, statistic='mean'):
        """Compute a statistic of the array between the intervals given along dimension dim.

        Parameters
        ----------
        arr : array_like
            2D array to take the mean over the given intervals
        intervals : array_like
            A new set of mesh edges. The mean is computed between each two edges in the array.
        axis : int, optional
            Which axis to take the mean
        statistic : string or callable, optional
            The statistic to compute (default is 'mean').
            The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.


        See Also
        --------
        scipy.stats.binned_statistic : for more information

        """

        assert np.size(intervals) > 1, ValueError("intervals must have size > 1")

        intervals = self._reconcile_intervals(intervals, axis=axis)

        if (axis == 0):
            bins = binned_statistic(self.z.centres, arr.T, bins = intervals, statistic=statistic)
            res = bins.statistic.T
        else:
            bins = binned_statistic(self.x.centres, arr, bins = intervals, statistic=statistic)
            res = bins.statistic

        return res, intervals


    def mask_cells(self, xAxis, x_distance=None, z_distance=None, values=None):
        """Mask cells by a distance.

        If the edges of the cell are further than distance away, extra cells are inserted such that
        the cell's new edges are at distance away from the centre.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'x', the horizontal axis uses self.x
            If xAxis is 'y', the horizontal axis uses self.y
            If xAxis is 'r', the horizontal axis uses cumulative distance along the line
        x_distance : float, optional
            Mask along the x axis using this distance.
            Defaults to None.
        y_distance : float, optional
            Mask along the y axis using this distance.
            Defaults to None.
        values : array_like, optional.
            If given, values will be remapped to the masked mesh.
            Has shape (y.nCells, x.nCells)

        Returns
        -------
        out : RectilinearMesh2D
            Masked mesh
        x_indices : ints, optional
            Location of the original centres in the expanded mesh along the x axis.
        y_indices : ints, optional
            Location of the original centres in the expanded mesh along the y axis.
        out_values : array_like, optional
            If values is given, values will be remapped to the masked mesh.

        """

        out_values = None
        if not values is None:
            out_values = values

        x_indices = None
        x = self.axis(xAxis)
        if not x_distance is None:
            x, x_indices = self.x.mask_cells(x_distance)
            if not values is None:
                out_values = np.full((self.z.nCells.value, x.nCells.value), fill_value=np.nan)
                for i in range(self.x.nCells.value):
                    out_values[:, x_indices[i]] = values[:, i]

        z_indices = None
        z = self.z
        if not z_distance is None:
            z, z_indices = self.z.mask_cells(z_distance)
            if not values is None:
                out_values2 = np.full((z.nCells.value, out_values.shape[1]), fill_value=np.nan)
                for i in range(self.z.nCells.value):
                    out_values2[z_indices[i], :] = out_values[i, :]
                out_values = out_values2

        height = None
        if not self.height is None:
            height = np.interp(x.edges, self.x.edges, self.height.edges)

        out = type(self)(xEdges=x.edges, yEdges=z.edges, heightEdges=height)

        return out, x_indices, z_indices, out_values


    def _reconcile_intervals(self, intervals, axis=0):

        assert np.size(intervals) > 1, ValueError("intervals must have size > 1")

        ax = self.other_axis(axis)

        i0 = np.maximum(0, np.searchsorted(intervals, ax.edges[0]))
        i1 = np.minimum(ax.nCells.value, np.searchsorted(intervals, ax.edges[-1])+1)

        intervals = intervals[i0:i1]

        return intervals

    def cellIndex(self, values, axis, clip=False, trim=False):
        """Return the cell indices of values along axis.

        Parameters
        ----------
        values : scalar or array_like
            Locations to obtain the cell index for
        axis : int
            Axis along which to obtain indices
        clip : bool
            A negative index which would normally wrap will clip to 0 instead.
        trim : bool
            Do not include out of axis indices. Negates clip, since they wont be included in the output.

        Returns
        -------
        out : ints
            indices for the locations along the axis

        """
        return self.axis(axis).cellIndex(values, clip=clip, trim=trim)

    def cellIndices(self, x, y, clip=False, trim=False):
        """Return the cell indices in x and z for two floats.

        Parameters
        ----------
        x : scalar or array_like
            x location
        y : scalar or array_like
            y location (or z location if instantiated with 3 co-ordinates)
        clip : bool
            A negative index which would normally wrap will clip to 0 instead.
        trim : bool
            Do not include out of axis indices. Negates clip, since they wont be included in the output.

        Returns
        -------
        out : ints
            indices for the locations along [axis0, axis1]

        """
        assert (np.size(x) == np.size(y)), ValueError("x and y must have the same size")
        if trim:
            flag = self.x.inBounds(x) & self.z.inBounds(y)
            i = np.where(flag)[0]
            out = np.empty([2, i.size], dtype=np.int32)
            out[1, :] = self.x.cellIndex(x[i])
            out[0, :] = self.z.cellIndex(y[i])
        else:
            out = np.empty([2, np.size(x)], dtype=np.int32)
            out[1, :] = self.x.cellIndex(x, clip=clip)
            out[0, :] = self.z.cellIndex(y, clip=clip)
        return np.squeeze(out)


    def ravelIndices(self, ixy, order='C'):
        """Return a global index into a 1D array given the two cell indices in x and z.

        Parameters
        ----------
        ixy : tuple of array_like
            A tuple of integer arrays, one array for each dimension.

        Returns
        -------
        out : int
            Global index.

        """

        return np.ravel_multi_index(ixy, self.shape, order=order)


    def unravelIndex(self, indices, order='C'):
        """Return a global index into a 1D array given the two cell indices in x and z.

        Parameters
        ----------
        indices : array_like
            An integer array whose elements are indices into the flattened
            version of an array.

        Returns
        -------
        unraveled_coords : tuple of ndarray
            Each array in the tuple has the same shape as the self.shape.

        """

        return np.unravel_index(indices, self.shape, order=order)


    def pcolor(self, values, xAxis='x', **kwargs):
        """Create a pseudocolour plot.

        Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

        Parameters
        ----------
        values : array_like
            2D array of colour values
        xAxis : str
            If xAxis is 'x', the horizontal axis uses self.x
            If xAxis is 'y', the horizontal axis uses self.y
            If xAxis is 'r', the horizontal axis uses cumulative distance along the line

        Other Parameters
        ----------------
        alpha : scalar or array_like, optional
            If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
            If array_like, each pixel is given an individual alpha value.
        log : 'e' or float, optional
            Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
            Values in c that are <= 0 are masked.
        equalize : bool, optional
            Equalize the histogram of the colourmap so that all colours have an equal amount.
        nbins : int, optional
            Number of bins to use for histogram equalization.
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'.
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        grid : bool, optional
            Plot the grid
        noColorbar : bool, optional
            Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.
        x_mask : float, optional
            Mask along the x axis using this distance.
            Defaults to None.
        z_mask : float, optional
            Mask along the z axis using this distance.
            Defaults to None.

        See Also
        --------
        geobipy.plotting.pcolor : For non matplotlib keywords.
        matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

        """

        assert np.all(values.shape == self.shape), ValueError("values must have shape {}".format(self.shape))

        x_mask = kwargs.pop('x_mask', None)
        z_mask = kwargs.pop('z_mask', None)

        if np.sum([x is None for x in [x_mask, z_mask]]) < 2:
            masked, x_indices, z_indices, values = self.mask_cells(xAxis, x_mask, z_mask, values)
            ax, pm, cb = cP.pcolor(values, x = masked.axis('x').edges, y = masked.z.edges, **kwargs)
        else:
            ax, pm, cb = cP.pcolor(values, x = self.axis(xAxis).edges, y = self.z.edges, **kwargs)


        return ax, pm, cb

    def pcolor(self, values, xAxis='x', zAxis='absolute', **kwargs):
        """Create a pseudocolour plot of a 2D array using the mesh.

        Parameters
        ----------
        values : array_like or StatArray
            A 2D array of colour values.
        xAxis : str
            If xAxis is 'x', the horizontal xAxis uses self.x
            If xAxis is 'y', the horizontal xAxis uses self.y
            If xAxis is 'r', the horizontal xAxis uses cumulative distance along the line
        zAxis : str
            If zAxis is 'absolute' the vertical axis is the height plus z.
            If zAxis is 'relative' the vertical axis is z.

        Other Parameters
        ----------------
        alpha : scalar or array_like, optional
            If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
            If array_like, each pixel is given an individual alpha value.
        log : 'e' or float, optional
            Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
            Values in c that are <= 0 are masked.
        equalize : bool, optional
            Equalize the histogram of the colourmap so that all colours have an equal amount.
        nbins : int, optional
            Number of bins to use for histogram equalization.
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'.
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        grid : bool, optional
            Plot the grid
        noColorbar : bool, optional
            Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.

        Returns
        -------
        ax
            matplotlib .Axes

        See Also
        --------
        matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

        """
        # assert isinstance(values, StatArray), TypeError("values must be a StatArray")
        assert np.all(values.shape == self.shape), ValueError("values must have shape {} but have shape {}".format(self.shape, values.shape))

        x_mask = kwargs.pop('x_mask', None)
        z_mask = kwargs.pop('z_mask', None)

        if self.height is None or zAxis == 'relative':
            if np.sum([x is None for x in [x_mask, z_mask]]) < 2:
                masked, x_indices, z_indices, values = self.mask_cells(xAxis, x_mask, z_mask, values)
                ax, pm, cb = cP.pcolor(values, x = masked.axis('x').edges, y = masked.z.edges, **kwargs)
            else:
                ax, pm, cb = cP.pcolor(values, x = self.axis(xAxis).edges, y = self.z.edges, **kwargs)

        else:
            masked = self
            if np.sum([x is None for x in [x_mask, z_mask]]) < 2:
                masked, x_indices, z_indices, values = self.mask_cells(xAxis, x_mask, z_mask, values)
                xAxis='x'

            xm = masked.xMesh(xAxis=xAxis)
            zm = masked.zMesh

            # if zAxis.lower() == 'relative':
            #     kwargs['flipY'] = kwargs.pop('flipY', True)

            ax, pm, cb = cP.pcolormesh(xm, zm, values, **kwargs)
            cP.xlabel(xm.label)
            cP.ylabel(zm.label)

        return ax, pm, cb


    def plotGrid(self, xAxis='x', **kwargs):
        """Plot the mesh grid lines.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'x', the horizontal axis uses self.x
            If xAxis is 'y', the horizontal axis uses self.y
            If xAxis is 'r', the horizontal axis uses sqrt(self.x^2 + self.y^2)

        """

        xscale = kwargs.pop('xscale', 'linear')
        yscale = kwargs.pop('yscale', 'linear')
        flipX = kwargs.pop('flipX', False)
        flipY = kwargs.pop('flipY', False)
        c = kwargs.pop('color', 'k')

        if self.height is None:

            tmp = StatArray.StatArray(np.full(self.shape, fill_value=np.nan))
            tmp.pcolor(x=self.axis(xAxis).edges, y=self.z.edges, grid=True, noColorbar=True, **kwargs)

        else:
            xtmp = self.axis(xAxis).edges

            ax = plt.gca()
            cP.pretty(ax)
            zMesh = self.zMesh
            ax.vlines(x = xtmp, ymin=zMesh[0, :], ymax=zMesh[-1, :], **kwargs)
            segs = np.zeros([self.z.nEdges, self.x.nEdges, 2])
            segs[:, :, 0] = np.repeat(xtmp[np.newaxis, :], self.z.nEdges, 0)
            segs[:, :, 1] = self.height.edges - np.repeat(self.z.edges[:, np.newaxis], self.x.nEdges, 1)

            ls = LineCollection(segs, color='k', linestyle='solid', **kwargs)
            ax.add_collection(ls)

            dz = 0.02 * np.abs(xtmp.max() - xtmp.min())
            ax.set_xlim(xtmp.min() - dz, xtmp.max() + dz)
            dz = 0.02 * np.abs(zMesh.max() - zMesh.min())
            ax.set_ylim(zMesh.min() - dz, zMesh.max() + dz)


            plt.xscale(xscale)
            plt.yscale(yscale)
            cP.xlabel(xtmp.label)
            cP.ylabel(self.y._centres.label)

            if flipX:
                ax.set_xlim(ax.get_xlim()[::-1])

            if flipY:
                ax.set_ylim(ax.get_ylim()[::-1])


    @property
    def summary(self):
        """ Display a summary of the 3D Point Cloud """
        msg = ("2D Rectilinear Mesh: \n"
              "Shape: : {} \n{}{}{}").format(self.shape, self.x.summary, self.y.summary, self.z.summary)
        if not self.height is None:
            msg += self.height.summary
        return msg


    def plotXY(self, **kwargs):
        """Plot the cell centres in x and y as points"""

        assert self.xyz, Exception("Mesh must be instantiated with three co-ordinates to use plotXY()")

        kwargs['marker'] = kwargs.pop('marker', 'o')
        kwargs['linestyle'] = kwargs.pop('linestyle', 'none')

        self.y.centres.plot(x=self.x.centres, **kwargs)

    def pyvista_mesh(self):
        # Create the spatial reference
        import pyvista as pv

        z = 0.05 * np.minimum(self.x.range, self.y.range)
        x, y, z = np.meshgrid(self.x.edges, self.y.edges, np.r_[0.0, z])

        return pv.StructuredGrid(x, y, z)

    def createHdf(self, parent, name, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)
        self.x.createHdf(grp, 'x', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.y.createHdf(grp, 'y', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        if self.xyz:
            self.z.createHdf(grp, 'z', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        if not self.height is None:
            self.height.createHdf(grp, 'height', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)

        return grp


    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        grp = parent[name]
        self.x.writeHdf(grp, 'x',  withPosterior=withPosterior, index=index)
        self.y.writeHdf(grp, 'y',  withPosterior=withPosterior, index=index)
        if self.xyz:
            self.z.writeHdf(grp, 'z',  withPosterior=withPosterior, index=index)

        if not self.height is None:
            self.height.writeHdf(grp, 'height',  withPosterior=withPosterior, index=index)


    def fromHdf(self, grp, index=None):
        """ Reads in the object from a HDF file """

        RectilinearMesh2D.__init__(self)

        self.x = RectilinearMesh1D.RectilinearMesh1D().fromHdf(grp['x'], index=index)
        self.y = RectilinearMesh1D.RectilinearMesh1D().fromHdf(grp['y'], index=index)
        self._z = self._y

        if 'z' in grp:
            self.z = RectilinearMesh1D.RectilinearMesh1D().fromHdf(grp['z'], index=index)
            self.xyz = True

        if 'height' in grp:
            self._height = RectilinearMesh1D.RectilinearMesh1D().fromHdf(grp['height'], index=index)

        return self


    def range(self, axis):
        return self.axis(axis).range

    def xMesh(self, xAxis='x'):
        """Creates an array suitable for plt.pcolormesh for the abscissa.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'x', the horizontal xAxis uses self.x
            If xAxis is 'y', the horizontal xAxis uses self.y
            If xAxis is 'r', the horizontal xAxis uses cumulative distance along the line.

        """

        # assert xAxis in ['x', 'y', 'r'], Exception("xAxis must be either 'x', 'y' or 'r'")
        if xAxis == 'index':
            xMesh = StatArray.StatArray(np.repeat(np.arange(self.x.nEdges, dtype=np.float64)[np.newaxis, :], self.z.nEdges, 0))
        elif xAxis == 'x':
            xMesh = np.repeat(self.x.edges[np.newaxis, :], self.z.nEdges, 0)
        elif xAxis == 'y':
            assert self.xyz, Exception("To plot against 'y' the mesh must be instantiated with three co-ordinates")
            xMesh = np.repeat(self.y.edges[np.newaxis, :], self.z.nEdges, 0)
        elif xAxis == 'r':
            assert self.xyz, Exception("To plot against 'r' the mesh must be instantiated with three co-ordinates")
            dx = np.diff(self.x.edges)
            dy = np.diff(self.y.edges)
            distance = StatArray.StatArray(np.zeros(self.x.nEdges), 'Distance', self.x.centres.units)
            distance[1:] = np.cumsum(np.sqrt(dx**2.0 + dy**2.0))
            xMesh = np.repeat(distance[np.newaxis, :], self.z.nEdges, 0)

        return xMesh

    @property
    def zMesh(self):
        """Creates an array suitable for plt.pcolormesh for the ordinate """
        return self.height.edges - np.repeat(self.z.edges[:, np.newaxis], self.x.nCells+1, 1)


    def vtkStructure(self):
        """Generates a vtk mesh structure that can be used in a vtk file.

        Returns
        -------
        out : pyvtk.VtkData
            Vtk data structure

        """

        # Generate the quad node locations in x
        x = self.x.edges
        y = self.y.edges
        z = self.z.edges

        nCells = self.x.nCells * self.z.nCells

        z = self.z.edges
        nNodes = self.x.nEdges * self.z.nEdges

        # Constuct the node locations for the vtk file
        nodes = np.empty([nNodes, 3])
        nodes[:, 0] = np.tile(x, self.z.nEdges)
        nodes[:, 1] = np.tile(y, self.z.nEdges)
        nodes[:, 2] = np.repeat(z, self.x.nEdges)

        tmp = np.int32([0, 1, self.x.nEdges+1, self.x.nEdges])
        a = np.ones(self.x.nCells, dtype=np.int32)
        a[0] = 2
        index = (np.repeat(tmp[:, np.newaxis], nCells, 1) + np.cumsum(np.tile(a, self.z.nCells))-2).T

        return VtkData(PolyData(points=nodes, polygons=index))


    def toVTK(self, fileName, pointData=None, cellData=None, format='binary'):
        """Save to a VTK file.

        Parameters
        ----------
        fileName : str
            Filename to save to.
        pointData : geobipy.StatArray or list of geobipy.StatArray, optional
            Data at each node in the mesh. Each entry is saved as a separate
            vtk attribute.
        cellData : geobipy.StatArray or list of geobipy.StatArray, optional
            Data at each cell in the mesh. Each entry is saved as a separate
            vtk attribute.
        format : str, optional
            "ascii" or "binary" format. Ascii is readable, binary is not but results in smaller files.

        Raises
        ------
        TypeError
            If pointData or cellData is not a geobipy.StatArray or list of them.
        ValueError
            If any pointData (cellData) entry does not have size equal to the number of points (cells).
        ValueError
            If any StatArray does not have a name or units. This is needed for the vtk attribute.

        """

        vtk = self.vtkStructure()

        if not pointData is None:
            assert isinstance(pointData, (StatArray.StatArray, list)), TypeError("pointData must a geobipy.StatArray or a list of them.")
            if isinstance(pointData, list):
                for p in pointData:
                    assert isinstance(p, StatArray.StatArray), TypeError("pointData entries must be a geobipy.StatArray")
                    assert all(p.shape == [self.z.nEdges, self.x.nEdges]), ValueError("pointData entries must have shape {}".format([self.z.nEdges, self.x.nEdges]))
                    assert p.hasLabels(), ValueError("StatArray needs a name")
                    vtk.point_data.append(Scalars(p.reshape(self.nNodes), p.getNameUnits()))
            else:
                assert all(pointData.shape == [self.z.nEdges, self.x.nEdges]), ValueError("pointData entries must have shape {}".format([self.z.nEdges, self.x.nEdges]))
                assert pointData.hasLabels(), ValueError("StatArray needs a name")
                vtk.point_data.append(Scalars(pointData.reshape(self.nNodes), pointData.getNameUnits()))

        if not cellData is None:
            assert isinstance(cellData, (StatArray.StatArray, list)), TypeError("cellData must a geobipy.StatArray or a list of them.")
            if isinstance(cellData, list):
                for p in cellData:
                    assert isinstance(p, StatArray.StatArray), TypeError("cellData entries must be a geobipy.StatArray")
                    assert np.all(p.shape == self.shape), ValueError("cellData entries must have shape {}".format(self.shape))
                    assert p.hasLabels(), ValueError("StatArray needs a name")
                    vtk.cell_data.append(Scalars(p.reshape(self.nCells), p.getNameUnits()))
            else:
                assert all(cellData.shape == self.shape), ValueError("cellData entries must have shape {}".format(self.shape))
                assert cellData.hasLabels(), ValueError("StatArray needs a name")
                vtk.cell_data.append(Scalars(cellData.reshape(self.nCells), cellData.getNameUnits()))

        vtk.tofile(fileName, format)
