""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import deepcopy
from importlib import reload

from .Mesh import Mesh
from ...classes.core import StatArray
from .RectilinearMesh1D import RectilinearMesh1D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.stats import binned_statistic
from ...base import plotting as cP
from ...base import utilities
# from ...base import geometry
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
    x : geobipy.RectilinearMesh1D, optional
        text
    y : geobipy.RectilinearMesh1D, optional
        text
    z : geobipy.RectilinearMesh1D, optional
        text
    relativeTo : geobipy.RectilinearMesh1D, optional
        text

    Other Parameters
    ----------------
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

    def __init__(self, x=None, y=None, relativeTo=None, **kwargs):
        """ Initialize a 2D Rectilinear Mesh"""

        self._distance = None
        self.xyz = False
    
        self.x = kwargs if x is None else x
        self.y = kwargs if y is None else y

    def __getitem__(self, slic):
        """Allow slicing of the histogram.

        """
        assert np.shape(slic) == (2,), ValueError("slic must be over 2 dimensions.")

        # slic = []
        axis = -1
        for i, x in enumerate(slic):
            if isinstance(x, (int, np.integer)):
                axis = i

        if axis == -1:
            relativeTo = None
            if self.relativeTo is not None:
                relativeTo = self.x.interpolate_centres_to_nodes(self.relativeTo)[slic[0]]

            out = type(self)(x=self.x[slic[0]], y=self.y[slic[1]], relativeToEdges=relativeTo)
            return out

        out = self.axis(1-axis)[slic[1-axis]]
        return out

    @property
    def area(self):
        return np.outer(self.x.widths, self.y.widths)

    @property
    def centres(self):
        """Ravelled cell centres

        Returns
        -------
        out : array_like
            ravelled cell centre locations.

        """
        out = np.zeros((self.nCells.item(), 2))
        out[:, 0] = np.tile(self.x.centres, self.y.nCells.item())
        out[:, 1] = self.y.centres.repeat(self.x.nCells.item())
        return out

    @property
    def distance(self):
        """The distance along the top of the mesh using the x and y co-ordinates. """
        if self._distance is None:

            if ~self.xyz:
                self._distance = self.x
            else:
                dx = np.diff(self.x.edges)
                dy = np.diff(self.y.edges)

                distance = StatArray.StatArray(np.zeros(self.x.nEdges), 'Distance', self.x.centres.units)
                distance[1:] = np.cumsum(np.sqrt(dx**2.0 + dy**2.0))

                self._distance = RectilinearMesh1D(edges = distance)
        return self._distance

    @property
    def height(self):
        return self._height

    # @property
    # def relativeTo(self):
    #     return self._relativeTo

    # @relativeTo.setter
    # def relativeTo(self, values):

    #     self._relativeTo = None
    #     if not values is None:
    #         self._relativeTo = StatArray.StatArray(values, "relativeTo", "m")

    @property
    def nCells(self):
        """The number of cells in the mesh.

        Returns
        -------
        out : int
            Number of cells

        """

        return self.x.nCells * self.y.nCells

    @property
    def ndim(self):
        return 2

    @property
    def nNodes(self):
        """The number of nodes in the mesh.

        Returns
        -------
        out : int
            Number of nodes

        """

        return self.x.nEdges * self.y.nEdges

    @property
    def nodes(self):
        """Ravelled cell nodes

        Returns
        -------
        out : array_like
            ravelled cell node locations.

        """
        out = np.zeros((self.nNodes.value, 2))
        out[:, 0] = np.tile(self.x.edges, self.y.nNodes.value)
        out[:, 1] = self.y.edges.repeat(self.x.nNodes.value)
        return out

    @property
    def shape(self):
        """The dimensions of the mesh

        Returns
        -------
        out : array_like
            Array of integers

        """

        return (self.x.nCells.item(), self.y.nCells.item())


    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, values):
        if isinstance(values, dict):
            # mesh of the z axis values
            values = RectilinearMesh1D(
                        centres=values.get('xCentres'),
                        edges=values.get('xEdges'),
                        log=values.get('xlog'),
                        relativeTo=values.get('xrelativeTo'))
        assert isinstance(values, RectilinearMesh1D), TypeError('x must be a RectilinearMesh1D')
        self._x = values

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, values):
        if isinstance(values, dict):
            # mesh of the z axis values
            values = RectilinearMesh1D(
                        centres=values.get('yCentres'),
                        edges=values.get('yEdges'),
                        log=values.get('ylog'),
                        relativeTo=values.get('yrelativeTo'))
        assert isinstance(values, RectilinearMesh1D), TypeError('y must be a RectilinearMesh1D')
        self._y = values

    def _animate(self, values, axis, filename, slic=None, **kwargs):
        
        fig = kwargs.pop('fig', plt.figure(figsize=(9, 9)))

        if slic is None:
            slic = [np.s_[:] for i in range(self.ndim)]
        else:
            slic = list(slic)

        slic[axis] = 0

        # Do the first slice
        sub = self[tuple(slic)]
        sub_v = values[tuple(slic)]

        sub.bar(values=sub_v, **kwargs)
        plt.xlim(sub.displayLimits)
        plt.ylim([np.min(values), np.max(values)])

        # tmp, _ = utilities._log(values, kwargs.get('log', None))
        # plt.set_clim(np.min(tmp), np.max(tmp))

        def animate(i):
            ax = self.axis(axis).centres
            plt.clf()
            plt.title('{} = {:.2f} {}'.format(ax.name, ax[i], ax.units))
            slic[axis] = i
            # tmp, _ = utilities._log(values[tuple(slic)].flatten(), kwargs.get('log', None))
            sub.bar(values=values[tuple(slic)], **kwargs)
            plt.xlim(sub.displayLimits)
            plt.ylim([np.min(values), np.max(values)])

        anim = FuncAnimation(fig, animate, interval=300, frames=self.axis(axis).nCells.item())

        plt.draw()
        anim.save(filename)

    # def _mean(self, arr, log=None, axis=0):
    #     a = self.axis(axis)
    #     b = self.other_axis(axis)

    #     t = np.sum(np.repeat(np.expand_dims(b.centres, axis), a.nCells, axis) * arr, 1-axis)
    #     s = arr.sum(axis = 1 - axis)

    #     i = np.where(s > 0.0)[0]
    #     out = np.zeros(t.size)
    #     out[i] = t[i] / s[i]

    #     if log:
    #         out, dum = utilities._log(out, log=log)

    #     return out

    

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
        return self._percentile(values=values, percent=50.0, log=log, axis=axis)

    def __deepcopy__(self, memo={}):
        """ Define the deepcopy for the StatArray """
        return RectilinearMesh2D(x=self.x, y=self.y)

    def edges(self, axis):
        """ Gets the cell edges in the given dimension """
        return self.axis(axis).edges

    def other_axis(self, axis):
        return self.axis(1-axis)

    def axis(self, axis):
        if isinstance(axis, str):
            assert axis in ['x', 'y'], Exception("axis must be either 'x', 'y'")

            if axis == 'x':
                return self.x
            elif axis == 'y':
                # assert self.xyz, Exception("To plot against 'y' the mesh must be instantiated with three co-ordinates")
                return self.y
            # elif axis == 'z':
            #     return self.z
            # elif axis == 'r':
            #     assert self.xyz, Exception("To plot against 'r' the mesh must be instantiated with three co-ordinates")
            #     return self.distance

        if axis == 0:
            return self.x
        elif axis == 1:
            return self.y

    @property
    def centres_bounds(self):
        return np.r_[self.x.centres[0], self.x.centres[-1], self.y.centres[0], self.y.centres[-1]]

    @property
    def bounds(self):
        return np.r_[self.x.bounds[0], self.x.bounds[1], self.y.bounds[0], self.y.bounds[1]]

    def in_bounds(self, x, y):
        """Return whether values are inside the cell edges

        Parameters
        ----------
        values : array_like
            Check if these are inside left <= values < right.

        Returns
        -------
        out : bools
            Are the values inside.

        """
        return self.x.in_bounds(x) & self.y.in_bounds(y)

    # def project_line_to_box(self, x, y):
    #     x ,_ = utilities._log(x, self.x.log)
    #     y ,_ = utilities._log(y, self.y.log)
    #     x, y = geometry.liang_barsky(x[0], y[0], x[1], y[1], self.bounds)
    #     x = utilities._power(x, self.x.log)
    #     y = utilities._power(y, self.y.log)
    #     return x, y

    def xGradientMatrix(self):
        tmp = self.x.gradientMatrix()
        return kron(diags(np.sqrt(self.y.widths)), tmp)


    def yGradientMatrix(self):
        nx = self.x.nCells.item()
        nz = self.y.nCells.item()
        tmp = 1.0 / np.sqrt(self.x.centreTocentre)
        a = np.repeat(tmp, nz) * np.tile(np.sqrt(self.y.widths), nx-1)
        return diags([a, -a], [0, nx], shape=(nz * (nx-1), nz*nz))


    def hasSameSize(self, other):
        """ Determines if the meshes have the same dimension sizes """
        # if self.arr.shape != other.arr.shape:
        #     return False
        if self.x.nCells != other.x.nCells:
            return False
        if self.y.nCells != other.y.nCells:
            return False
        return True

    @property
    def is_regular(self):
        return self.x.is_regular and self.y.is_regular

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

        x = deepcopy(self.x)
        x.edges = np.arange(self.x.edges[0], self.x.edges[-1]+dx, dx)
        y = deepcopy(self.y)
        y.edges = np.arange(self.y.edges[0], self.y.edges[-1]+dy, dy)

        z = None
        if self.xyz:
            z = y
            y = deepcopy(self.y)
            y.edges = np.arange(self.y.edges[0], self.y.edges[-1]+dx, dx)

        relativeTo = None
        if not self.relativeTo is None:
            relativeTo = self[:, 0].resample(dx, self.relativeTo)

        mesh = RectilinearMesh2D(x=x, y=y)#, relativeTo = relativeTo)

        f = interpolate.interp2d(self.y.centres, self.x.centres, values, kind=kind)
        return mesh, f(mesh.y.centres, mesh.x.centres)

    def remove_axis(self, axis):
        return self.other_axis(axis)
        
    def interpolate_centres_to_nodes(self, values, kind='cubic'):
        if self.x.nCells <= 3 or self.y.nCells <= 3:
            kind = 'linear'
        f = interpolate.interp2d(self.y.centres, self.x.centres, values, kind=kind)
        return f(self.y.edges, self.x.edges)

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
            bins = binned_statistic(self.x.centres, arr.T, bins = intervals, statistic=statistic)
            res = bins.statistic.T
        else:
            bins = binned_statistic(self.y.centres, arr, bins = intervals, statistic=statistic)
            res = bins.statistic

        return res, intervals


    def mask_cells(self, axis=None, x_distance=None, y_distance=None, values=None):
        """Mask cells by a distance.

        If the edges of the cell are further than distance away, extra cells are inserted such that
        the cell's new edges are at distance away from the centre.

        Parameters
        ----------
        xAxis : array_like
            Alternative axis to use for masking.  Must have size self.x.nEdges
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
        x = self.x
        if not x_distance is None:
            x, x_indices = self.x.mask_cells(x_distance)
            if not values is None:
                out_values = np.full((x.nCells.item(), self.y.nCells.item()), fill_value=np.nan)
                for i in range(self.x.nCells.item()):
                    out_values[x_indices[i], :] = values[i, :]

        y_indices = None
        y = self.y
        if not y_distance is None:
            y, y_indices = self.y.mask_cells(y_distance)
            if not values is None:
                out_values2 = np.full((out_values.shape[0], y.nCells.item()), fill_value=np.nan)
                for i in range(self.y.nCells.item()):
                    out_values2[:, y_indices[i]] = out_values[:, i]
                out_values = out_values2

        y_relativeTo = None
        if self.y.relativeTo is not None:
            re = self[0, :].interpolate_centres_to_nodes(self.y.relativeTo)
            y_relativeTo = np.interp(x.edges, self.x.edges, re)

        out = type(self)(x=x, y=y, yrelativeTo = y_relativeTo)

        return out, x_indices, y_indices, out_values


    def _reconcile_intervals(self, intervals, axis=0):

        assert np.size(intervals) > 1, ValueError("intervals must have size > 1")

        ax = self.other_axis(axis)

        i0 = np.maximum(0, np.searchsorted(intervals, ax.edges[0]))
        i1 = np.minimum(ax.nCells.item(), np.searchsorted(intervals, ax.edges[-1])+1)

        intervals = intervals[i0:i1]

        return intervals

    def _reorder_for_pyvista(self, values):
        return values.flatten(order='C')

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

    def cellIndices(self, x, y=None, clip=False, trim=False):
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
        if np.ndim(x) == 2:
            x = x[:, 0]
            y = x[:, 1]

        assert (np.size(x) == np.size(y)), ValueError("x and y must have the same size")
        if trim:
            flag = self.x.inBounds(x) & self.y.inBounds(y)
            i = np.where(flag)[0]
            out = np.empty([2, i.size], dtype=np.int32)
            out[0, :] = self.x.cellIndex(x[i])
            out[1, :] = self.y.cellIndex(y[i])
        else:
            out = np.empty([2, np.size(x)], dtype=np.int32)
            out[0, :] = self.x.cellIndex(x, clip=clip)
            out[1, :] = self.y.cellIndex(y, clip=clip)
        return np.squeeze(out)

    def line_indices(self, x, y):
        i = self.cellIndices(x, y)
        return utilities.bresenham(i[0, :], i[1, :])

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

    def pcolor(self, values, axis=None, yAxis='absolute', **kwargs):
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
            If zAxis is 'absolute' the vertical axis is the relativeTo plus z.
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
        y_mask = kwargs.pop('y_mask', None)

        if (self.x._relativeTo is None) and (self.y._relativeTo is None):
            # cP.pcolor(values, x=self.x.edges_absolute, y=self.y.edges_absolute, **kwargs)
            masked = self
            if np.sum([x is None for x in [x_mask, y_mask]]) < 2:
                masked, x_indices, z_indices, values = self.mask_cells(axis, x_mask, y_mask, values)
                xAxis='x'

            xm = masked.xMesh('x')
            ym = masked.yMesh

            if self.x.log is not None:
                kwargs['xscale'] = 'log'
            if self.y.log is not None:
                kwargs['yscale'] = 'log'

            ax, pm, cb = cP.pcolormesh(xm, ym, values.T, **kwargs)
            cP.xlabel(xm.label)
            cP.ylabel(ym.label)

            return ax, pm, cb


        else:
            # Need to expand the yaxis edges since they could be draped.
            if (x_mask is not None) or (y_mask is not None):
                masked, x_indices, z_indices, values = self.mask_cells(axis, x_mask, y_mask, values)
                ax, pm, cb = cP.pcolor(values, x = masked.axis('x').edges, y = masked.z.edges, **kwargs)
            else:
                x = self.xMesh()
                y = self.yMesh

                if self.x.log is not None:
                    kwargs['xscale'] = 'log'
                if self.y.log is not None:
                    kwargs['yscale'] = 'log'

                if y.shape[0] != x.shape[0]:
                    x = x.T

                if np.all(values.shape == np.asarray(x.shape[::-1]) - 1):
                    values = values.T

                ax, pm, cb = cP.pcolor(x=x, y=y, values=values, **kwargs)

        return ax, pm, cb


    def plotGrid(self, **kwargs):
        """Plot the mesh grid lines.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'x', the horizontal axis uses self.x
            If xAxis is 'y', the horizontal axis uses self.y
            If xAxis is 'r', the horizontal axis uses sqrt(self.x^2 + self.y^2)

        """
        kwargs['xscale'] = kwargs.pop('xscale', 'linear' if self.x.log is None else 'log')
        kwargs['yscale'] = kwargs.pop('yscale', 'linear' if self.y.log is None else 'log')

        if (self.x._relativeTo is None) and (self.y._relativeTo is None):
    
            tmp = StatArray.StatArray(np.full(self.shape, fill_value=np.nan)).T
            tmp.pcolor(x=self.x.edges_absolute, y=self.y.edges_absolute, grid=True, colorbar=False, **kwargs)

        else:
            xscale = kwargs.pop('xscale')
            yscale = kwargs.pop('yscale')
            flipX = kwargs.pop('flipX', False)
            flipY = kwargs.pop('flipY', False)
            c = kwargs.pop('color', 'k')

            ax = plt.gca()
            cP.pretty(ax)

            x_mesh = self.xMesh()
            y_mesh = self.yMesh
            if y_mesh.shape[0] == x_mesh.shape[0]:
                a = np.dstack([x_mesh, y_mesh])
                b = np.dstack([x_mesh.T, y_mesh.T])
            else:
                a = np.dstack([x_mesh, y_mesh.T])
                b = np.dstack([x_mesh.T, y_mesh])


            ls = LineCollection(a, color='k', linestyle='solid', **kwargs)
            ax.add_collection(ls)

            ls = LineCollection(b, color='k', linestyle='solid', **kwargs)
            ax.add_collection(ls)

            dz = 0.02 * np.abs(x_mesh.max() - x_mesh.min())
            ax.set_xlim(x_mesh.min() - dz, x_mesh.max() + dz)
            dz = 0.02 * np.abs(y_mesh.max() - y_mesh.min())
            ax.set_ylim(y_mesh.min() - dz, y_mesh.max() + dz)

            plt.xscale(xscale)
            plt.yscale(yscale)
            cP.xlabel(x_mesh.label)
            cP.ylabel(self.y._centres.label)

            if flipX:
                ax.set_xlim(ax.get_xlim()[::-1])

            if flipY:
                ax.set_ylim(ax.get_ylim()[::-1])

    def plot_relative_to(self, centres=False, **kwargs):
        """Plot the relativeTo of the mesh as a line. """

        kwargs['c'] = kwargs.pop('color', 'k')
        kwargs['linewidth'] = kwargs.pop('linewidth', 1.0)

        if centres:
            self.relativeTo.plot(self.x.centres, **kwargs)
        else:
            re = self[:, 0].interpolate_centres_to_nodes(self.relativeTo)
            re.plot(self.x.edges, **kwargs)

    def plot_line(self, value, axis=0, **kwargs):

        c = kwargs.pop('color', '#5046C8')
        ls = kwargs.pop('linestyle', 'dashed')
        lw = kwargs.pop('linewidth', 2)
        a = kwargs.pop('alpha', 0.6)

        if axis == 0:
            cP.plot(value, self.y.centres, color=c, linestyle=ls,
                    linewidth=lw, alpha=a, **kwargs)
        else:
            cP.plot(self.x.centres, value, color=c, linestyle=ls,
                    linewidth=lw, alpha=a, **kwargs)

    @property
    def summary(self):
        """ Display a summary of the 3D Point Cloud """
        msg = ("2D Rectilinear Mesh: \n"
              "Shape: : {} \nx\n{}y\n{}").format(self.shape, self.x.summary, self.y.summary)
        # if not self.relativeTo is None:
        #     msg += self.relativeTo.summary
        return msg

    # def plotXY(self, **kwargs):
    #     """Plot the cell centres in x and y as points"""

    #     assert self.xyz, Exception("Mesh must be instantiated with three co-ordinates to use plotXY()")

    #     kwargs['marker'] = kwargs.pop('marker', 'o')
    #     kwargs['linestyle'] = kwargs.pop('linestyle', 'none')

    #     self.y.centres.plot(x=self.x.centres, **kwargs)

    def pyvista_mesh(self):
        # Create the spatial reference
        import pyvista as pv

        z = 0.05 * np.minimum(self.x.range, self.y.range)
        x, y, z = np.meshgrid(self.x.edges, self.y.edges, np.r_[0.0, z])

        return pv.StructuredGrid(x, y, z)

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        if add_axis is not None:
            return self._create_hdf_3d(parent, name, withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        grp = self.create_hdf_group(parent, name)
        self.x.createHdf(grp, 'x', withPosterior=withPosterior, fillvalue=fillvalue)
        self.y.createHdf(grp, 'y', withPosterior=withPosterior, fillvalue=fillvalue)
        if not self.relativeTo is None:
            self.relativeTo.createHdf(grp, 'relativeTo', withPosterior=withPosterior, fillvalue=fillvalue)

        return grp

    def _create_hdf_3d(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        from .RectilinearMesh3D import RectilinearMesh3D
        if isinstance(add_axis, (int, np.int_)):
            x = np.arange(add_axis, dtype=np.float64)
        else:
            x = add_axis
        x = RectilinearMesh1D(centres=x)

        relativeTo = None if self._relativeTo is None else np.zeros((x.nCells, self.x.nCells))

        mesh = RectilinearMesh3D(x=x, y=self.x, z=self.y, relativeTo=relativeTo)

        out = mesh.createHdf(parent, name, withPosterior=withPosterior, fillvalue=fillvalue)
        x.writeHdf(out, 'x')

        return out

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        if index is not None:
            return self._write_hdf_3d(parent, name, index, withPosterior=withPosterior)

        grp = parent[name]
        self.x.writeHdf(grp, 'x',  withPosterior=withPosterior)
        self.y.writeHdf(grp, 'y',  withPosterior=withPosterior)

        if not self.relativeTo is None:
            self.relativeTo.writeHdf(grp, 'relativeTo',  withPosterior=withPosterior)

    def _write_hdf_3d(self, parent, name, index, withPosterior=True):
        grp = parent[name]
        assert '3D' in grp.attrs['repr'], TypeError("HDF creation must have an axis added.")

        self.x.writeHdf(grp, 'y',  withPosterior=withPosterior)
        self.y.writeHdf(grp, 'z',  withPosterior=withPosterior)

        if not self.relativeTo is None:
            self.relativeTo.writeHdf(grp, 'relativeTo',  withPosterior=withPosterior, index=index)


    @classmethod
    def fromHdf(cls, grp, index=None, skip_posterior=False):
        # from .RectilinearMesh3D import RectilinearMesh3D

        if 'stitched' in grp.attrs['repr']:
            from .RectilinearMesh2D_stitched import RectilinearMesh2D_stitched
            return RectilinearMesh2D_stitched.fromHdf(grp, index, skip_posterior=skip_posterior)

        if '3D' in grp.attrs['repr']:
            if index is None:
                assert False, ValueError("RectilinearMesh2D cannot be read from a RectilinearMesh3D without an index")
                # return RectilinearMesh3D.fromHdf(grp)

            else: # Read a 2D mesh from 3D
                x = RectilinearMesh1D.fromHdf(grp['y'], skip_posterior=skip_posterior)
                y = RectilinearMesh1D.fromHdf(grp['z'], skip_posterior=skip_posterior)

                relativeTo = None
                if 'relativeTo' in grp:
                    relativeTo = StatArray.StatArray.fromHdf(grp['relativeTo'], index=index, skip_posterior=skip_posterior)

                out = cls(x=x, y=y)
                out._relativeTo = relativeTo
                return out
        else:
            if index is not None:

                return RectilinearMesh1D.fromHdf(grp, index=index, skip_posterior=skip_posterior)
            else:
                x = RectilinearMesh1D.fromHdf(grp['x'], skip_posterior=skip_posterior)
                y = RectilinearMesh1D.fromHdf(grp['y'], skip_posterior=skip_posterior)
                relativeTo = None
                if 'relativeTo' in grp:
                    relativeTo = StatArray.StatArray.fromHdf(grp['relativeTo'], skip_posterior=skip_posterior)

                out = cls(x=x, y=y)
                out._relativeTo = relativeTo
                return out

    def fromHdf_cell_values(self, grp, key, index=None, skip_posterior=False):
        return StatArray.StatArray.fromHdf(grp, key, index=index, skip_posterior=skip_posterior)
            

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
            x_mesh = StatArray.StatArray(np.repeat(np.arange(self.x.nEdges, dtype=np.float64)[:, None], self.y.nEdges, 1))
        elif xAxis == 'x':
            if self.x._relativeTo is None:
                x_mesh = np.repeat(self.x.edges_absolute[None, :], self.y.nEdges, 0)
            else:
                if self.x.relativeTo.size == 1:
                    x_mesh = np.repeat(self.x.edges_absolute[None, :], self.y.nEdges, 0)
                else:
                    re = self.y.interpolate_centres_to_nodes(self.x.relativeTo)
                    edges = np.repeat(re[:, None], self.x.nEdges, 1) + self.x.edges
                    x_mesh = utilities._power(edges, self.x.log)

        # elif xAxis == 'y':
        #     assert self.xyz, Exception("To plot against 'y' the mesh must be instantiated with three co-ordinates")
        #     xMesh = np.repeat(self.y.edges[np.newaxis, :], self.z.nEdges, 0)
        # elif xAxis == 'r':
        #     assert self.xyz, Exception("To plot against 'r' the mesh must be instantiated with three co-ordinates")
        #     dx = np.diff(self.x.edges)
        #     dy = np.diff(self.y.edges)
        #     distance = StatArray.StatArray(np.zeros(self.x.nEdges), 'Distance', self.x.centres.units)
        #     distance[1:] = np.cumsum(np.sqrt(dx**2.0 + dy**2.0))
        #     xMesh = np.repeat(distance[np.newaxis, :], self.z.nEdges, 0)

        return x_mesh

    @property
    def yMesh(self):
        """Creates an array suitable for plt.pcolormesh for the ordinate """
        if self.y._relativeTo is None:
            y_mesh = np.repeat(self.y.edges_absolute[:, None], self.x.nEdges, 1)
        else:
            re = self.x.interpolate_centres_to_nodes(self.y.relativeTo, kind='linear')
            edges = np.repeat(re[:, None], self.y.nEdges, 1) + self.y.edges
            y_mesh = utilities._power(edges, self.y.log)

        return y_mesh