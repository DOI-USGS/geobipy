""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import deepcopy

from numpy import abs, arange, asarray
from numpy import cumsum, diff, dot, dstack, empty, expand_dims, float32, float64, full, int_, int32, integer, interp, isnan
from numpy import max, maximum, meshgrid, min, minimum, nan, ndim, outer, r_, ravel_multi_index
from numpy import repeat, s_, searchsorted, shape, size, sqrt, squeeze, tile, unravel_index
from numpy import where, zeros
from numpy import all as npall

from .Mesh import Mesh
from ..core.DataArray import DataArray
from ..statistics.StatArray import StatArray
from .RectilinearMesh1D import RectilinearMesh1D
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from ...base import plotting as cP
from ...base import utilities
from scipy.sparse import (kron, diags)
from scipy import interpolate
import progressbar

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

    RectilinearMesh2D([x_centres or x_edges], [y_centres or y_edges], [z_centres or z_edges])

    Parameters
    ----------
    x : geobipy.RectilinearMesh1D, optional
        text
    y : geobipy.RectilinearMesh1D, optional
        text
    z : geobipy.RectilinearMesh1D, optional
        text
    relative_to : geobipy.RectilinearMesh1D, optional
        text

    Other Parameters
    ----------------
    x_centres : geobipy.StatArray, optional
        The locations of the centre of each cell in the "x" direction. Only x_centres or x_edges can be given.
    x_edges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges, in the "x" direction. Only x_centres or x_edges can be given.
    y_centres : geobipy.StatArray, optional
        The locations of the centre of each cell in the "y" direction. Only y_centres or y_edges can be given.
    y_edges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges, in the "y" direction. Only y_centres or y_edges can be given.
    z_centres : geobipy.StatArray, optional
        The locations of the centre of each cell in the "z" direction. Only z_centres or z_edges can be given.
    z_edges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges, in the "z" direction. Only z_centres or z_edges can be given.
    [x, y, z]edgesMin : float, optional
        See geobipy.RectilinearMesh1D for edgesMin description.
    [x, y, z]edgesMax : float, optional
        See geobipy.RectilinearMesh1D for edgesMax description.
    [x, y, z]log : 'e' or float, optional
        See geobipy.RectilinearMesh1D for log description.
    [x, y, z]relative_to : float, optional
        See geobipy.RectilinearMesh1D for relative_to description.

    Returns
    -------
    out : RectilinearMesh2D
        The 2D mesh.

    """
    x_axis = 0
    y_axis = 1

    def __init__(self, x=None, y=None, **kwargs):
        """ Initialize a 2D Rectilinear Mesh"""

        self._distance = None
        self.xyz = False

        self.x = kwargs if x is None else x
        self.y = kwargs if y is None else y

        self.check_x_relative_to()
        self.check_y_relative_to()


    def __getitem__(self, slic):
        """Allow slicing of the histogram.

        """
        assert shape(slic) == (2,), ValueError("slic must be over 2 dimensions.")

        # slic = []
        axis = -1
        for i, x in enumerate(slic):
            if isinstance(x, (int, integer)):
                axis = i

        if axis == -1:
            out = type(self)(x=self.x[slic[0]], y=self.y[slic[1]])
            if self.x._relative_to is not None:
                if self.x._relative_to.size > 1:
                    out.x.relative_to = self.x._relative_to[slic[1]]
                else:
                    out.x.relative_to = self.x._relative_to
            if self.y._relative_to is not None:
                if self.y._relative_to.size > 1:
                    out.y.relative_to = self.y._relative_to[slic[0]]
                else:
                    out.y.relative_to = self.y._relative_to
            return out

        out = self.axis(1-axis)[slic[1-axis]]
        return out

    def check_x_relative_to(self):
        if self.x.relative_to is not None:
            if ndim(self.x.relative_to) == 1 and self.x.relative_to.size > 1:
                assert self.x.relative_to.size == self.y.size, ValueError(f"1D relative_to on x needs size {self.y.size}")

    def check_y_relative_to(self):
        if self.y.relative_to is not None:
            if ndim(self.y.relative_to) == 1 and self.y.relative_to.size > 1:
                assert self.y.relative_to.size == self.x.size, ValueError(f"1D relative_to on y needs size {self.x.size}")


    @property
    def addressof(self):
        msg =  '{}: {}\n'.format(type(self).__name__, hex(id(self)))
        msg += "x:\n{}".format(("|   "+self.x.addressof.replace("\n", "\n|   "))[:-4])
        msg += "y:\n{}".format(("|   "+self.y.addressof.replace("\n", "\n|   "))[:-4])
        return msg

    @property
    def area(self):
        return outer(self.x.widths, self.y.widths)

    def centres(self, axis=0):
        """Ravelled cell centres

        Returns
        -------
        out : array_like
            ravelled cell centre locations.

        """
        return self.x_centres if axis == 0 else self.y_centres

    @property
    def distance(self):
        """The distance along the top of the mesh using the x and y co-ordinates. """
        if self._distance is None:

            if not self.xyz:
                self._distance = self.x
            else:
                dx = diff(self.x.edges)
                dy = diff(self.y.edges)

                distance = DataArray(zeros(self.x.nEdges), 'Distance', self.x.centres.units)
                distance[1:] = cumsum(sqrt(dx**2.0 + dy**2.0))

                self._distance = RectilinearMesh1D(edges = distance)
        return self._distance

    @property
    def height(self):
        return self._height

    # @property
    # def relative_to(self):
    #     return self._relative_to

    # @relative_to.setter
    # def relative_to(self, values):

    #     self._relative_to = None
    #     if not values is None:
    #         self._relative_to = DataArray(values, "relative_to", "m")

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
        out = zeros((self.nNodes.item(), 2))
        out[:, 0] = tile(self.x.edges, self.y.nNodes.value)
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
                        centres=values.get('x_centres'),
                        edges=values.get('x_edges'),
                        log=values.get('x_log'),
                        relative_to=values.get('x_relative_to'),
                        dimension=self.x_axis)

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
                        centres=values.get('y_centres'),
                        edges=values.get('y_edges'),
                        log=values.get('y_log'),
                        relative_to=values.get('y_relative_to'),
                        dimension=self.y_axis)
        assert isinstance(values, RectilinearMesh1D), TypeError('y must be a RectilinearMesh1D')
        self._y = values

    def _animate(self, values, axis, filename, slic=None, **kwargs):

        fig = kwargs.pop('fig', plt.figure(figsize=(9, 9)))

        if slic is None:
            slic = [s_[:] for i in range(self.ndim)]
        else:
            slic = list(slic)

        slic[axis] = 0

        # Do the first slice
        sub = self[tuple(slic)]
        sub_v = values[tuple(slic)]

        sub.bar(values=sub_v, **kwargs)
        plt.xlim(sub.displayLimits)
        plt.ylim([min(values), max(values)])

        # tmp, _ = utilities._log(values, kwargs.get('log', None))
        # plt.set_clim(min(tmp), max(tmp))

        def animate(i):
            ax = self.axis(axis).centres
            plt.clf()
            plt.title('{} = {:.2f} {}'.format(ax.name, ax[i], ax.units))
            slic[axis] = i
            # tmp, _ = utilities._log(values[tuple(slic)].flatten(), kwargs.get('log', None))
            sub.bar(values=values[tuple(slic)], **kwargs)
            plt.xlim(sub.displayLimits)
            plt.ylim([min(values), max(values)])

        anim = FuncAnimation(fig, animate, interval=300, frames=self.axis(axis).nCells.item())

        plt.draw()
        anim.save(filename)

    def _compute_probability(self, distribution, pdf, log, log_probability, axis=0, **kwargs):
        centres = self.centres(axis=axis)
        centres, _ = utilities._log(centres, log)

        ax = self.other_axis(axis)

        shp = list(self.shape)

        n_dim = distribution.ndim
        if distribution.multivariate:
            n_dim = 1
        shp[axis] = n_dim
        probability = zeros(shp)

        track = kwargs.pop('track', True)

        r = range(ax.nCells.item())
        if track:
            Bar = progressbar.ProgressBar()
            r = Bar(r)

        # Loop over the axis and compute the probability of each dimension in the
        # distribution with the pdf of the histogram
        for i in r:
            j = [i]; j.insert(axis, s_[:]); j = tuple(j)
            p = distribution.probability(centres[j], log_probability)
            probability[j] = dot(p, pdf[j])

        # Normalize probabilities along the dims of the distribution
        if n_dim > 1:
            probability = probability / expand_dims(sum(probability, axis), axis=axis)

        mesh = deepcopy(self)
        if n_dim > 1:
            mesh.set_axis(axis, RectilinearMesh1D(centres=DataArray(np.arange(n_dim), name='component')))
        else:
            mesh = mesh.remove_axis(axis)

        from ..model.Model import Model
        return Model(mesh=mesh, values=DataArray(squeeze(probability.T), name='Marginal Probability'))

    def __deepcopy__(self, memo={}):
        """ Define the deepcopy for the StatArray """
        return RectilinearMesh2D(x=self.x, y=self.y)

    def add_axis(self, axis, ax=None, **kwargs):
        from .RectilinearMesh3D import RectilinearMesh3D

        assert 0 <= axis <= 2, ValueError("Invalid axis 0 <= axis <= 2")

        if ax is None:
            ax = RectilinearMesh1D(**kwargs)

        match axis:
            case 0:
                self.x.dimension += 1
                self.y.dimension += 1
                return RectilinearMesh3D(x=ax, y=self.x, z=self.y)
            case 1:
                self.y.dimension += 1
                return RectilinearMesh3D(x=self.x, y=ax, z=self.y)
            case 2:
                return RectilinearMesh3D(x=self.x, y=self.y, z=ax)

    def edges(self, axis):
        """ Gets the cell edges in the given dimension """
        if axis == 0:
            return self.x_edges
        return self.y_edges

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

    def set_axis(self, axis, value):
        if axis == 0:
            self.x = value
        elif axis == 1:
            self.y = value

    @property
    def centres_bounds(self):
        return r_[self.x.centres[0], self.x.centres[-1], self.y.centres[0], self.y.centres[-1]]

    @property
    def bounds(self):
        return r_[self.x.bounds[0], self.x.bounds[1], self.y.bounds[0], self.y.bounds[1]]

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

    def x_gradient_operator(self):
        tmp = self.x.gradient_operator
        return kron(diags(sqrt(self.y.widths)), tmp)


    def y_gradient_operator(self):
        nx = self.x.nCells.item()
        nz = self.y.nCells.item()
        tmp = 1.0 / sqrt(self.x.centreTocentre)
        a = repeat(tmp, nz) * tile(sqrt(self.y.widths), nx-1)
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

    def resample(self, dx, dy, values, method='cubic'):

        x = deepcopy(self.x); y = deepcopy(self.y)

        if isinstance(dx, (float, float32, float64)):
            x.edges = arange(self.x.edges[0], self.x.edges[-1]+dx, dx)
        else:
            x.centres = dx

        if isinstance(dy, (float, float32, float64)):
            y.edges = arange(self.y.edges[0], self.y.edges[-1]+dy, dy)
        else:
            y.centres = dy

        # z = None
        # if self.xyz:
        #     z = y
        #     y = deepcopy(self.y)
        #     y.edges = arange(self.y.edges[0], self.y.edges[-1]+dx, dx)

        mesh = RectilinearMesh2D(x=x, y=y)

        if self.x._relative_to is not None:
            mesh.x.relative_to = self.y.resample(dx, self.x.relative_to)
        if self.y._relative_to is not None:
            mesh.y.relative_to = self.x.resample(dy, self.y.relative_to)

        f = interpolate.RegularGridInterpolator((self.x.centres, self.y.centres), values, method=method, bounds_error=False)

        xx, yy = meshgrid(mesh.x.centres, mesh.y.centres, indexing='ij', sparse=True)

        return mesh, f((xx, yy))

    def interpolate_centres_to_nodes(self, values, method='cubic'):
        if self.x.nCells <= 3 or self.y.nCells <= 3:
            method = 'linear'

        f = interpolate.RegularGridInterpolator((self.x.centres, self.y.centres), values, method=method, bounds_error=False)
        xx, yy = meshgrid(self.x.edges, self.y.edges, indexing='ij', sparse=True)

        out = f((xx, yy))

        out[0, 1:-1] = out[1, 1:-1] - (abs(out[2, 1:-1] - out[1, 1:-1]))
        out[-1, 1:-1] = out[-2, 1:-1] - (abs(out[-3, 1:-1] - out[-2, 1:-1]))
        out[:, 0] = out[:, 1] - (abs(out[:, 2] - out[:, 1]))
        out[:, -1] = out[:, -2] - (abs(out[:, -3] - out[:, 2]))

        return out

    def fill_nans_with_extrapolation(self, values, **kwargs):

        from ..pointcloud.Point import Point

        if npall(values.shape[::-1] == self.shape):
            values = values.T
        assert npall(values.shape == self.shape), ValueError("values must have shape {} but have shape {}".format(self.shape, values.shape))

        i = ~isnan(values)
        x = self.x_centres[i]
        y = self.y_centres[i]
        v = values[i]

        p2d = Point(x, y, z=v)

        out, _ = p2d.interpolate(values=v, mesh=self, method='sibson', **kwargs)
        return out


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

        assert size(intervals) > 1, ValueError("intervals must have size > 1")

        intervals = self._reconcile_intervals(intervals, axis=axis)

        if (axis == 0):
            bins = binned_statistic(self.x.centres, arr.T, bins = intervals, statistic=statistic)
            res = bins.statistic.T
        else:
            bins = binned_statistic(self.y.centres, arr, bins = intervals, statistic=statistic)
            res = bins.statistic

        return res, intervals


    def mask_cells(self, x_distance=None, y_distance=None, values=None):
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

        x_indices = s_[:]
        x = deepcopy(self.x)
        if not x_distance is None:
            x, x_indices = self.x.mask_cells(x_distance)
            if not values is None:
                out_values = full((x.nCells.item(), self.y.nCells.item()), fill_value=nan)
                for i in range(self.x.nCells.item()):
                    out_values[x_indices[i], :] = values[i, :]

        y_indices = s_[:]
        y = deepcopy(self.y)
        if not y_distance is None:
            y, y_indices = self.y.mask_cells(y_distance)
            if not values is None:
                out_values2 = full((out_values.shape[0], y.nCells.item()), fill_value=nan)
                for i in range(self.y.nCells.item()):
                    out_values2[:, y_indices[i]] = out_values[:, i]
                out_values = out_values2

        out = type(self)(x=x, y=y)

        if self.x._relative_to is not None:
            re = self.y.interpolate_centres_to_nodes(self.x.relative_to)
            if npall(diff(self.y.edges) < 0.0):
                vals = interp(x=y.centres[::-1], xp=self.y.edges[::-1], fp=re[::-1])[::-1]
            else:
                vals = interp(x=y.centres, xp=self.y.edges, fp=re)

            out.x._relative_to = vals

        if self.y._relative_to is not None:
            re = self.x.interpolate_centres_to_nodes(self.y.relative_to)
            if npall(diff(self.x.edges) < 0.0):
                vals = interp(x=x.centres[::-1], xp=self.x.edges[::-1], fp=re[::-1])[::-1]
            else:
                vals = interp(x=x.centres, xp=self.x.edges, fp=re)
            out.y._relative_to = vals

        return out, x_indices, y_indices, out_values

    def _reconcile_intervals(self, intervals, axis=0):

        assert size(intervals) > 1, ValueError("intervals must have size > 1")

        ax = self.other_axis(axis)

        i0 = maximum(0, searchsorted(intervals, ax.edges[0]))
        i1 = minimum(ax.nCells.item(), searchsorted(intervals, ax.edges[-1])+1)

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
        if ndim(x) == 2:
            x = x[:, 0]
            y = x[:, 1]

        assert (size(x) == size(y)), ValueError("x and y must have the same size")
        if trim:
            flag = self.x.in_bounds(x) & self.y.in_bounds(y)
            i = where(flag)[0]
            out = empty([2, i.size], dtype=int32)
            out[0, :] = self.x.cellIndex(x[i])
            out[1, :] = self.y.cellIndex(y[i])
        else:
            out = empty([2, size(x)], dtype=int32)
            out[0, :] = self.x.cellIndex(x, clip=clip)
            out[1, :] = self.y.cellIndex(y, clip=clip)
        return squeeze(out)

    def line_indices(self, x, y):
        i = self.cellIndices(x, y)
        out = utilities.bresenham(i[0, :], i[1, :])
        out[0, :] = minimum(out[0, :], self.x.nCells)
        out[1, :] = minimum(out[1, :], self.y.nCells)
        return out

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

        return ravel_multi_index(ixy, self.shape, order=order)


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

        return unravel_index(indices, self.shape, order=order)

    def pcolor(self, values, yAxis='absolute', **kwargs):
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
            If zAxis is 'absolute' the vertical axis is the relative_to plus z.
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
        if self.x.log is not None:
            kwargs['xscale'] = 'log'
        if self.y.log is not None:
            kwargs['yscale'] = 'log'

        x_mask = kwargs.pop('x_mask', None); y_mask = kwargs.pop('y_mask', None)
        mask = (x_mask is not None) & (y_mask is not None)

        masked = self
        if mask:
            masked, x_indices, z_indices, values = self.mask_cells(x_mask, y_mask, values)
            if 'alpha' in kwargs:
                kwargs['alpha'] = kwargs['alpha'].expand(x_indices, z_indices, masked.shape)
                # _, _, _, kwargs['alpha'] = self.mask_cells(x_mask, y_mask, kwargs['alpha'])

        if (self.x._relative_to is None) and (self.y._relative_to is None):

            xm = masked.x_edges; ym = masked.y_edges

            ax, pm, cb = cP.pcolormesh(xm, ym, values, **kwargs)
            ax.set_xlabel(xm.label); ax.set_ylabel(ym.label)
        else:
            # Need to expand the yaxis edges since they could be draped.
            if mask:
                ax, pm, cb = masked.pcolor(values, **kwargs)
            else:
                x = self.x_edges; y = self.y_edges

                ax, pm, cb = cP.pcolor(x=x, y=y, values=values, **kwargs)

        return ax, pm, cb


    def plot(self, *args, **kwargs):
        return self.pcolor(*args, **kwargs)

    def plot_grid(self, **kwargs):
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

        if (self.x._relative_to is None) and (self.y._relative_to is None):
            tmp = DataArray(full(self.shape, fill_value=nan)).T
            tmp.pcolor(x=self.x.edges_absolute, y=self.y.edges_absolute, grid=True, colorbar=False, **kwargs)

        else:
            xscale = kwargs.pop('xscale')
            yscale = kwargs.pop('yscale')
            flipX = kwargs.pop('flipX', False)
            flipY = kwargs.pop('flipY', False)
            c = kwargs.pop('color', 'k')

            ax = kwargs.get('ax', plt.gca())

            cP.pretty(ax)

            x_mesh = self.x_edges
            y_mesh = self.y_edges

            a = dstack([x_mesh, y_mesh])
            b = dstack([x_mesh.T, y_mesh.T])

            ls = LineCollection(a, color='k', linestyle='solid', **kwargs)
            ax.add_collection(ls)

            ls = LineCollection(b, color='k', linestyle='solid', **kwargs)
            ax.add_collection(ls)

            dz = 0.02 * abs(x_mesh.max() - x_mesh.min())
            ax.set_xlim(x_mesh.min() - dz, x_mesh.max() + dz)
            dz = 0.02 * abs(y_mesh.max() - y_mesh.min())
            ax.set_ylim(y_mesh.min() - dz, y_mesh.max() + dz)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlabel(self.x._centres.label)
            ax.set_ylabel(self.y._centres.label)

            if flipX:
                ax.set_xlim(ax.get_xlim()[::-1])

            if flipY:
                ax.set_ylim(ax.get_ylim()[::-1])

    def plot_relative_to(self, axis=0, **kwargs):
        """Plot the relative_to of the mesh as a line. """

        kwargs['c'] = kwargs.pop('color', 'k')
        kwargs['linewidth'] = kwargs.pop('linewidth', 1.0)

        return self.axis(axis).relative_to.plot(x = self.other_axis(axis).centres, **kwargs)

    def plot_line(self, values, **kwargs):
        axis = kwargs.get('axis', 0)
        ax = self.axis(1-axis)
        if self.x.log is not None:
            kwargs['xscale'] = 'log'
        if self.y.log is not None:
            kwargs['yscale'] = 'log'

        if size(values) == ax.nCells:
            self.plot_line_centres(values, **kwargs)
        elif values.size == ax.nEdges:
            self.plot_line_edges(values, **kwargs)
        else:
            assert False, ValueError('values need to have shape {} or {} but have shape {}'.format(ax.nCells, ax.nEdges, values.shape))

    def plot_line_centres(self, values, **kwargs):
        axis = kwargs.pop('axis', 0)
        if axis == 0:
            cP.plot(values, self.y.centres_absolute, **kwargs)
        else:
            cP.plot(self.x.centres_absolute, values, **kwargs)

    def plot_line_edges(self, values, **kwargs):
        axis = kwargs.pop('axis', 0)
        if axis == 1:
            cP.hlines(values, xmin=self.x.edges_absolute[:-1], xmax=self.x.edges_absolute[1:], **kwargs)
        else:
            cP.vlines(values, ymin=self.y.edges_absolute[:-1], ymax=self.y.edges_absolute[1:], **kwargs)

    # def plot_value_posteriors(self, axes, values, axis, value_kwargs={}, **kwargs):
    #     # assert len(axes) == 5, ValueError("Must have length 5 list of axes for the posteriors. self.init_posterior_plots can generate them")

    #     # best = kwargs.get('best', None)
    #     # if best is not None:
    #     #     ncells_kwargs['line'] = best.nCells
    #     #     edges_kwargs['line'] = best.edges[1:]

    #     flipx = kwargs.pop('flipX', False)
    #     flipy = kwargs.pop('flipY', False)

    #     mean = values.posterior.mean(axis=axis)
    #     mean.pcolor(ax=axes[0], **value_kwargs)
    #     tmp = values.posterior.percentile(percent=5.0, axis=axis)
    #     tmp.pcolor(ax=axes[2], **value_kwargs)
    #     tmp = values.posterior.percentile(percent=95.0, axis=axis)
    #     tmp.pcolor(ax=axes[4], **value_kwargs)
    #     tmp = values.posterior.entropy(axis=axis)
    #     tmp.pcolor(ax=axes[1])
    #     tmp = values.posterior.opacity(axis=axis)
    #     a, b, cb = tmp.pcolor(axis=axis, ax=axes[3], ticks=[0.0, 0.5, 1.0], cmap='plasma')

    #     if cb is not None:
    #         labels = ['Less', '', 'More']
    #         cb.ax.set_yticklabels(labels)
    #         cb.set_label("Confidence")

    @property
    def summary(self):
        """ Display a summary of the 3D Point Cloud """
        msg = ("{}: \n"
              "Shape: : {} \nx\n{}y\n{}").format(type(self).__name__, self.shape, self.x.summary, self.y.summary)
        # if not self.relative_to is None:
        #     msg += self.relative_to.summary
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

        z = minimum(0.001 * minimum(self.x.range, self.y.range), 1.0)
        x, y, z = meshgrid(self.x.edges, self.y.edges, r_[0.0, z])

        return pv.StructuredGrid(x, y, z)

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None, upcast=True):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        if (add_axis is not None) and (upcast):
            return self._create_hdf_3d(parent, name, withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        grp = self.create_hdf_group(parent, name)
        self.x.createHdf(grp, 'x', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=upcast)
        self.y.createHdf(grp, 'y', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=upcast)
        # if not self.relative_to is None:
        #     self.relative_to.createHdf(grp, 'relative_to', withPosterior=withPosterior, fillvalue=fillvalue)

        return grp

    def _create_hdf_3d(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        from .RectilinearMesh3D import RectilinearMesh3D
        if isinstance(add_axis, (int, int_)):
            x = arange(add_axis, dtype=float64)
        else:
            x = add_axis
        if not isinstance(x, RectilinearMesh1D):
            x = RectilinearMesh1D(centres=x, dimension=0)

        out = self.create_hdf_group(parent, name, hdf_name='RectilinearMesh3D')

        x.toHdf(out, 'x', withPosterior=withPosterior)
        self.x.createHdf(out, 'y', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=False)
        self.y.createHdf(out, 'z', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=False)

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

    def _write_hdf_3d(self, parent, name, index, withPosterior=True):
        grp = parent[name]
        assert '3D' in grp.attrs['repr'], TypeError("HDF creation must have an axis added.")

        self.x.writeHdf(grp, 'y', index=index, withPosterior=withPosterior, upcast=False)
        self.y.writeHdf(grp, 'z', index=index, withPosterior=withPosterior, upcast=False)


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
                x = RectilinearMesh1D.fromHdf(grp['y'], index=index, skip_posterior=skip_posterior)
                y = RectilinearMesh1D.fromHdf(grp['z'], index=index, skip_posterior=skip_posterior)

                out = cls(x=x, y=y)
                return out
        else:
            if index is not None:

                return RectilinearMesh1D.fromHdf(grp, index=index, skip_posterior=skip_posterior)
            else:
                x = RectilinearMesh1D.fromHdf(grp['x'], index=index, skip_posterior=skip_posterior)
                y = RectilinearMesh1D.fromHdf(grp['y'], index=index, skip_posterior=skip_posterior)

                return cls(x=x, y=y)

    def fromHdf_cell_values(self, grp, key, index=None, skip_posterior=False):
        return StatArray.fromHdf(grp[key], index=index, skip_posterior=skip_posterior)


    def range(self, axis):
        return self.axis(axis).range

    @property
    def x_centres(self):
        """Creates an array suitable for plt.pcolormesh for the abscissa.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'x', the horizontal xAxis uses self.x
            If xAxis is 'y', the horizontal xAxis uses self.y
            If xAxis is 'r', the horizontal xAxis uses cumulative distance along the line.

        """
        if self.x._relative_to is None:
            out = repeat(self.x.centres_absolute[:, None], self.y.nCells, 1)
        else:
            if self.x.relative_to.size == 1:
                out = repeat(self.x.centres_absolute[:, None], self.y.nCells, 1)
            else:
                edges = self.x.relative_to + self.x.centres[:, None]
                out = utilities._power(edges, self.x.log)

        return out

    @property
    def x_edges(self):
        """Creates an array suitable for plt.pcolormesh for the ordinate """
        re_tmp = None
        if self.x._relative_to is not None:
            nd = ndim(self.x.relative_to)
            if nd == 1:
                if self.x.relative_to.size > 1:
                    re_tmp = deepcopy(self.x.relative_to)
                    self.x.relative_to = self.y.interpolate_centres_to_nodes(self.x.relative_to)

        x_edges = self.x.edges_absolute

        if ndim(x_edges) == 1:
            x_edges = repeat(x_edges[:, None], self.y.nEdges, 1)

        if re_tmp is not None:
            self.x.relative_to = re_tmp
            x_edges.name = self.x.relative_to.name
            x_edges.units = self.x.relative_to.units

        return x_edges

    @property
    def y_centres(self):
        """Creates an array suitable for plt.pcolormesh for the abscissa.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'x', the horizontal xAxis uses self.x
            If xAxis is 'y', the horizontal xAxis uses self.y
            If xAxis is 'r', the horizontal xAxis uses cumulative distance along the line.

        """
        if self.y._relative_to is None:
            out = repeat(self.y.centres_absolute[None, :], self.x.nCells, 0)
        else:
            if self.y.relative_to.size == 1:
                out = repeat(self.y.centres_absolute[None, :], self.x.nCells, 0)
            else:
                out = repeat(self.y.relative_to[:, None], self.y.nCells, 1) + self.y.centres
                out = utilities._power(out, self.y.log)

        return out

    @property
    def y_edges(self):
        """Creates an array suitable for plt.pcolormesh for the ordinate """
        re_tmp = None
        if self.y._relative_to is not None:
            nd = ndim(self.y.relative_to)
            if nd == 1:
                if self.y.relative_to.size > 1:
                    re_tmp = deepcopy(self.y.relative_to)
                    self.y.relative_to = self.x.interpolate_centres_to_nodes(self.y.relative_to)

        y_edges = self.y.edges_absolute

        if ndim(y_edges) == 1:
            y_edges = repeat(y_edges[None, :], self.x.nEdges, 0)

        if re_tmp is not None:
            self.y.relative_to = re_tmp

            y_edges.name = self.y.relative_to.name
            y_edges.units = self.y.relative_to.units

        return y_edges