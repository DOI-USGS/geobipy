""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import deepcopy
from numpy import argwhere, asarray, dot, empty, expand_dims, full, int32, integer, insert
from numpy import max,  min, nan, ndim, outer, prod, ravel_multi_index
from numpy import repeat, s_, size, squeeze, sum, swapaxes, take, unravel_index
from numpy import where, zeros
from numpy import all as npall
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ..core.DataArray import DataArray
from ..statistics.StatArray import StatArray
from .RectilinearMesh1D import RectilinearMesh1D
from .RectilinearMesh2D import RectilinearMesh2D
from scipy.stats import binned_statistic
from ...base import utilities
from scipy.sparse import kron
import progressbar

class RectilinearMesh3D(RectilinearMesh2D):
    """Class defining a 3D rectilinear mesh with cell centres and edges.

    Contains a simple mesh with cell edges, widths, and centre locations.
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
    relative_toCentres : geobipy.StatArray, optional
        The relative_to of each point at the x, y locations. Only relative_toCentres or relative_toEdges can be given, not both.
        Has shape (y.nCells, x.nCells).
    relative_toEdges : geobipy.StatArray, optional
        The relative_to of each point at the x, y locations of the edges of each cell, including the outermost edges. Only relative_toCentres or relative_toEdges can be given, not both.
        Has shape (y.nEdges, x.nEdges).

    Other Parameters
    ----------------
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
    z_axis = 2

    def __init__(self, x=None, y=None, z=None, **kwargs):
        """ Initialize a 2D Rectilinear Mesh"""

        self._x_edges = None
        self._y_edges = None
        self._z_edges = None
        self.x = kwargs if x is None else x
        self.y = kwargs if y is None else y
        self.z = kwargs if z is None else z

        self.check_x_relative_to()
        self.check_y_relative_to()
        self.check_z_relative_to()

    def __getitem__(self, slic):
        """Slice into the mesh. """
        axis = []
        for i, x in enumerate(slic):
            if isinstance(x, (integer, int)):
                axis.append(i)

        assert not len(axis) == 3, ValueError("slice cannot be a single cell")

        axis = squeeze(asarray(axis, dtype=int32))

        match axis.size:
            case 0: # Returning a 3D mesh
                return self.__slice_3d(slic)
            case 1: # Returning a 2D mesh
                return self.__slice_2d(slic, axis)
            case 2: # Returning a 1D mesh
                return self.__slice_1d(slic, axis)
            case _:
                raise ValueError(f"invalid slice {slic}")


    def __slice_3d(self, slic):
        x = self.x[slic[0]]
        y = self.y[slic[1]]
        z = self.z[slic[2]]
        if x._relative_to is not None:
            nd = ndim(x.relative_to)
            slc = s_[:]
            if nd == 2:
                slc = slic[1:]
            elif nd == 1:
                s = x.relative_to.size
                if s == self.shape[1]:
                    slc = slic[1]
                elif s == self.shape[2]:
                    slc = slic[2]

            x.relative_to = x.relative_to[slc]

        if y._relative_to is not None:
            nd = ndim(y.relative_to)
            slc = s_[:]
            if nd == 2:
                slc = slic[::2]
            elif nd == 1:
                s = y.relative_to.size
                if s == self.shape[0]:
                    slc = slic[0]
                elif s == self.shape[2]:
                    slc = slic[2]

            y.relative_to = y.relative_to[slc]

        if z._relative_to is not None:
            nd = ndim(z.relative_to)
            slc = s_[:]
            if nd == 2:
                slc = slic[:2]
            elif nd == 1:
                s = z.relative_to.size
                if s == self.shape[0]:
                    slc = slic[0]
                elif s == self.shape[1]:
                    slc = slic[1]

            z.relative_to = z.relative_to[slc]

        out = type(self)(x=x, y=y, z=z)
        return out

    def __slice_2d(self, slic, axis):
        a = [x for x in (0, 1, 2) if not x in axis]
        b = [x for x in (0, 1, 2) if x in axis][0]

        x = deepcopy(self.axis(a[0])) # X is always X or Y
        if x._relative_to is not None :
            if (x._relative_to.size > 1) and (npall(x._relative_to != 0.0)):
                rt_dims = asarray([t for t in (0, 1, 2) if t != x.dimension])
                axis = squeeze(argwhere(rt_dims == b))
                if x.relative_to.ndim == 2:
                    x.relative_to = take(x.relative_to, slic[b], axis)
            else:
                x.relative_to = None

        y = deepcopy(self.axis(a[1])) # Y is always Z or Y
        if y._relative_to is not None:
            if (y._relative_to.size > 1) and (npall(y._relative_to != 0.0)):
                rt_dims = asarray([t for t in (0, 1, 2) if t != y.dimension])
                axis = squeeze(argwhere(rt_dims == b))
                if y.relative_to.ndim == 2:
                    y.relative_to = take(y.relative_to, slic[b], axis)
            else:
                y.relative_to = None

        x = x[slic[a[0]]]
        x.dimension = 0
        y = y[slic[a[1]]]
        y.dimension = 1

        return RectilinearMesh2D(x=x, y=y)

    def __slice_1d(self, slic, axis):
        a = [x for x in (0, 1, 2) if not x in axis]
        b = [x for x in (0, 1, 2) if x in axis]

        out = self.axis(a[0])[slic[a[0]]]
        if out._relative_to is not None:
            out.relative_to = out.relative_to[slic[b[0]], slic[b[1]]]

        return out

    @property
    def addressof(self):
        msg = super().addressof
        msg += "z:\n{}".format(("|   "+self.z.addressof.replace("\n", "\n|   "))[:-4])
        return msg

    def centres(self, axis):
        if axis == 0:
            return self.x_centres
        elif axis == 1:
            return self.y_centres
        else:
            return self.z_centres

    # def edges(self, axis):
    #     if axis == 0:
    #         return self.x_edges
    #     elif axis == 1:
    #         return self.y_edges
    #     else:
    #         return self.z_edges

    @property
    def area(self):
        return outer(self.x.widths, self.y.widths)[:, :, None] * self.z.widths

    @property
    def x_centres(self):
        out = self.x.centres_absolute
        nd = ndim(out)
        if nd == 3:
            return out
        elif nd == 2:
            return repeat(out[:, :, None], self.z.nCells, 2)
        elif nd == 1:
            return repeat(repeat(out[:, None], self.y.nCells, 1)[:, :, None], self.z.nCells, 2)

    @property
    def y_centres(self):
        out = self.y.centres_absolute
        nd = ndim(out)
        if nd == 3:
            return out
        elif nd == 2:
            return repeat(out[:, :, None], self.z.nCells, 2)
        elif nd == 1:
            return repeat(repeat(out[None, :], self.x.nCells, 0)[:, :, None], self.z.nCells, 2)

    @property
    def z_centres(self):
        out = self.z.centres_absolute
        nd = ndim(out)
        if nd == 3:
            return out
        # elif nd == 2:
            # return repeat(out[:, :, None], self.z.nCells, 2)
        elif nd == 1:
            return repeat(repeat(out[None, :], self.y.nCells, 0)[None, :, :], self.x.nCells, 0)

    @property
    def x_edges(self):
        re_tmp = None
        if self.x._relative_to is not None:
            re_tmp = deepcopy(self.x.relative_to)
            nd = ndim(self.x.relative_to)
            if nd == 2:
                mesh = self.remove_axis(0)
                re_nodes = mesh.interpolate_centres_to_nodes(self.x.relative_to)
                self.x.relative_to = re_nodes

        out = super().x_edges

        if re_tmp is not None:
            self.x.relative_to = re_tmp

        nd = ndim(out)
        if nd == 3:
            return out
        elif nd == 2:
            return repeat(out[:, :, None], self.z.nEdges, 2)
        elif nd == 1:
            return repeat(repeat(out[:, None], self.y.nEdges, 1)[:, :, None], self.z.nEdges, 2)


    @property
    def y_edges(self):
        re_tmp = None
        if self.y._relative_to is not None:
            re_tmp = deepcopy(self.y.relative_to)
            nd = ndim(self.y.relative_to)
            if nd == 2:
                mesh = self.remove_axis(1)
                re_nodes = mesh.interpolate_centres_to_nodes(self.y.relative_to)
                self.y.relative_to = re_nodes

        out = super().y_edges

        if re_tmp is not None:
            self.y.relative_to = re_tmp

        nd = ndim(out)
        if nd == 3:
            return out
        elif nd == 2:
            return repeat(out[:, :, None], self.z.nEdges, 2)
        elif nd == 1:
            return repeat(repeat(out[None, :], self.x.nEdges, 0)[:, :, None], self.z.nEdges, 2)

    @property
    def z_edges(self):

        if self._z_edges is None:
            self._z_edges = self.get_generic_z_edges()

        return self._z_edges

    def get_generic_z_edges(self):
        re_tmp = None
        if self.z._relative_to is not None:

            nd = ndim(self.z.relative_to)
            if nd == 2:
                re_tmp = deepcopy(self.z.relative_to)
                mesh = self.remove_axis(2)
                re_nodes = mesh.interpolate_centres_to_nodes(self.z.relative_to)
                self.z.relative_to = re_nodes

        out = self.z.edges_absolute

        if re_tmp is not None:
            self.z.relative_to = re_tmp

        if ndim(out) == 1:
            return repeat(repeat(out[None, :], self.y.nEdges, 0)[None, :, :], self.x.nEdges, 0)

        return out


        # out = self.z.edges_absolute
        # nd = ndim(out)
        # if nd == 3:
        #     return out
        # # elif nd == 2:
        #     # return repeat(out[:, :, None], self.z.nCells, 2)
        # elif nd == 1:
        #     return repeat(repeat(out[None, :], self.y.nEdges, 0)[None, :, :], self.x.nEdges, 0)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, values):
        if isinstance(values, dict):
            # mesh of the z axis values
            values = RectilinearMesh1D(
                        centres=values.get('z_centres'),
                        edges=values.get('z_edges'),
                        log=values.get('z_log'),
                        relative_to=values.get('z_relative_to'),
                        dimension=self.z_axis)

        assert isinstance(values, RectilinearMesh1D), TypeError('z must be a RectilinearMesh1D')
        assert values.dimension == 2
        self._z = values

    def check_x_relative_to(self):
        if self.x.relative_to is not None:
            if ndim(self.x.relative_to) == 2:
                assert npall(self.x.relative_to.shape == self.shape[1:]), ValueError(f"2D relative_to on x has shape {self.x.relative_to.shape} but needs shape {self.shape[:2]}")

    def check_y_relative_to(self):
        if self.y.relative_to is not None:
            if ndim(self.y.relative_to) == 2:
                assert npall(self.y.relative_to.shape == self.shape[0::2]), ValueError(f"2D relative_to on y has shape {self.y.relative_to.shape} but needs size {self.shape[0::2]}")

    def check_z_relative_to(self):
        if self.y.relative_to is not None:
            if ndim(self.z.relative_to) == 2:
                assert npall(self.z.relative_to.shape == self.shape[:2]), ValueError(f"2D relative_to on z has shape {self.z.relative_to.shape} but needs size {self.shape[0::2]}")

    def other_axis(self, axis):

        if axis == 0:
            return self.y, self.z
        elif axis == 1:
            return self.x, self.z
        elif axis == 2:
            return self.x, self.y

    def axis(self, axis):
        if isinstance(axis, str):
            assert axis in ['x', 'y', 'z'], Exception("axis must be either 'x', 'y', 'z'")

            if axis == 'x':
                return self.x
            elif axis == 'y':
                return self.y
            elif axis == 'z':
                return self.z

        if axis == 0:
            return self.x
        elif axis == 1:
            return self.y
        elif axis == 2:
            return self.z

    def set_axis(self, axis, value):
        if axis == 0:
            self.x = value
        elif axis == 1:
            self.y = value
        elif axis == 2:
            self.z = value

    def other_axis_indices(self, axis):
        if axis == 0:
            return 1, 2
        elif axis == 1:
            return 0, 2
        elif axis == 2:
            return 0, 1

    @property
    def nCells(self):
        """The number of cells in the mesh.

        Returns
        -------
        out : int
            Number of cells

        """
        return prod(self.shape)

    @property
    def ndim(self):
        return 3

    @property
    def nNodes(self):
        """The number of nodes in the mesh.

        Returns
        -------
        out : int
            Number of nodes

        """
        return self.x.nEdges * self.y.nEdges * self.z.nEdges

    @property
    def shape(self):
        """The dimensions of the mesh

        Returns
        -------
        out : array_like
            Array of integers

        """
        return (self.x.nCells.item(), self.y.nCells.item(), self.z.nCells.item())

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

        kwargs['trim'] = None
        ax, pc, cb = sub.pcolor(values=sub_v, **kwargs)

        tmp, _ = utilities._log(values, kwargs.get('log', None))
        pc.set_clim(min(tmp), max(tmp))

        kwargs['colorbar'] = False

        def animate(i):
            plt.title('{:.2f}'.format(self.axis(axis).centres[i]))
            slic[axis] = i
            tmp, _ = utilities._log(values[tuple(slic)].flatten(), kwargs.get('log', None))
            pc.set_array(tmp)

        anim = FuncAnimation(fig, animate, interval=300, frames=self.axis(axis).nCells.item())

        plt.draw()
        anim.save(filename)

    def _compute_probability(self, distribution, pdf, log, log_probability, axis=0, **kwargs):
        centres = self.centres(axis=axis)
        centres, _ = utilities._log(centres, log)

        ax, bx = self.other_axis(axis)

        a = [x for x in (0, 1, 2) if not x == axis]
        b = [x for x in (0, 1, 2) if x == axis]

        shp = list(self.shape)
        shp[axis] = distribution.ndim

        probability = zeros(shp)

        track = kwargs.pop('track', True)

        r = range(ax.nCells.item() * bx.nCells.item())
        if track:
            Bar = progressbar.ProgressBar()
            r = Bar(r)

        mesh_2d = self.remove_axis(axis)

        for i in r:
            j = list(mesh_2d.unravelIndex(i))
            j.insert(axis, s_[:])
            j = tuple(j)
            p = distribution.probability(centres[j], log_probability)

            probability[j] = dot(p.T, pdf[j])

        probability = probability / expand_dims(sum(probability, axis), axis=axis)

        return DataArray(probability, name='Marginal Probability')

    def __deepcopy__(self, memo={}):
        """ Define the deepcopy for the StatArray """
        return RectilinearMesh3D(x=self.x, y=self.y, z=self.z)

    def edges(self, axis):
        """ Gets the cell edges in the given dimension """
        if axis == 0:
            return self.x_edges
        elif axis == 1:
            return self.y_edges
        else:
            return self.z_edges

    def getXAxis(self, axis='x', centres=False):
        assert axis in ['x', 'y', 'z', 'r'], Exception("axis must be either 'x', 'y', 'z', 'r'")
        if axis == 'x':
            ax = self.x
        elif axis == 'y':
            assert self.xyz, Exception("To plot against 'y' the mesh must be instantiated with three co-ordinates")
            ax = self.y
        elif axis == 'z':
            ax = self.z
        elif axis == 'r':
            assert self.xyz, Exception("To plot against 'r' the mesh must be instantiated with three co-ordinates")
            ax = self.distance

        return ax.centres if centres else ax.edges


    # def xGradientMatrix(self):
    #     tmp = self.x.gradientMatrix()
    #     return kron(diag(sqrt(self.z.cellWidths)), tmp)


    # def zGradientMatrix(self):
    #     nx = self.x.nCells
    #     nz = self.z.nCells
    #     tmp = 1.0 / sqrt(rm2.x.centreTocentre)
    #     a = repeat(tmp, nz) * tile(sqrt(rm2.z.cellWidths), nx-1)
    #     return diags([a, -a], [0, nx], shape=(nz * (nx-1), nz*nz))


    # def hasSameSize(self, other):
    #     """ Determines if the meshes have the same dimension sizes """
    #     # if self.arr.shape != other.arr.shape:
    #     #     return False
    #     if self.x.nCells != other.x.nCells:
    #         return False
    #     if self.z.nCells != other.z.nCells:
    #         return False
    #     return True


    # def intervalStatistic(self, arr, intervals, axis=0, statistic='mean'):
    #     """Compute a statistic of the array between the intervals given along dimension dim.

    #     Parameters
    #     ----------
    #     arr : array_like
    #         2D array to take the mean over the given intervals
    #     intervals : array_like
    #         A new set of mesh edges. The mean is computed between each two edges in the array.
    #     axis : int, optional
    #         Which axis to take the mean
    #     statistic : string or callable, optional
    #         The statistic to compute (default is 'mean').
    #         The following statistics are available:

    #       * 'mean' : compute the mean of values for points within each bin.
    #         Empty bins will be represented by NaN.
    #       * 'median' : compute the median of values for points within each
    #         bin. Empty bins will be represented by NaN.
    #       * 'count' : compute the count of points within each bin.  This is
    #         identical to an unweighted histogram.  `values` array is not
    #         referenced.
    #       * 'sum' : compute the sum of values for points within each bin.
    #         This is identical to a weighted histogram.
    #       * 'min' : compute the minimum of values for points within each bin.
    #         Empty bins will be represented by NaN.
    #       * 'max' : compute the maximum of values for point within each bin.
    #         Empty bins will be represented by NaN.
    #       * function : a user-defined function which takes a 1D array of
    #         values, and outputs a single numerical statistic. This function
    #         will be called on the values in each bin.  Empty bins will be
    #         represented by function([]), or NaN if this returns an error.


    #     See Also
    #     --------
    #     scipy.stats.binned_statistic : for more information

    #     """

    #     assert size(intervals) > 1, ValueError("intervals must have size > 1")

    #     intervals = self._reconcile_intervals(intervals, axis=axis)

    #     if (axis == 0):
    #         bins = binned_statistic(self.z.centres, arr.T, bins = intervals, statistic=statistic)
    #         res = bins.statistic.T
    #     else:
    #         bins = binned_statistic(self.x.centres, arr, bins = intervals, statistic=statistic)
    #         res = bins.statistic

    #     return res, intervals


    # def _reconcile_intervals(self, intervals, axis=0):

    #     assert size(intervals) > 1, ValueError("intervals must have size > 1")

    #     ax

    #     if (axis == 0):
    #         # Make sure the intervals are within the axis.
    #         i0 = maximum(0, searchsorted(intervals, self.z.edges[0]))
    #         i1 = minimum(self.z.nCells, searchsorted(intervals, self.z.edges[-1])+1)
    #         intervals = intervals[i0:i1]

    #     else:
    #         i0 = maximum(0, searchsorted(intervals, self.x.edges[0]))
    #         i1 = minimum(self.x.nCells, searchsorted(intervals, self.x.edges[-1])+1)
    #         intervals = intervals[i0:i1]

    #     return intervals


    def cellIndices(self, x, y, z, clip=False, trim=False):
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
        assert (size(x) == size(y) == size(z)), ValueError("x, y, z must have the same size")
        if trim:
            flag = self.x.in_bounds(x) & self.y.in_bounds(y) & self.z.in_bounds(z)
            i = where(flag)[0]
            out = empty([3, i.size], dtype=int32)
            out[0, :] = self.x.cellIndex(x[i])
            out[1, :] = self.y.cellIndex(y[i])
            out[2, :] = self.z.cellIndex(z[i])
        else:
            out = empty([3, size(x)], dtype=int32)
            out[0, :] = self.x.cellIndex(x, clip=clip)
            out[1, :] = self.y.cellIndex(y, clip=clip)
            out[2, :] = self.z.cellIndex(z, clip=clip)
        return squeeze(out)

    # def _mean(self, values, axis=0):

    #     a = self.axis(axis)
    #     if a._relative_to is None:
    #         return super()._mean(values, axis)

    #     s = tuple([s_[:] if i == axis else None for i in range(self.ndim)])

    #     centres = self.centres_absolute(axis)

    #     t = sum(centres * values, axis = axis)
    #     s = values.sum(axis = axis)

    #     if size(t) == 1:
    #         out = t / s
    #     else:
    #         i = where(s > 0.0)
    #         out = StatArray(t.shape)
    #         out[i] = t[i] / s[i]

    #     return out

    # def _percentile(self, values, percent=95.0, axis=0):
    #     """Gets the percent interval along axis.

    #     Get the statistical interval, e.g. median is 50%.

    #     Parameters
    #     ----------
    #     values : array_like
    #         Values used to compute interval like histogram counts.
    #     percent : float
    #         Interval percentage.  0.0 < percent < 100.0
    #     log : 'e' or float, optional
    #         Take the log of the interval to a base. 'e' if log = 'e', or a number e.g. log = 10.
    #     axis : int
    #         Along which axis to obtain the interval locations.

    #     Returns
    #     -------
    #     interval : array_like
    #         Contains the interval along the specified axis. Has size equal to self.shape[axis].

    #     """
    #     percent *= 0.01

    #     # total of the counts
    #     total = values.sum(axis=axis)
    #     # Cumulative sum
    #     cs = cumsum(values, axis=axis)
    #     # Cumulative "probability"
    #     d = expand_dims(total, axis)
    #     tmp = zeros_like(cs, dtype=float64)
    #     divide(cs, d, out=tmp, where=d > 0.0)
    #     # Find the interval
    #     i = apply_along_axis(searchsorted, axis, tmp, percent)
    #     i[i == values.shape[axis]] = values.shape[axis]-1

    #     centres = self.centres(axis)

    #     if size(percent) == 1:
    #         i = expand_dims(i, axis=axis)
    #     return squeeze(take_along_axis(centres, i, axis=axis))


    def plot_grid(self):
        raise NotImplementedError("Slice a 3D mesh before using plotGrid.")



    def ravelIndices(self, indices, order='C'):
        """Return a global index into a 1D array given the two cell indices in x and z.

        Parameters
        ----------
        indices : array_like
            A tuple of integer arrays, one array for each dimension.

        Returns
        -------
        out : int
            Global index.

        """
        return ravel_multi_index(indices, self.shape, order=order)

    def unravelIndex(self, index, order='C'):
        """Return local indices given a global one.

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

        return unravel_index(index, self.shape, order=order)

    def pcolor(self, axis, index=None, **kwargs):

        values = kwargs['values']
        kwargs['axis'] = axis

        if index is not None:
            slic = [s_[:] for i in range(self.ndim)]
            slic[axis] = index
            slic = tuple(slic)

            tmp = self[slic]
            kwargs['values'] = values[slic]

            return tmp.pcolor(**kwargs)

        mesh = self.remove_axis(axis)


        x_mask = kwargs.get('x_mask', None); y_mask = kwargs.get('y_mask', None)
        mask = (x_mask is not None) or (y_mask is not None)

        masked = mesh
        if mask:
            masked, x_indices, y_indices, _ = mesh.mask_cells(x_mask, y_mask, None)

            kwargs['values'] = values.expand(x_indices, y_indices, masked.shape, axis=axis)
            if 'classes' in kwargs:
                if 'id' in kwargs['classes']:
                    kwargs['classes']['id'] = kwargs['classes']['id'].expand(x_indices, y_indices, masked.shape)


        return masked.pcolor(**kwargs)

    def pyvista_mesh(self, **kwargs):
        """Creates a pyvista plotting object linked to VTK.

        Use mesh.plot(show_edges=True, show_grid=True) to plot the mesh.

        Returns
        -------

        """
        import pyvista as pv

        x = swapaxes(self.x_edges, 0, 1)
        y = swapaxes(self.y_edges, 0, 1)
        z = swapaxes(self.z_edges, 0, 1)

        mesh = pv.StructuredGrid(x, y, z)

        return mesh

    def _reorder_for_pyvista(self, values):
        return utilities.reorder_3d_for_pyvista(values)

    @property
    def summary(self):
        """ Display a summary of the 3D Point Cloud """
        msg = ("3D Rectilinear Mesh: \n"
              "Shape: : {} \nx\n{}y\n{}z\n{}").format(self.shape, self.x_edges.summary, self.y_edges.summary, self.z_edges.summary)
        # if not self.relative_to is None:
        #     msg += self.relative_to.summary
        return msg

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None, upcast=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside parent
        grp = self.create_hdf_group(parent, name)

        self.x.createHdf(grp, 'x', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=False)
        self.y.createHdf(grp, 'y', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=False)
        self.z.createHdf(grp, 'z', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=False)
        # if not self.relative_to is None:
        #     self.relative_to.createHdf(grp, 'relative_to', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        return grp

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        grp = parent.get(name)
        self.x.writeHdf(grp, 'x',  withPosterior=withPosterior, index=index)
        self.y.writeHdf(grp, 'y',  withPosterior=withPosterior, index=index)
        self.z.writeHdf(grp, 'z',  withPosterior=withPosterior, index=index)
        # if not self.relative_to is None:
        #     self.relative_to.writeHdf(grp, 'relative_to', withPosterior=withPosterior, index=index)

    @classmethod
    def fromHdf(cls, grp, index=None, skip_posterior=False):
        """ Reads in the object from a HDF file """

        if index is None:
            x = RectilinearMesh1D.fromHdf(grp['x'], skip_posterior=skip_posterior)
            x.dimension = 0
            y = RectilinearMesh1D.fromHdf(grp['y'], skip_posterior=skip_posterior)
            y.dimension = 1
            z = RectilinearMesh1D.fromHdf(grp['z'], skip_posterior=skip_posterior)
            z.dimension = 2

            if y._relative_to is not None:
                nd = ndim(y.relative_to)
                if nd == 1:
                    y.relative_to = repeat(y.relative_to[:, None], z.nCells, 1)
            if z._relative_to is not None:
                nd = ndim(z.relative_to)
                if nd == 1:
                    z.relative_to = repeat(z.relative_to[:, None], y.nCells, 1)

            out = cls(x=x, y=y, z=z)
        else:
            if isinstance(index, slice):
                assert False, Exception('Cant slice into RectlilinearMesh3D yet from HDF.')
            else:
                return RectilinearMesh2D.fromHdf(grp, index=index, skip_posterior=skip_posterior)

        return out

    def xRange(self):
        """ Get the range of x

        Returns
        -------
        out : numpy.float64
            The range of x

        """

        return self.x.range

    def zRange(self):
        """ Get the range of z

        Returns
        -------
        out : numpy.float64
            The range of z

        """

        return self.z.range


    # def vtkStructure(self):
    #     """Generates a vtk mesh structure that can be used in a vtk file.

    #     Returns
    #     -------
    #     out : pyvtk.VtkData
    #         Vtk data structure

    #     """

    #     # Generate the quad node locations in x
    #     x = self.x.edges
    #     y = self.y.edges
    #     z = self.z.edges

    #     nCells = self.x.nCells * self.z.nCells

    #     z = self.z.edges
    #     nNodes = self.x.nEdges * self.z.nEdges

    #     # Constuct the node locations for the vtk file
    #     nodes = empty([nNodes, 3])
    #     nodes[:, 0] = tile(x, self.z.nEdges)
    #     nodes[:, 1] = tile(y, self.z.nEdges)
    #     nodes[:, 2] = repeat(z, self.x.nEdges)

    #     tmp = int32([0, 1, self.x.nEdges+1, self.x.nEdges])
    #     a = ones(self.x.nCells, dtype=int32)
    #     a[0] = 2
    #     index = (repeat(tmp[:, newaxis], nCells, 1) + cumsum(tile(a, self.z.nCells))-2).T

    #     return VtkData(PolyData(points=nodes, polygons=index))


    # def toVTK(self, fileName, pointData=None, cellData=None, format='binary'):
    #     """Save to a VTK file.

    #     Parameters
    #     ----------
    #     fileName : str
    #         Filename to save to.
    #     pointData : geobipy.StatArray or list of geobipy.StatArray, optional
    #         Data at each node in the mesh. Each entry is saved as a separate
    #         vtk attribute.
    #     cellData : geobipy.StatArray or list of geobipy.StatArray, optional
    #         Data at each cell in the mesh. Each entry is saved as a separate
    #         vtk attribute.
    #     format : str, optional
    #         "ascii" or "binary" format. Ascii is readable, binary is not but results in smaller files.

    #     Raises
    #     ------
    #     TypeError
    #         If pointData or cellData is not a geobipy.StatArray or list of them.
    #     ValueError
    #         If any pointData (cellData) entry does not have size equal to the number of points (cells).
    #     ValueError
    #         If any StatArray does not have a name or units. This is needed for the vtk attribute.

    #     """

    #     vtk = self.vtkStructure()

    #     if not pointData is None:
    #         assert isinstance(pointData, (StatArray, list)), TypeError("pointData must a geobipy.StatArray or a list of them.")
    #         if isinstance(pointData, list):
    #             for p in pointData:
    #                 assert isinstance(p, StatArray), TypeError("pointData entries must be a geobipy.StatArray")
    #                 assert all(p.shape == [self.z.nEdges, self.x.nEdges]), ValueError("pointData entries must have shape {}".format([self.z.nEdges, self.x.nEdges]))
    #                 assert p.hasLabels(), ValueError("StatArray needs a name")
    #                 vtk.point_data.append(Scalars(p.reshape(self.nNodes), p.getNameUnits()))
    #         else:
    #             assert all(pointData.shape == [self.z.nEdges, self.x.nEdges]), ValueError("pointData entries must have shape {}".format([self.z.nEdges, self.x.nEdges]))
    #             assert pointData.hasLabels(), ValueError("StatArray needs a name")
    #             vtk.point_data.append(Scalars(pointData.reshape(self.nNodes), pointData.getNameUnits()))

    #     if not cellData is None:
    #         assert isinstance(cellData, (StatArray, list)), TypeError("cellData must a geobipy.StatArray or a list of them.")
    #         if isinstance(cellData, list):
    #             for p in cellData:
    #                 assert isinstance(p, StatArray), TypeError("cellData entries must be a geobipy.StatArray")
    #                 assert all(p.shape == self.shape), ValueError("cellData entries must have shape {}".format(self.shape))
    #                 assert p.hasLabels(), ValueError("StatArray needs a name")
    #                 vtk.cell_data.append(Scalars(p.reshape(self.nCells), p.getNameUnits()))
    #         else:
    #             assert all(cellData.shape == self.shape), ValueError("cellData entries must have shape {}".format(self.shape))
    #             assert cellData.hasLabels(), ValueError("StatArray needs a name")
    #             vtk.cell_data.append(Scalars(cellData.reshape(self.nCells), cellData.getNameUnits()))

    #     vtk.tofile(fileName, format)

    @classmethod
    def generate_from_rasters(cls, rasters:list, edges=True, absolute=True, **kwargs):

        import numpy as np
        import rioxarray as rio
        from ..model.Model import Model
        from ...base.utilities import nodata_value

        n_layers = len(rasters) - 1

        ds = rio.open_rasterio(rasters[0], from_disk=True)
        mod = Model.from_tif(ds)
        ds.close()

        # Replace finite null values with nan
        mod.values[mod.values == nodata_value(mod.values.dtype)] = np.nan

        mod = mod.fill_nans_with_extrapolation()

        self = cls(x=mod.mesh.x, y=mod.mesh.y, z_edges=np.arange(len(rasters), dtype=np.float64))

        self._z_edges = StatArray(np.zeros(np.asarray(self.shape)+1, dtype=np.float64))
        thickness = StatArray(np.zeros(np.asarray(self.shape), dtype=np.float64), name='thickness', units='m')

        d = mod.interpolate_centres_to_nodes()
        self._z_edges[:, :, -1] = d

        # Loop over layer thickness files
        for i, tif in enumerate(rasters[1:]):
            ds = rio.open_rasterio(tif, from_disk=True)
            mod = Model.from_tif(ds)
            ds.close()

            if np.all(mod.values.shape[::-1] == self.shape[:-1]):
                thickness[:, :, i] = mod.values.T
            else:
                thickness[:, :, i] = mod.values

            mod = mod.fill_nans_with_extrapolation()

            self._z_edges[:, :, i+1] = self._z_edges[:, :, i] - mod.interpolate_centres_to_nodes()


        return self, thickness
