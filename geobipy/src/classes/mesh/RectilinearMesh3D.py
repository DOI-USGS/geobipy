""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import deepcopy
from numpy import dot, empty, expand_dims, int32, integer
from numpy import max,  min, ndim, outer, prod, ravel_multi_index
from numpy import repeat, s_, size, squeeze, swapaxes, take, unravel_index
from numpy import where, zeros
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ...classes.core import StatArray
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
    relativeToCentres : geobipy.StatArray, optional
        The relativeTo of each point at the x, y locations. Only relativeToCentres or relativeToEdges can be given, not both.
        Has shape (y.nCells, x.nCells).
    relativeToEdges : geobipy.StatArray, optional
        The relativeTo of each point at the x, y locations of the edges of each cell, including the outermost edges. Only relativeToCentres or relativeToEdges can be given, not both.
        Has shape (y.nEdges, x.nEdges).

    Other Parameters
    ----------------
    [x, y, z]log : 'e' or float, optional
        See geobipy.RectilinearMesh1D for log description.
    [x, y, z]relativeTo : float, optional
        See geobipy.RectilinearMesh1D for relativeTo description.

    Returns
    -------
    out : RectilinearMesh2D
        The 2D mesh.

    """

    def __init__(self, x=None, y=None, z=None, **kwargs):
        """ Initialize a 2D Rectilinear Mesh"""

        self.x = kwargs if x is None else x
        self.y = kwargs if y is None else y
        self.z = kwargs if z is None else z

        # if self.x._relativeTo is not None:
        #     assert all(self.x.relativeTo.shape == self.shape[1:]), "x axis relative to must have shape {} but has shape {}".format(self.shape[1:], self.x.relativeTo.shape)

        # if self.y._relativeTo is not None:
        #     assert all(self.y.relativeTo.shape == self.shape[::2]), "y axis relative to must have shape {} but has shape {}".format(self.shape[::2], self.y.relativeTo.shape)

        # if self.z._relativeTo is not None:
        #     assert all(self.z.relativeTo.shape == self.shape[:2]), "z axis relative to must have shape {} but has shape {}".format(self.shape[:2], self.z.relativeTo.shape)

    def __getitem__(self, slic):
        """Slice into the mesh. """
        axis = []
        for i, x in enumerate(slic):
            if isinstance(x, (integer, int)):
                axis.append(i)

        assert not len(axis) == 3, ValueError("Slic cannot be a single cell")

        if len(axis) == 0: # Returning a 3D mesh
            x = self.x[slic[0]]
            y = self.y[slic[1]]
            z = self.z[slic[2]]
            if x._relativeTo is not None:
                nd = ndim(x.relativeTo)
                slc = s_[:]
                if nd == 2:
                    slc = slic[1:]
                elif nd == 1:
                    s = x.relativeTo.size
                    if s == self.shape[1]:
                        slc = slic[1]
                    elif s == self.shape[2]:
                        slc = slic[2]

                x.relativeTo = x.relativeTo[slc]

            if y._relativeTo is not None:
                nd = ndim(y.relativeTo)
                slc = s_[:]
                if nd == 2:
                    slc = slic[::2]
                elif nd == 1:
                    s = y.relativeTo.size
                    if s == self.shape[0]:
                        slc = slic[0]
                    elif s == self.shape[2]:
                        slc = slic[2]

                y.relativeTo = y.relativeTo[slc]

            if z._relativeTo is not None:
                nd = ndim(z.relativeTo)
                slc = s_[:]
                if nd == 2:
                    slc = slic[:2]
                elif nd == 1:
                    s = z.relativeTo.size
                    if s == self.shape[0]:
                        slc = slic[0]
                    elif s == self.shape[1]:
                        slc = slic[1]

                z.relativeTo = z.relativeTo[slc]

            out = type(self)(x=x, y=y, z=z)
            return out

        if len(axis) == 1: # Returning a 2D mesh
            a = [x for x in (0, 1, 2) if not x in axis]
            b = [x for x in (0, 1, 2) if x in axis]

            x = deepcopy(self.axis(a[0]))
            if x._relativeTo is not None:
                if x._relativeTo.size > 1:
                    if a[0] == 0:
                        if b[0] == 1:
                            axis = 0
                        if b[0] == 2:
                            axis = 1
                    elif a[0] == 1:
                        if b[0] == 0:
                            axis = 0
                        if b[0] == 2:
                            axis = 'OOPS'
                    x.relativeTo = take(x.relativeTo, slic[b[0]], axis)

            y = deepcopy(self.axis(a[1]))
            if y._relativeTo is not None:
                if y._relativeTo.size > 1:
                    if a[1] == 1:
                        if b[0] == 0:
                            axis = 0
                        elif b[0] == 2:
                            axis = 1
                    elif a[1] == 2:
                        if b[0] == 0:
                            axis = 0
                        elif b[0] == 1:
                            axis = 1

                    y.relativeTo = take(y.relativeTo, slic[b[0]], axis)

            x = x[slic[a[0]]]
            x.dimension = 0
            y = y[slic[a[1]]]
            y.dimension = 1

            out = RectilinearMesh2D(x=x, y=y)

        else: # Returning a 1D mesh
            a = [x for x in (0, 1, 2) if not x in axis]
            b = [x for x in (0, 1, 2) if x in axis]

            out = self.axis(a[0])[slic[a[0]]]
            if out._relativeTo is not None:
                out.relativeTo = out.relativeTo[slic[b[0]], slic[b[1]]]

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
        if self.x._relativeTo is not None:
            re_tmp = deepcopy(self.x.relativeTo)
            nd = ndim(self.x.relativeTo)
            if nd == 2:
                mesh = self.remove_axis(0)
                re_nodes = mesh.interpolate_centres_to_nodes(self.x.relativeTo)
                self.x.relativeTo = re_nodes

        out = super().x_edges

        if re_tmp is not None:
            self.x.relativeTo = re_tmp

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
        if self.y._relativeTo is not None:
            re_tmp = deepcopy(self.y.relativeTo)
            nd = ndim(self.y.relativeTo)
            if nd == 2:
                mesh = self.remove_axis(1)
                re_nodes = mesh.interpolate_centres_to_nodes(self.y.relativeTo)
                self.y.relativeTo = re_nodes

        out = super().y_edges

        if re_tmp is not None:
            self.y.relativeTo = re_tmp

        nd = ndim(out)
        if nd == 3:
            return out
        elif nd == 2:
            return repeat(out[:, :, None], self.z.nEdges, 2)
        elif nd == 1:
            return repeat(repeat(out[None, :], self.x.nEdges, 0)[:, :, None], self.z.nEdges, 2)

    @property
    def z_edges(self):

        re_tmp = None
        if self.z._relativeTo is not None:
            nd = ndim(self.z.relativeTo)
            if nd == 2:
                re_tmp = deepcopy(self.z.relativeTo)
                mesh = self.remove_axis(2)
                re_nodes = mesh.interpolate_centres_to_nodes(self.z.relativeTo)
                self.z.relativeTo = re_nodes

        out = self.z.edges_absolute

        if re_tmp is not None:
            self.z.relativeTo = re_tmp

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
                        relativeTo=values.get('z_relative_to'),
                        dimension=2)

        assert isinstance(values, RectilinearMesh1D), TypeError('z must be a RectilinearMesh1D')
        assert values.dimension == 2
        self._z = values

        # if self.z._relativeTo is not None:
        #     assert all(self.z.relativeTo.shape == self.shape[:2]), "z axis relative to must have shape {}".format(self.shape[:2])

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
            tmp, _ = utilities._log(values[tuple(slic)].T.flatten(), kwargs.get('log', None))
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
            probability[j] = dot(p, pdf[j])
        probability = probability / expand_dims(sum(probability, axis), axis=axis)

        return StatArray.StatArray(probability, name='marginal_probability')

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
            flag = self.x.inBounds(x) & self.y.inBounds(y) & self.z.inBounds(z)
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
    #     if a._relativeTo is None:
    #         return super()._mean(values, axis)

    #     s = tuple([s_[:] if i == axis else None for i in range(self.ndim)])

    #     centres = self.centres_absolute(axis)

    #     t = sum(centres * values, axis = axis)
    #     s = values.sum(axis = axis)

    #     if size(t) == 1:
    #         out = t / s
    #     else:
    #         i = where(s > 0.0)
    #         out = StatArray.StatArray(t.shape)
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


    def plotGrid(self):
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


    # def pcolor(self, values, x='x', **kwargs):
    #     """Create a pseudocolour plot.

    #     Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

    #     Parameters
    #     ----------
    #     values : array_like
    #         2D array of colour values
    #     location : float
    #         location of axis aligned slice to pcolor
    #     xAxis : str
    #         If xAxis is 'x', the horizontal axis uses self.x
    #         If xAxis is 'y', the horizontal axis uses self.y
    #         If xAxis is 'r', the horizontal axis uses cumulative distance along the line

    #     Other Parameters
    #     ----------------
    #     alpha : scalar or array_like, optional
    #         If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
    #         If array_like, each pixel is given an individual alpha value.
    #     log : 'e' or float, optional
    #         Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
    #         Values in c that are <= 0 are masked.
    #     equalize : bool, optional
    #         Equalize the histogram of the colourmap so that all colours have an equal amount.
    #     nbins : int, optional
    #         Number of bins to use for histogram equalization.
    #     xscale : str, optional
    #         Scale the x axis? e.g. xscale = 'linear' or 'log'
    #     yscale : str, optional
    #         Scale the y axis? e.g. yscale = 'linear' or 'log'.
    #     flipX : bool, optional
    #         Flip the X axis
    #     flipY : bool, optional
    #         Flip the Y axis
    #     grid : bool, optional
    #         Plot the grid
    #     noColorbar : bool, optional
    #         Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
    #     trim : bool, optional
    #         Set the x and y limits to the first and last non zero values along each axis.

    #     See Also
    #     --------
    #     geobipy.plotting.pcolor : For non matplotlib keywords.
    #     matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

    #     """

    #     assert all(values.shape == self.shape), ValueError("values must have shape {}".format(self.shape))

    #     xtmp = self.axis(x)

    #     ax, pm, cb = cP.pcolor(values, x = xtmp, y = self.z.edges, **kwargs)

    #     return ax, pm, cb


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
              "Shape: : {} \nx\n{}y\n{}z\n{}").format(self.shape, self.x.summary, self.y.summary, self.z.summary)
        # if not self.relativeTo is None:
        #     msg += self.relativeTo.summary
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
        # if not self.relativeTo is None:
        #     self.relativeTo.createHdf(grp, 'relativeTo', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

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
        # if not self.relativeTo is None:
        #     self.relativeTo.writeHdf(grp, 'relativeTo', withPosterior=withPosterior, index=index)

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

            if y._relativeTo is not None:
                nd = ndim(y.relativeTo)
                if nd == 1:
                    y.relativeTo = repeat(y.relativeTo[:, None], z.nCells, 1)
            if z._relativeTo is not None:
                nd = ndim(z.relativeTo)
                if nd == 1:
                    z.relativeTo = repeat(z.relativeTo[:, None], y.nCells, 1)

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
    #         assert isinstance(pointData, (StatArray.StatArray, list)), TypeError("pointData must a geobipy.StatArray or a list of them.")
    #         if isinstance(pointData, list):
    #             for p in pointData:
    #                 assert isinstance(p, StatArray.StatArray), TypeError("pointData entries must be a geobipy.StatArray")
    #                 assert all(p.shape == [self.z.nEdges, self.x.nEdges]), ValueError("pointData entries must have shape {}".format([self.z.nEdges, self.x.nEdges]))
    #                 assert p.hasLabels(), ValueError("StatArray needs a name")
    #                 vtk.point_data.append(Scalars(p.reshape(self.nNodes), p.getNameUnits()))
    #         else:
    #             assert all(pointData.shape == [self.z.nEdges, self.x.nEdges]), ValueError("pointData entries must have shape {}".format([self.z.nEdges, self.x.nEdges]))
    #             assert pointData.hasLabels(), ValueError("StatArray needs a name")
    #             vtk.point_data.append(Scalars(pointData.reshape(self.nNodes), pointData.getNameUnits()))

    #     if not cellData is None:
    #         assert isinstance(cellData, (StatArray.StatArray, list)), TypeError("cellData must a geobipy.StatArray or a list of them.")
    #         if isinstance(cellData, list):
    #             for p in cellData:
    #                 assert isinstance(p, StatArray.StatArray), TypeError("cellData entries must be a geobipy.StatArray")
    #                 assert all(p.shape == self.shape), ValueError("cellData entries must have shape {}".format(self.shape))
    #                 assert p.hasLabels(), ValueError("StatArray needs a name")
    #                 vtk.cell_data.append(Scalars(p.reshape(self.nCells), p.getNameUnits()))
    #         else:
    #             assert all(cellData.shape == self.shape), ValueError("cellData entries must have shape {}".format(self.shape))
    #             assert cellData.hasLabels(), ValueError("StatArray needs a name")
    #             vtk.cell_data.append(Scalars(cellData.reshape(self.nCells), cellData.getNameUnits()))

    #     vtk.tofile(fileName, format)
