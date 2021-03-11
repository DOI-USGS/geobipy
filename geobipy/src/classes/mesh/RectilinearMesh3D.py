""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import deepcopy
from ...classes.core.myObject import myObject
from ...classes.core import StatArray
from .RectilinearMesh1D import RectilinearMesh1D
import numpy as np
from scipy.stats import binned_statistic
from ...base import plotting as cP
from ...base import utilities as cF
from scipy.sparse import kron

try:
    from pyvtk import VtkData, CellData, Scalars, PolyData
except:
    pass


class RectilinearMesh3D(myObject):
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

    def __init__(self, xCentres=None, xEdges=None, yCentres=None, yEdges=None, zCentres=None, zEdges=None, **kwargs):
        """ Initialize a 2D Rectilinear Mesh"""

        self._x = None
        self._y = None
        self._z = None
        self._distance = None
        self.xyz = None

        if (all(x is None for x in [xCentres, yCentres, zCentres, xEdges, yEdges, zEdges])):
            return

        xExtras = dict((k[1:], kwargs.pop(k, None)) for k in ('xedgesMin', 'xedgesMax', 'xlog'))
        self._x = RectilinearMesh1D(cellCentres=xCentres, cellEdges=xEdges, relativeTo=kwargs.pop('xrelativeTo', 0.0), **xExtras)

        yExtras = dict((k[1:], kwargs.pop(k, None)) for k in ('yedgesMin', 'yedgesMax', 'ylog'))
        self._y = RectilinearMesh1D(cellCentres=yCentres, cellEdges=yEdges, relativeTo=kwargs.pop('yrelativeTo', 0.0),  **yExtras)

        zExtras = dict((k[1:], kwargs.pop(k, None)) for k in ('zedgesMin', 'zedgesMax', 'zlog'))
        self._z = RectilinearMesh1D(cellCentres=zCentres, cellEdges=zEdges, relativeTo=kwargs.pop('zrelativeTo', 0.0), **zExtras)



    def __getitem__(self, slic):
        """Slice into the mesh. """

        slic = []
        axis = []
        for i, x in enumerate(slic0):
            if not isinstance(x, int):
                tmp = x
                if isinstance(x.stop, int):
                    tmp = slice(x.start, x.stop+1, x.step) # If a slice, add one to the end for bins.
            else:
                tmp = x
                axis.append(i)

            slic.append(tmp)

        assert not len(axis) == 3, ValueError("Slic cannot be a single cell")
        slic = tuple(slic)

        if len(axis) == 0:
            out = RectilinearMesh3D(xBins=self._x.cellEdges[slic[2]], yBins=self._y.cellEdges[slic[1]], zBins=self._z.cellEdges[slic[0]])
            return out

        if len(axis) == 1:
            a = [x for x in (0, 1, 2) if not x in axis]
            out = RectilinearMesh2D(xBins=self.axis(a[1]).cellEdges[slic[a[1]]], yBins=self.axis(a[0]).cellEdges[slic[a[0]]])
        else:
            a = [x for x in (0, 1, 2) if not x in axis][0]
            out = RectilinearMesh1D(bins=self.axis(a).cellEdges[slic[a]])

        return out

    def other_axis(self, axis):

        if axis == 0:
            return self.y, self.z
        elif axis == 1:
            return self.x, self.z
        elif axis == 2:
            return self.x, self.y


    def axis(self, axis):
        if axis == 0:
            return self.x
        elif axis == 1:
            return self.y
        elif axis == 2:
            return self.z

    def other_axis_indices(self, axis):
        if axis == 0:
            return 0, 1
        elif axis == 1:
            return 0, 2
        elif axis == 2:
            return 1, 2


    # @property
    # def distance(self):
    #     """The distance along the top of the mesh using the x and y co-ordinates. """

    #     assert self.xyz, Exception("To set the distance, the mesh must be instantiated with three co-ordinates")

    #     if self._distance is None:

    #         dx = np.diff(self.x.cellEdges)
    #         dy = np.diff(self.y.cellEdges)

    #         distance = StatArray.StatArray(np.zeros(self.x.nEdges), 'Distance', self.x.cellCentres.units)
    #         distance[1:] = np.cumsum(np.sqrt(dx**2.0 + dy**2.0))

    #         self._distance = RectilinearMesh1D(cellEdges = distance)
    #     return self._distance


    @property
    def nCells(self):
        """The number of cells in the mesh.

        Returns
        -------
        out : int
            Number of cells

        """

        return np.prod(self.shape)


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

        return (self.z.nCells, self.y.nCells, self.x.nCells)


    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z


    def _mean(self, values, log=None, axis=0):

        a = self.axis(axis)
        b, c = self.other_axis(axis)
        i, j = self.other_axis_indices(axis)

        tmp = np.expand_dims(a.cellCentres, (i, j))
        tmp = np.repeat(tmp, c.nCells, i)
        tmp = np.repeat(tmp, b.nCells, j)

        t = np.sum(tmp * values, axis = 2-axis)
        s = values.sum(axis = 2-axis)

        i = np.where(s > 0.0)
        out = StatArray.StatArray(t.shape)
        out[i] = t[i] / s[i]

        if log:
            out, dum = cF._log(out, log=log)

        return out


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
        return self._percent_interval(values, 50.0, log, axis)


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
        total = values.sum(axis=2-axis)
        # Cumulative sum
        cs = np.cumsum(values, axis=2-axis)
        # # Cumulative "probability"
        tmp = np.divide(cs, np.expand_dims(total, 2-axis))
        # # Find the interval
        i = np.apply_along_axis(np.searchsorted, 2-axis, tmp, percent)
        # # Obtain the values at those locations
        out = self.axis(axis).cellCentres[i]

        if (not log is None):
            out, dum = cF._log(out, log=log)

        return out


    def deepcopy(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        """ Define the deepcopy for the StatArray """
        return RectilinearMesh2D(xEdges=self.x.cellEdges, yEdges=self.y.cellEdges, zEdges=self.z.cellEdges)


    def cellEdges(self, axis):
        """ Gets the cell edges in the given dimension """
        return self.axis(axis).cellEdges


    def getXAxis(self, axis='x', centres=False):
        assert axis in ['x', 'y', 'r'], Exception("axis must be either 'x', 'y' or 'r'")
        if axis == 'x':
            return self.x.cellCentres if centres else self.x.cellEdges
        elif axis == 'y':
            assert self.xyz, Exception("To plot against 'y' the mesh must be instantiated with three co-ordinates")
            return self.y.cellCentres if centres else self.y.cellEdges
        elif axis == 'r':
            assert self.xyz, Exception("To plot against 'r' the mesh must be instantiated with three co-ordinates")
            return self.distance.cellCentres if centres else self.distance.cellEdges


    # def xGradientMatrix(self):
    #     tmp = self.x.gradientMatrix()
    #     return kron(np.diag(np.sqrt(self.z.cellWidths)), tmp)


    # def zGradientMatrix(self):
    #     nx = self.x.nCells
    #     nz = self.z.nCells
    #     tmp = 1.0 / np.sqrt(rm2.x.centreTocentre)
    #     a = np.repeat(tmp, nz) * np.tile(np.sqrt(rm2.z.cellWidths), nx-1)
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
            bins = binned_statistic(self.z.cellCentres, arr.T, bins = intervals, statistic=statistic)
            res = bins.statistic.T
        else:
            bins = binned_statistic(self.x.cellCentres, arr, bins = intervals, statistic=statistic)
            res = bins.statistic

        return res, intervals


    def _reconcile_intervals(self, intervals, axis=0):

        assert np.size(intervals) > 1, ValueError("intervals must have size > 1")

        ax

        if (axis == 0):
            # Make sure the intervals are within the axis.
            i0 = np.maximum(0, np.searchsorted(intervals, self.z.cellEdges[0]))
            i1 = np.minimum(self.z.nCells, np.searchsorted(intervals, self.z.cellEdges[-1])+1)
            intervals = intervals[i0:i1]

        else:
            i0 = np.maximum(0, np.searchsorted(intervals, self.x.cellEdges[0]))
            i1 = np.minimum(self.x.nCells, np.searchsorted(intervals, self.x.cellEdges[-1])+1)
            intervals = intervals[i0:i1]

        return intervals


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
        assert (np.size(x) == np.size(y) == np.size(z)), ValueError("x, y, z must have the same size")
        if trim:
            flag = self.x.inBounds(x) & self.y.inBounds(y) & self.z.inBounds(z)
            i = np.where(flag)[0]
            out = np.empty([3, i.size], dtype=np.int32)
            out[2, :] = self.x.cellIndex(x[i])
            out[1, :] = self.y.cellIndex(y[i])
            out[0, :] = self.z.cellIndex(z[i])
        else:
            out = np.empty([3, np.size(x)], dtype=np.int32)
            out[2, :] = self.x.cellIndex(x, clip=clip)
            out[1, :] = self.y.cellIndex(y, clip=clip)
            out[0, :] = self.z.cellIndex(z, clip=clip)
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


    # def pcolor(self, values, xAxis='x', **kwargs):
    #     """Create a pseudocolour plot.

    #     Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

    #     Parameters
    #     ----------
    #     values : array_like
    #         2D array of colour values
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

    #     assert np.all(values.shape == self.shape), ValueError("values must have shape {}".format(self.shape))

    #     xtmp = self.getXAxis(xAxis)

    #     ax, pm, cb = cP.pcolor(values, x = xtmp, y = self.z.cellEdges, **kwargs)

    #     return ax, pm, cb


    def plot_pyvista(self, **kwargs):
        """Creates a pyvista plotting object linked to VTK.

        Use mesh.plot(show_edges=True, show_grid=True) to plot the mesh.

        Returns
        -------

        """
        import pyvista as pv

        x, y, z = np.meshgrid(self.x.cellEdges, self.y.cellEdges, self.z.cellEdges)

        mesh = pv.StructuredGrid(x, y, z)

        return mesh

    @property
    def summary(self):
        return


    # def plotXY(self, **kwargs):
    #     """Plot the cell centres in x and y as points"""

    #     assert self.xyz, Exception("Mesh must be instantiated with three co-ordinates to use plotXY()")

    #     kwargs['marker'] = kwargs.pop('marker', 'o')
    #     kwargs['linestyle'] = kwargs.pop('linestyle', 'none')

    #     self.y.cellCentres.plot(x=self.x.cellCentres, **kwargs)



    def createHdf(self, parent, name, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)
        self.x.createHdf(grp,'x', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.y.createHdf(grp,'y', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.z.createHdf(grp,'z', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)

        return grp


    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        grp = h5obj.get(name)
        self.x.writeHdf(grp, 'x',  withPosterior=withPosterior, index=index)
        self.y.writeHdf(grp, 'y',  withPosterior=withPosterior, index=index)
        self.z.writeHdf(grp, 'z',  withPosterior=withPosterior, index=index)


    def fromHdf(self, grp, index=None):
        """ Reads in the object from a HDF file """
        x = RectilinearMesh1D().fromHdf(grp['x'], index=index)
        y = RectilinearMesh1D().fromHdf(grp['y'], index=index)
        z = RectilinearMesh1D().fromHdf(grp['z'], index=index)
        self.__init__(xCentres=x, yCentres=y, zCentres=z)
        return self


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


    def vtkStructure(self):
        """Generates a vtk mesh structure that can be used in a vtk file.

        Returns
        -------
        out : pyvtk.VtkData
            Vtk data structure

        """

        # Generate the quad node locations in x
        x = self.x.cellEdges
        y = self.y.cellEdges
        z = self.z.cellEdges

        nCells = self.x.nCells * self.z.nCells

        z = self.z.cellEdges
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
