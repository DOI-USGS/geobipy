""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import deepcopy
from ...classes.core.myObject import myObject
from ...classes.core.StatArray import StatArray
from .RectilinearMesh1D import RectilinearMesh1D
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from ...base import customPlots as cP
from ...base.customFunctions import safeEval
from ...base.customFunctions import _logSomething, isInt

try:
    from pyvtk import VtkData, CellData, Scalars, PolyData
except:
    pass


class RectilinearMesh2D(myObject):
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

    Returns
    -------
    out : RectilinearMesh2D
        The 2D mesh.

    """


    def __init__(self, xCentres=None, xEdges=None, yCentres=None, yEdges=None, zCentres=None, zEdges=None):
        """ Initialize a 2D Rectilinear Mesh"""

        self.x = None
        self.y = None
        self.z = None
        self.distance = None
        self.xyz = None

        if (all(x is None for x in [xCentres, yCentres, zCentres, xEdges, yEdges, zEdges])):
            return

        # RectilinearMesh1D of the x axis values
        self.x = RectilinearMesh1D(cellCentres=xCentres, cellEdges=xEdges)
        # StatArray of the y axis values
        self.y = RectilinearMesh1D(cellCentres=yCentres, cellEdges=yEdges)

        if (not zCentres is None or not zEdges is None):
            assert self.x.nCells == self.y.nCells, Exception("x and y axes must have the same number of cells.")
            # StatArray of third axis
            self.z = RectilinearMesh1D(cellCentres=zCentres, cellEdges=zEdges)
            self.xyz = True
            self.setDistance()
            
        else:
            self.xyz = False
            self.z = self.y     


    @property
    def dims(self):
        """The dimensions of the mesh

        Returns
        -------
        out : array_like
            Array of integers

        """

        return np.asarray([self.z.nCells, self.x.nCells], dtype=np.int)


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


    def axisMedian(self, arr, log=None, axis=0):
        """Gets the median for the specified axis.
        
        Parameters
        ----------
        arr : array_like
            2D array to get the median.
        log : 'e' or float, optional
            Take the log of the median to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the median.

        Returns
        -------
        med : array_like
            Contains the locations of the medians along the specified axis. Has size equal to arr.shape[axis].

        """

        total = arr.sum(axis = 1-axis)
        tmp = np.cumsum(arr, axis = 1-axis)

        if axis == 0:
            tmp = tmp / np.repeat(total[:, np.newaxis], arr.shape[1], 1)
        else:
            tmp = tmp / np.repeat(total[np.newaxis, :], arr.shape[0], 0)

        ixM = np.apply_along_axis(np.searchsorted, 1-axis, tmp, 0.5)

        if axis == 0:
            med = self.x.cellCentres[ixM]
        else:
            med = self.z.cellCentres[ixM]

        if (not log is None):
            med, dum = _logSomething(med, log=log)

        return med


    def deepcopy(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        """ Define the deepcopy for the StatArray """
        other = RectilinearMesh2D(xEdges=self.x.cellEdges, zEdges=self.z.cellEdges)
        # other.arr[:,:] += self.arr
        return other


    def cellEdges(self, axis):
        """ Gets the cell edges in the given dimension """
        return self.z.cellEdges if axis == 0 else self.x.cellEdges


    def hasSameSize(self, other):
        """ Determines if the meshes have the same dimension sizes """
        # if self.arr.shape != other.arr.shape:
        #     return False
        if self.x.nCells != other.x.nCells:
            return False
        if self.z.nCells != other.y.nCells:
            return False
        return True


    def intervalMean(self, arr, intervals, axis=0, statistic='mean'):
        """Compute the mean of an array between the intervals given along dimension dim.
        
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

        if (axis == 0):
            bins = binned_statistic(self.z.cellCentres, arr.T, bins = intervals, statistic=statistic)
            res = bins.statistic.T
        else:
            bins = binned_statistic(self.x.cellCentres, arr, bins = intervals, statistic=statistic)
            res = bins.statistic

        return res


    def cellIndices(self, x1, x2, clip=False):
        """Return the cell indices in x and z for two floats.

        Parameters
        ----------
        x1 : float
            x location
        x2 : float
            y location (or z location if instantiated with 3 co-ordinates)
        clip : bool
            A negative index which would normally wrap will clip to 0 instead.

        Returns
        -------
        out : ints
            indices for the locations along [axis0, axis1]

        """
        out = np.empty(2, dtype=np.int32)
        out[1] = self.x.cellIndex(x1, clip=clip)
        out[0] = self.z.cellIndex(x2, clip=clip)
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

        return np.ravel_multi_index(ixy, self.dims, order=order)

    
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
            Each array in the tuple has the same shape as the self.dims.

        """

        return np.unravel_index(indices, self.dims, order=order)


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
            Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.   
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.
        
        See Also
        --------
        geobipy.customPlots.pcolor : For non matplotlib keywords.
        matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

        """

        assert np.all(values.shape == self.dims), ValueError("values must have shape {}".format(self.dims))

        xtmp = self.getXAxis(xAxis)

        ax = cP.pcolor(values, x = xtmp, y = self.z.cellEdges, **kwargs)
        
        return ax


    def plotGrid(self, xAxis='x', **kwargs):
        """Plot the mesh grid lines. 
        
        Parameters
        ----------
        xAxis : str
            If xAxis is 'x', the horizontal axis uses self.x
            If xAxis is 'y', the horizontal axis uses self.y
            If xAxis is 'r', the horizontal axis uses sqrt(self.x^2 + self.y^2)
        
        """

        tmp = StatArray(np.full([self.z.nCells, self.x.nCells], fill_value=np.nan))

        xtmp = self.getXAxis(xAxis)

        tmp.pcolor(x=xtmp, y=self.z.cellEdges, grid=True, noColorbar=True, **kwargs)

    
    def setDistance(self):
        assert self.xyz, Exception("To set the distance, the mesh must be instantiated with three co-ordinates")
        dx = np.diff(self.x.cellEdges)
        dy = np.diff(self.y.cellEdges)
        distance = StatArray(np.zeros(self.x.nEdges), 'Distance', self.x.cellCentres.units)
        distance[1:] = np.cumsum(np.sqrt(dx**2.0 + dy**2.0))
        self.distance = RectilinearMesh1D(cellEdges = distance)


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


    def plotXY(self, **kwargs):
        """Plot the cell centres in x and y as points"""

        assert self.xyz, Exception("Mesh must be instantiated with three co-ordinates to use plotXY()")

        kwargs['marker'] = kwargs.pop('marker', 'o')
        kwargs['linestyle'] = kwargs.pop('linestyle', 'none')

        self.y.cellCentres.plot(x=self.x.cellCentres, **kwargs)


    def hdfName(self):
        """ Reprodicibility procedure """
        return('Rmesh2D()')


    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self._counts.createHdf(grp, 'arr', nRepeats=nRepeats, fillvalue=fillvalue)
        self.x.createHdf(grp,'x', nRepeats=nRepeats, fillvalue=fillvalue)
        self.y.createHdf(grp,'y', nRepeats=nRepeats, fillvalue=fillvalue)
        self.z.createHdf(grp,'z', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        ai = None
        bi = None
        if not index is None:
            assert isInt(index), ValueError('index must be an integer')
            ai = np.s_[index,:,:]
            bi = np.s_[index,:]

        self._counts.writeHdf(parent, myName+'/arr',  index=ai)
        self.x.writeHdf(parent, myName+'/x',  index=bi)
        self.y.writeHdf(parent, myName+'/y',  index=bi)
        self.z.writeHdf(parent, myName+'/z',  index=bi)


    def toHdf(self, h5obj, myName):
        """ Write the StatArray to an HDF object
        h5obj: :An HDF File or Group Object.
        """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self._counts.toHdf(grp, 'arr')
        self.x.toHdf(grp, 'x')
        self.y.toHdf(grp, 'y')
        self.z.toHdf(grp, 'z')


    def fromHdf(self, grp, index=None):
        """ Reads in the object from a HDF file """

        ai=None
        bi=None
        if (not index is None):
            assert isInt(index), ValueError('index must be an integer')
            ai = np.s_[index, :, :]
            bi = np.s_[index, :]

        item = grp.get('arr')
        obj = eval(safeEval(item.attrs.get('repr')))
        arr = obj.fromHdf(item, index=ai)
        item = grp.get('x')
        obj = eval(safeEval(item.attrs.get('repr')))
        x = obj.fromHdf(item, index=bi)
        item = grp.get('y')
        obj = eval(safeEval(item.attrs.get('repr')))
        y = obj.fromHdf(item, index=bi)
        tmp = RectilinearMesh2D(x, y)
        tmp._counts = arr
        return tmp


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
            assert isinstance(pointData, (StatArray, list)), TypeError("pointData must a geobipy.StatArray or a list of them.")
            if isinstance(pointData, list):
                for p in pointData:
                    assert isinstance(p, StatArray), TypeError("pointData entries must be a geobipy.StatArray")
                    assert all(p.shape == [self.z.nEdges, self.x.nEdges]), ValueError("pointData entries must have shape {}".format([self.z.nEdges, self.x.nEdges]))
                    assert p.hasLabels(), ValueError("StatArray needs a name")
                    vtk.point_data.append(Scalars(p.reshape(self.nNodes), p.getNameUnits()))
            else:
                assert all(pointData.shape == [self.z.nEdges, self.x.nEdges]), ValueError("pointData entries must have shape {}".format([self.z.nEdges, self.x.nEdges]))
                assert pointData.hasLabels(), ValueError("StatArray needs a name")
                vtk.point_data.append(Scalars(pointData.reshape(self.nNodes), pointData.getNameUnits()))

        if not cellData is None:
            assert isinstance(cellData, (StatArray, list)), TypeError("cellData must a geobipy.StatArray or a list of them.")
            if isinstance(cellData, list):
                for p in cellData:
                    assert isinstance(p, StatArray), TypeError("cellData entries must be a geobipy.StatArray")
                    assert all(p.shape == self.dims), ValueError("cellData entries must have shape {}".format(self.dims))
                    assert p.hasLabels(), ValueError("StatArray needs a name")
                    vtk.cell_data.append(Scalars(p.reshape(self.nCells), p.getNameUnits()))
            else:
                assert all(cellData.shape == self.dims), ValueError("cellData entries must have shape {}".format(self.dims))
                assert cellData.hasLabels(), ValueError("StatArray needs a name")
                vtk.cell_data.append(Scalars(cellData.reshape(self.nCells), cellData.getNameUnits()))

        vtk.tofile(fileName, format)