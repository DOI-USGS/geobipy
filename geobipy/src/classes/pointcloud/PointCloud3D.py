from copy import deepcopy
from ...classes.core.myObject import myObject
import numpy as np
from ...classes.core import StatArray
from ...base import fileIO as fIO
from ...base import interpolation
from ...base import customFunctions as cf
from ...base import customPlots as cP
from .Point import Point
from ...base import MPI
from scipy.spatial import cKDTree

try:
    from pyvtk import VtkData, Scalars, PolyData, PointData, UnstructuredGrid
except:
    pass


class PointCloud3D(myObject):
    """3D Point Cloud with x,y,z co-ordinates

    PointCloud3D(N, x, y, z)

    Parameters
    ----------
    N : int
        Number of points
    x : array_like or geobipy.StatArray, optional
        The x co-ordinates. Default is zeros of size N
    y : array_like or geobipy.StatArray, optional
        The y co-ordinates. Default is zeros of size N
    z : array_like or geobipy.StatArray, optional
        The z co-ordinates. Default is zeros of size N
    units : str, optional
        The units of the co-ordinates.  Default is "m"

    Returns
    -------
    out : geobipy.PointCloud3D
        The 3D point cloud

    """

    def __init__(self, x=None, y=None, z=None, elevation=None):
        """ Initialize the class """

        # Number of points in the cloud
        self.nPoints = 0

        # StatArray of the x co-ordinates
        self.x = x

        # StatArray of the y co-ordinates
        self.y = y

        # StatArray of the z co-ordinates
        self.z = z

        # StatArray of elevation
        self.elevation = elevation


    def __getitem__(self, i):
        """Define get item

        Parameters
        ----------
        i : ints or slice
            The indices of the points in the pointcloud to return

        out : geobipy.PointCloud3D
            The potentially smaller point cloud

        """
        i = np.unique(i)
        return PointCloud3D(self.x[i],
                            self.y[i],
                            self.z[i],
                            self.elevation[i])


    @property
    def x(self):
        return self._x


    @x.setter
    def x(self, values):
        if (values is None):
            self._x = StatArray.StatArray(self.nPoints, "Easting", "m")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("x must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._x = values.deepcopy()
            else:
                self._x = StatArray.StatArray(values, "Easting", "m")


    @property
    def y(self):
        return self._y


    @y.setter
    def y(self, values):
        if (values is None):
            self._y = StatArray.StatArray(self.nPoints, "Northing", "m")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("y must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._y = values.deepcopy()
            else:
                self._y = StatArray.StatArray(values, "Northing", "m")


    @property
    def z(self):
        return self._z


    @z.setter
    def z(self, values):
        if (values is None):
            self._z = StatArray.StatArray(self.nPoints, "Height", "m")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("z must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._z = values.deepcopy()
            else:
                self._z = StatArray.StatArray(values, "Height", "m")


    @property
    def elevation(self):
        return self._elevation


    @elevation.setter
    def elevation(self, values):
        if (values is None):
            self._elevation = StatArray.StatArray(self.nPoints, "Elevation", "m")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("elevation must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._elevation = values.deepcopy()
            else:
                self._elevation = StatArray.StatArray(values, "Elevation", "m")


    @property
    def nPoints(self):
        """Get the number of points"""
        return self._nPoints


    @nPoints.setter
    def nPoints(self, value):
        assert isinstance(value, np.int), TypeError("nPoints must be an integer")
        self._nPoints = value


    @property
    def scalar(self):
        assert self.size == 1, ValueError("Cannot return array as scalar")
        return self[0]


    def append(self, other):
        """Append pointclouds together

        Parameters
        ----------
        other : geobipy.PointCloud3D
            3D pointcloud

        """

        self.nPoints = self.nPoints + other.nPoints
        self.x = np.hstack([self.x, other.x])
        self.y = np.hstack([self.y, other.y])
        self.z = np.hstack([self.z, other.z])
        self.elevation = np.hstack([self.elevation, other.elevation])


    @property
    def bounds(self):
        """Gets the bounding box of the data set """
        return np.asarray([np.nanmin(self.x), np.nanmax(self.x), np.nanmin(self.y), np.nanmax(self.y)])


    def getPoint(self, i):
        """Get a point from the 3D Point Cloud

        Parameters
        ----------
        i : int
            The index of the point to return

        Returns
        -------
        out : geobipy.Point
            A point

        Raises
        ------
        ValueError : If i is not a single int

        """
        assert np.size(i) == 1, ValueError("i must be a single integer")
        assert 0 <= i <= self.nPoints, ValueError("Must have 0 <= i <= {}".format(self.nPoints))
        return Point(self.x[i], self.y[i], self.z[i])


    def getXAxis(self, xAxis='x'):
        """Obtain the xAxis against which to plot values.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'index', returns numpy.arange(self.nPoints)
            If xAxis is 'x', returns self.x
            If xAxis is 'y', returns self.y
            If xAxis is 'z', returns self.z
            If xAxis is 'r2d', returns cumulative distance along the line in 2D using x and y.
            If xAxis is 'r3d', returns cumulative distance along the line in 3D using x, y, and z.

        Returns
        -------
        out : array_like
            The requested xAxis.

        """
        assert xAxis in ['index', 'x', 'y', 'z', 'r2d', 'r3d'], Exception("xAxis must be either 'index', x', 'y', 'z', 'r2d', or 'r3d'")
        if xAxis == 'index':
            return StatArray.StatArray(np.arange(self.x.size), name="Index")
        elif xAxis == 'x':
            return self.x
        elif xAxis == 'y':
            return self.y
        elif xAxis == 'z':
            return self.z
        elif xAxis == 'r2d':
            r = np.diff(self.x)**2.0
            r += np.diff(self.y)**2.0
            distance = StatArray.StatArray(np.zeros(self.x.size), 'Distance', self.x.units)
            distance[1:] = np.cumsum(np.sqrt(r))
            return distance
        elif xAxis == 'r3d':
            r = np.diff(self.x)**2.0
            r += np.diff(self.y)**2.0
            r += np.diff(self.z)**2.0
            distance = StatArray.StatArray(np.zeros(self.x.size), 'Distance', self.x.units)
            distance[1:] = np.cumsum(np.sqrt(r))
            return distance


    def interpolate(self, dx, dy, values, method='ct', mask = False, clip = True, i=None, **kwargs):

        if method.lower() == 'ct':
            return self.interpCloughTocher(dx, dy, values, mask, clip, i, **kwargs)
        else:
            return self.interpMinimumCurvature(dx, dy, values, mask, clip, i, **kwargs)

    def interpCloughTocher(self, dx, dy, values, mask = False, clip = None, i=None, **kwargs):
        """ Interpolate values at the points to a grid """

        extrapolate = kwargs.pop('extrapolate', None)
        # Get the bounding box

        assert dx > 0.0, ValueError("dx must be positive!")
        assert dy > 0.0, ValueError("dy must be positive!")

        kdtree = None
        if (not i is None):
            xTmp = self.x[i]
            yTmp = self.y[i]
            vTmp = values[i]
            tmp = np.column_stack((xTmp, yTmp))
            if (mask or extrapolate):
                kdtree = cKDTree(tmp)
        else:
            tmp = np.column_stack((self.x, self.y))
            vTmp = values
            if (mask or extrapolate):
                self.setKdTree(nDims = 2)
                kdtree = self.kdtree

        xc, yc, vals = interpolation.CT(dx, dy, self.bounds, tmp, vTmp , mask = mask, kdtree = kdtree, clip = clip, extrapolate=extrapolate)

        x = StatArray.StatArray(xc, name=self.x.name, units=self.x.units)
        y = StatArray.StatArray(yc, name=self.y.name, units=self.y.units)

        return x, y, vals, kwargs


    def interpMinimumCurvature(self, dx, dy, values, mask=False, clip=True, i=None, **kwargs):


        iterations = kwargs.pop('iterations', 2000)
        tension = kwargs.pop('tension', 0.25)
        accuracy = kwargs.pop('accuracy', 0.01)

        # Get the discretization
        assert dx > 0.0, ValueError("dx must be positive!")
        assert dy > 0.0, ValueError("dy must be positive!")

        x, y, vals = interpolation.minimumCurvature(self.x, self.y, values, self.bounds, dx, dy, mask=mask, clip=clip, iterations=iterations, tension=tension, accuracy=accuracy)
        x = StatArray.StatArray(x, name=self.x.name, units=self.x.units)
        y = StatArray.StatArray(y, name=self.y.name, units=self.y.units)
        return x, y, vals, kwargs


    def mapPlot(self, dx, dy, i=None, **kwargs):
        """ Create a map of a parameter """

        cTmp = kwargs.pop('c', self.z)

        mask = kwargs.pop('mask', False)

        clip = kwargs.pop('clip', True)

        method = kwargs.pop('method', "ct").lower()


        if method == 'ct':
            x, y, vals, kwargs = self.interpCloughTocher(dx, dy, values=cTmp, mask=mask, clip=clip, i=i, **kwargs)
        elif method == 'mc':
            x, y, vals, kwargs = self.interpMinimumCurvature(dx, dy, values=cTmp, mask=mask, clip=clip, i=i, **kwargs)
        else:
            assert False, ValueError("method must be either 'ct' or 'mc' ")

        return cP.pcolor(vals, x.edges(), y.edges(), **kwargs)


    def nearest(self, x, k=1, eps=0, p=2, radius=np.inf):
        """Obtain the k nearest neighbours

        See Also
        --------
        See scipy.spatial.cKDTree.query for argument descriptions and return types

        """

        assert (not self.kdtree is None), TypeError('kdtree has not been set, use self.setKdTree()')
        return self.kdtree.query(x, k, eps, p, distance_upper_bound=radius)


    def plot(self, values, xAxis='index', **kwargs):
        """Line plot of values against a co-ordinate.

        Parameters
        ----------
        values : array_like
            Values to plot against a co-ordinate
        xAxis : str
            If xAxis is 'index', returns numpy.arange(self.nPoints)
            If xAxis is 'x', returns self.x
            If xAxis is 'y', returns self.y
            If xAxis is 'z', returns self.z
            If xAxis is 'r2d', returns cumulative distance along the line in 2D using x and y.
            If xAxis is 'r3d', returns cumulative distance along the line in 3D using x, y, and z.

        Returns
        -------
        ax : matplotlib.axes
            Plot axes handle

        See Also
        --------
        geobipy.customPlots.plot : For additional keyword arguments

        """
        x = self.getXAxis(xAxis)
        ax = cP.plot(x, values, **kwargs)
        return ax


    def _readNpoints(self, filename):
        """Read the number of points in a data file

        Parameters
        ----------
        filename : list of str.
            Path to the data files.

        Returns
        -------
        nPoints : int
            Number of observations.

        """
        if isinstance(filename, str):
            filename = (filename)
        nSystems = len(filename)
        nPoints = np.empty(nSystems, dtype=np.int64)
        for i in range(nSystems):
            nPoints[i] = fIO.getNlines(filename[i], 1)
        for i in range(1, nSystems):
            assert nPoints[i] == nPoints[0], Exception('Number of data points {} in file {} does not match {} in file {}'.format(nPoints[i], filename[i], nPoints[0], filename[0]))
        return nPoints[0]


    def __readColumnIndices(self, filename):
        """Read in the header information for an FdemData file.

        Parameters
        ----------
        fileName : str
            Path to the data file.

        Returns
        -------
        nPoints : int
            Number of measurements.
        columnIndex : ints
            The column indices for line, id, x, y, z, elevation, data, uncertainties.
        hasErrors : bool
            Whether the file contains uncertainties or not.

        """

        indices = []

        # Get the column headers of the data file
        channels = [channel.lower() for channel in fIO.getHeaderNames(filename)]
        nChannels = len(channels)

        x_names = ('e', 'x','easting')
        y_names = ('n', 'y', 'northing')
        z_names = ('alt', 'laser', 'bheight', 'height')
        e_names = ('z','dtm','dem_elev','dem_np','topo', 'elev', 'elevation')

        # Check for each aspect of the data file and the number of columns
        nCoordinates = 0
        for channel in channels:
            if (channel in x_names):
                nCoordinates += 1
            elif (channel in y_names):
                nCoordinates += 1
            elif (channel in z_names):
                nCoordinates += 1
            elif(channel in e_names):
                nCoordinates += 1

        assert nCoordinates >= 3, Exception("File must contain columns for easting, northing, height. May also have an elevation column \n {}".format(self.fileInformation()))

        # To grab the EM data, skip the following header names. (More can be added to this)
        # Initialize a column identifier for x y z
        index = np.zeros(nCoordinates, dtype=np.int32)
        for j, channel in enumerate(channels):
            if (channel in x_names):
                index[0] = j
            elif (channel in y_names):
                index[1] = j
            elif (channel in z_names):
                index[2] = j
            elif(channel in e_names):
                index[3] = j

        return index


    def read(self, fileName):
        """Reads x y z co-ordinates from an ascii csv file.

        Parameters
        ----------
        fileName : str
            Path to the file to read from.

        """
        indices = self.__readColumnIndices(fileName)
        values = fIO.read_columns(fileName, nHeaders=1, indices=indices)

        tmp = fIO.getHeaderNames(fileName, indices)

        self.__init__(*np.hsplit(values, np.size(indices)))
        self.x.name = tmp[0]
        self.y.name = tmp[1]
        self.z.name = tmp[2]
        if np.size(indices) > 3:
            self.elevation.name = tmp[3]

        return self


    def fileInformation(self):
        """Description of PointCloud3D file.

        Returns
        -------
        out : str
            File description.

        """

        tmp = ("The file is structured using columns with the first line containing a header line.\n"
                "When reading, the columnIndices are used to read the x, y, z co-ordinates.\n"
                "The corresponding entries in the header are used to give the co-ordinates their label. ")

        return tmp


    def scatter2D(self, **kwargs):
        """Create a 2D scatter plot using the x, y coordinates of the point cloud.

        Can take any other matplotlib arguments and keyword arguments e.g. markersize etc.

        Parameters
        ----------
        c : 1D array_like or StatArray, optional
            Colour values of the points, default is the height of the points
        i : sequence of ints, optional
            Plot a subset of x, y, c, using the indices in i.

        See Also
        --------
        geobipy.customPlots.Scatter2D : For additional keyword arguments you may use.

        """

        kwargs['linewidth'] = kwargs.pop('linewidth', 0.1)
        kwargs['c'] = kwargs.pop('c', self.z)
        ax, sc, cbar = cP.scatter2D(self.x, y=self.y, **kwargs)
        return ax, sc, cbar


    def setKdTree(self, nDims=3):
        """Creates a k-d tree of the point co-ordinates

        Parameters
        ----------
        nDims : int
            Either 2 or 3 to exclude or include the vertical co-ordinate

        """
        self.kdtree = None
        if (nDims == 2):
            tmp = np.column_stack((self.x, self.y))
        elif (nDims == 3):
            tmp = np.column_stack((self.x, self.y, self.z))
        self.kdtree = cKDTree(tmp)


    def summary(self, out=False):
        """ Display a summary of the 3D Point Cloud """
        msg = ("3D Point Cloud: \n"
              "Number of Points: : {} \n {} {} {} {}").format(self._nPoints, self.x.summary(True), self.y.summary(True), self.z.summary(True), self.elevation.summary(True))
        return msg if out else print(msg)


#    def interpRBF(self, values, nDims, dx = None, function = None, epsilon = None, smooth = None, norm = None, **kwargs):
#        """ Interpolate values to a grid using radial basis functions """
#        # Get the bounding box
#        self.getBounds()
#        print(self.bounds)
#        # Get the discretization
#        if (dx is None):
#            tmp = np.min((self.bounds[1]-self.bounds[0], self.bounds[3]-self.bounds[2]))
#            dx = 0.01 * tmp
#        assert dx > 0.0, ValueError("Interpolation cell size must be positive!")
#
#        if (nDims == 2):
#            z = np.ones(self.N)
#        elif (nDims == 3):
#            z = self.z
#
#        x,y,vals = interpolation.RBF(dx, self.bounds, self.x, self.y, z, values)
#        return x, y, vals




    def vtkStructure(self):
        """Generates a vtk mesh structure that can be used in a vtk file.

        Returns
        -------
        out : pyvtk.VtkData
            Vtk data structure

       """

        nodes = np.vstack([self.x, self.y, self.z]).T

        vtk = VtkData(UnstructuredGrid(nodes, vertex=np.arange(self._nPoints)))
        vtk.point_data.append(Scalars(self.z, self.z.getNameUnits()))
        return vtk


    def toVTK(self, fileName, pointData=None, format='binary'):
        """Save the PointCloud3D to a VTK file.

        Parameters
        ----------
        fileName : str
            Filename to save to.
        pointData : geobipy.StatArray or list of geobipy.StatArray, optional
            Data at each point in the point cloud. Each entry is saved as a separate
            vtk attribute.
        format : str, optional
            "ascii" or "binary" format. Ascii is readable, binary is not but results in smaller files.

        Raises
        ------
        TypeError
            If pointData is not a geobipy.StatArray or list of them.
        ValueError
            If any pointData entry does not have size equal to the number of points.
        ValueError
            If any StatArray does not have a name or units. This is needed for the vtk attribute.

        """

        vtk = self.vtkStructure()

        if not pointData is None:
            assert isinstance(pointData, (StatArray.StatArray, list)), TypeError("pointData must a geobipy.StatArray or a list of them.")
            if isinstance(pointData, list):
                for p in pointData:
                    assert isinstance(p, StatArray.StatArray), TypeError("pointData entries must be a geobipy.StatArray")
                    assert p.size == self.nPoints, ValueError("pointData entries must have size {}".format(self.nPoints))
                    assert p.hasLabels(), ValueError("StatArray needs a name")
                    vtk.point_data.append(Scalars(p, p.getNameUnits()))
            else:
                assert pointData.size == self.nPoints, ValueError("pointData entries must have sizd {}".format(self.nPoints))
                assert pointData.hasLabels(), ValueError("StatArray needs a name")
                vtk.point_data.append(Scalars(pointData, pointData.getNameUnits()))

        vtk.tofile(fileName, format=format)


    def Bcast(self, world, root=0):
        """Broadcast a PointCloud3D using MPI

        Parameters
        ----------
        world : mpi4py.MPI.COMM_WORLD
            MPI communicator
        root : int, optional
            The MPI rank to broadcast from. Default is 0.

        Returns
        -------
        out : geobipy.PointCloud3D
            PointCloud3D broadcast to each rank

        """

        x = self.x.Bcast(world, root=root)
        y = self.y.Bcast(world, root=root)
        z = self.z.Bcast(world, root=root)
        e = self.elevation.Bcast(world, root=root)
        # N = MPI.Bcast(self._nPoints, world, root=root)
        return PointCloud3D(x=x, y=y, z=z, elevation=e)


    def Scatterv(self, starts, chunks, world, root=0):
        """ScatterV a PointCloud3D using MPI

        Parameters
        ----------
        myStart : sequence of ints
            Indices into self that define the starting locations of the chunks to be sent to each rank.
        myChunk : sequence of ints
            The size of each chunk that each rank will receive.
        world : mpi4py.MPI.Comm
            The MPI communicator over which to Scatterv.
        root : int, optional
            The MPI rank to broadcast from. Default is 0.

        Returns
        -------
        out : geobipy.PointCloud3D
            The PointCloud3D distributed amongst ranks.

        """

        # N = chunks[world.rank]
        x = self.x.Scatterv(starts, chunks, world, root=root)
        y = self.y.Scatterv(starts, chunks, world, root=root)
        z = self.z.Scatterv(starts, chunks, world, root=root)
        e = self.elevation.Scatterv(starts, chunks, world, root=root)
        return PointCloud3D(x=x, y=y, z=z, elevation=e)
