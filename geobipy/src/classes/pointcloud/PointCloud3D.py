from copy import deepcopy
from ...classes.core.myObject import myObject
import numpy as np
from ...classes.core import StatArray
from ...base import fileIO as fIO
# from ...base import interpolation
from ...base import utilities as cf
from ...base import plotting as cP
from .Point import Point
from ...base import MPI
from ..mesh.RectilinearMesh1D import RectilinearMesh1D
from ..mesh.RectilinearMesh2D import RectilinearMesh2D
from..model.Model import Model
from scipy.spatial import cKDTree
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

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

        # # Number of points in the cloud
        self._nPoints = 0

            # StatArray of the x co-ordinates
        self.x = x

        # StatArray of the y co-ordinates
        self.y = y

        # StatArray of the z co-ordinates
        self.z = z

        # StatArray of elevation
        self.elevation = elevation

    def __deepcopy__(self, memo={}):
        result = type(self).__new__(type(self))
        result.x = self.x
        result.y = self.y
        result.z = self.z
        return result

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
                self._x = deepcopy(values)
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
                self._y = deepcopy(values)
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
                self._z = deepcopy(values)
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
                self._elevation = deepcopy(values)
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

    def _reconcile_channels(self, channels):

        for i, channel in enumerate(channels):
            channel = channel.lower()
            if (channel in ['n', 'x','northing']):
                channels[i] = 'x'
            elif (channel in ['e', 'y', 'easting']):
                channels[i] = 'y'
            elif (channel in ['alt', 'laser', 'bheight', 'height']):
                channels[i] = 'height'
            elif(channel in ['z','dtm','dem_elev', 'dem', 'dem_np','topo', 'elev', 'elevation']):
                channels[i] = 'elevation'

        return channels


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

    def block_indices(self, dx=None, dy=None, x_grid=None, y_grid=None):
        """Returns the indices of the points lying in blocks across the domain..

        Performed before a block median filter by extracting the point location within blocks across the domain.
        Idea taken from pygmt, however I extracted the indices and this was quite a bit faster.

        Parameters
        ----------
        dx : float
            Grid spacing in x.
        dy : float
            Grid spacing in y.

        Returns
        -------
        ints : Index into self whose points are the median location within blocks across the domain.
        """
        if x_grid is None:
            x_grid = self.centred_grid_nodes(self.bounds[:2], dx)
        if y_grid is None:
            y_grid = self.centred_grid_nodes(self.bounds[2:], dy)

        ax = RectilinearMesh1D(edges=x_grid)
        ix = ax.cellIndex(self.x)
        ay = RectilinearMesh1D(edges=y_grid)
        iy = ay.cellIndex(self.y)

        return np.ravel_multi_index([ix, iy], (ax.nCells.value, ay.nCells.value))


    def block_mean(self, dx, dy, values=None):

        i_cell = self.block_indices(dx, dy)

        isort = np.argsort(i_cell)
        n_new = np.unique(i_cell).size
        cuts = np.hstack([0, np.squeeze(np.argwhere(np.diff(i_cell[isort]) > 0)), i_cell.size])

        if values is None:
            values = self.z

        x_new = np.empty(n_new)
        y_new = np.empty(n_new)
        z_new = np.empty(n_new)
        e_new = np.empty(n_new)

        for i in range(cuts.size-1):
            i_cut = isort[cuts[i]:cuts[i+1]]
            x_new[i] = np.mean(self.x[i_cut])
            y_new[i] = np.mean(self.y[i_cut])
            z_new[i] = np.mean(self.z[i_cut])
            e_new[i] = np.mean(self.elevation[i_cut])

        return PointCloud3D(x=x_new, y=y_new, z=z_new, elevation=e_new)


    def block_median_indices(self, dx=None, dy=None, x_grid=None, y_grid=None, values=None):
        """Index to the median point within juxtaposed blocks across the domain.

        Parameters
        ----------
        dx : float
            Increment in x.
        dy : float
            Increment in y.
        values : array_like, optional
            Used to compute the median in each block.
            Defaults to None.

        Returns
        -------
        ints : Index of the median point in each block.
        """
        i_cell = self.block_indices(dx, dy, x_grid, y_grid)

        isort = np.argsort(i_cell)
        n_new = np.unique(i_cell).size

        cuts = np.squeeze(np.argwhere(np.diff(i_cell[isort]) > 0))
        if cuts[0] != 0:
            cuts = np.hstack([0, cuts])
        if cuts[-1] != i_cell.size:
            cuts = np.hstack([cuts, i_cell.size])

        if values is None:
            values = self.z

        i_new = np.empty(cuts.size-1, dtype=np.int64)
        for i in range(cuts.size-1):
            i_cut = isort[cuts[i]:cuts[i+1]]
            tmp = values[i_cut]
            i_new[i] = i_cut[np.argpartition(tmp, tmp.size // 2)[tmp.size // 2]]

        return i_new


    def block_median(self, dx=None, dy=None, x_grid=None, y_grid=None, values=None):
        """Median point within juxtaposed blocks across the domain.

        Parameters
        ----------
        dx : float
            Increment in x.
        dy : float
            Increment in y.
        values : array_like, optional
            Used to compute the median in each block.
            Defaults to None.

        Returns
        -------
        geobipt.PointCloud3d : Contains one point in each block.
        """
        return self[self.block_median_indices(dx, dy, x_grid, y_grid, values)]

    def centred_mesh(self, dx, dy):

        x_grid = self.centred_grid_nodes(self.bounds[:2], dx)
        y_grid = self.centred_grid_nodes(self.bounds[2:], dy)
        return RectilinearMesh2D(xEdges=x_grid, yEdges=y_grid)


    def centred_grid_nodes(self, bounds, spacing):
        """Generates grid nodes centred over bounds

        Parameters
        ----------
        bounds : array_like
            bounds of the dimension
        spacing : float
            distance between nodes

        """
        # Get the discretization
        assert spacing > 0.0, ValueError("spacing must be positive!")
        sp = 0.5 * spacing
        return np.arange(bounds[0] - sp, bounds[1] + (2*sp), spacing)

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


    def interpolate(self, dx=None, dy=None, mesh=None, values=None , method='ct', mask = False, clip = True, i=None, block=False, **kwargs):
        """Interpolate values to a grid.

        The grid is automatically generated such that it is centred over the point cloud.

        Parameters
        ----------
        dx : float
            Grid spacing in x.
        dy : float
            Grid spacing in y.
        values : array_like, optional
            Values to interpolate.  Must have size self.nPoints.
            Defaults to None.
        method : str, optional
            * 'ct' uses Clough Tocher interpolation
            * 'mc' uses Minimum curvature and requires pygmt to be installed.
            Defaults to 'ct'.
        mask : float, optional
            Cells of distance mask away from points are NaN.
            Defaults to False.
        clip : bool, optional
            Clip any overshot grid values to the min/max of values.
            Defaults to True.
        i : ints, optional
            Use only the i locations during interpolation.
            Defaults to None.
        block : bool, optional
            Perform a block median filter before interpolation. Inherrently smooths the final grid, but alleviates aliasing.
            Defaults to False.

        Returns
        -------
        geobipy.Model : Interpolated values.
        """

        if values is None:
            values = self.z

        if mesh is None:
            mesh = self.centred_mesh(dx, dy)

        if i is None:
            i = self.block_median_indices(x_grid=mesh.x.edges, y_grid=mesh.y.edges, values=values) if block else None

        if method.lower() == 'ct':
            return self.interpCloughTocher(mesh, values, mask=mask, clip=clip, i=i, **kwargs)
        else:
            return self.interpMinimumCurvature(mesh, values, mask=mask, clip=clip, i=i, **kwargs)

    def interpCloughTocher(self, mesh, values, mask = False, clip = None, i=None, **kwargs):
        """ Interpolate values at the points to a grid """

        extrapolate = kwargs.pop('extrapolate', None)
        # Get the bounding box

        values[values == np.inf] = np.nan
        mn = np.nanmin(values)
        mx = np.nanmax(values)

        values -= mn
        if (mx - mn) != 0.0:
            values = values / (mx - mn)

        kdtree = None
        if (not i is None):
            xTmp = self.x[i]
            yTmp = self.y[i]
            vTmp = values[i]
            XY = np.column_stack((xTmp, yTmp))
            if (mask or extrapolate):
                kdtree = cKDTree(XY)
        else:
            XY = np.column_stack((self.x, self.y))
            vTmp = values
            if (mask or extrapolate):
                self.setKdTree(nDims = 2)
                kdtree = self.kdtree

        # Create the CT function for interpolation
        f = CloughTocher2DInterpolator(XY, vTmp)

        # Interpolate to the grid
        vals = f(mesh.centres).reshape(*mesh.shape)

        # Reshape to a 2D array
        vals = StatArray.StatArray(vals, name=cf.getName(values), units = cf.getUnits(values))

        if mask or extrapolate:
            if (kdtree is None):
                kdt = cKDTree(XY)
            else:
                kdt = kdtree

        # Use distance masking
        if mask:
            g = np.meshgrid(mesh.x.centres, mesh.y.centres)
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
                xi[:, 0] = x[iNan[:, 1]]
                xi[:, 1] = y[iNan[:, 0]]
                # Get the nearest neighbours
                dists, indexes = kdt.query(xi)
                # Assign the nearest value to the Nan locations
                vals[iNan[:, 0], iNan[:, 1]] = values[indexes]
            else:
                assert False, 'Extrapolation option not available. Choose [Nearest]'

        if (mx - mn) != 0.0:
            vals = vals * (mx - mn)
        vals += mn

        out = Model(mesh=mesh, values=vals)

        return out, kwargs

    def interpMinimumCurvature(self, mesh, values, mask=False, clip=True, i=None, operator=None, condition=None, **kwargs):

        from pygmt import surface

        assert isinstance(mesh, RectilinearMesh2D), TypeError("mesh must be RectilinearMesh2D")
        assert mesh.is_regular, ValueError("Minimum curvature must interpolate to a regular mesh")

        iterations = kwargs.pop('iterations', 2000)
        tension = kwargs.pop('tension', 0.25)
        accuracy = kwargs.pop('accuracy', 0.01)

        x = self.x
        y = self.y

        if not i is None:
            x = x[i]
            y = y[i]
            values = values[i]

        values[values == np.inf] = np.nan
        mn = np.nanmin(values)
        mx = np.nanmax(values)

        values -= mn
        if (mx - mn) != 0.0:
            values = values / (mx - mn)

        dx = mesh.x.widths[0]
        dy = mesh.x.widths[1]

        if clip:
            clip_min = kwargs.pop('clip_min', np.nanmin(values))
            clip_max = kwargs.pop('clip_max', np.nanmax(values))
            xr = surface(x=x, y=y, z=values, spacing=(dx, dy), region=mesh.centres_bounds, N=iterations, T=tension, C=accuracy, Ll=[clip_min], Lu=[clip_max])
        else:
            xr = surface(x=x, y=y, z=values, spacing=(dx, dy), region=mesh.centres_bounds, N=iterations, T=tension, C=accuracy)

        vals = xr.values

        if (mx - mn) != 0.0:
            vals = vals * (mx - mn)
        vals += mn

        # Use distance masking
        if mask:
            kdt = cKDTree(np.column_stack((x, y)))
            xi = _ndim_coords_from_arrays(tuple(np.meshgrid(mesh.x.centres, mesh.y.centres)), ndim=2)
            dists, indexes = kdt.query(xi)
            vals[dists > mask] = np.nan

        vals = StatArray.StatArray(vals, name=cf.getName(values), units = cf.getUnits(values))

        out = Model(mesh=mesh, values=vals)

        return out, kwargs


    def mapPlot(self, dx, dy, i=None, **kwargs):
        """ Create a map of a parameter """

        values = kwargs.pop('values', self.z)
        mask = kwargs.pop('mask', None)
        clip = kwargs.pop('clip', True)
        method = kwargs.pop('method', "ct").lower()

        values, _ = self.interpolate(dx, dy, values=values, mask=mask, method=method, i=i, clip=clip, **kwargs)

        return values.pcolor(**kwargs)


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
        geobipy.plotting.plot : For additional keyword arguments

        """
        x = self.getXAxis(xAxis)
        ax = cP.plot(x, values, **kwargs)
        return ax

    def pyvista_mesh(self):
        import pyvista as pv

        out = pv.PolyData(np.vstack([self.x, self.y, self.z]).T)
        out['height'] = self.z
        out["elevation"] = self.elevation

        return out

    def pyvista_plotter(self, plotter=None, **kwargs):

        import pyvista as pv

        if plotter is None:
            plotter = pv.Plotter()
        plotter.add_mesh(self.pyvista_mesh())

        labels = dict(xlabel=self.x.label, ylabel=self.y.label, zlabel=self.z.label)
        plotter.add_axes(**labels)

        return plotter



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

    def _csv_n_points(self, filename):
        """Read the number of points in a data file

        Parameters
        ----------
        filename : str or list of str.
            Path to the files.

        Returns
        -------
        nPoints : int
            Number of observations.

        """
        if isinstance(filename, str):
            filename = [filename]

        nPoints = np.asarray([fIO.getNlines(df, 1) for df in filename])

        if nPoints.size > 1:
            assert np.all(np.diff(nPoints) == 0), Exception('Number of data points must match in all data files')
        return nPoints[0]

    def _csv_column_indices(self, filename):
        """Get the column indices from a csv file.

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

        """

        indices = []

        # Get the column headers of the data file
        channels = fIO.getHeaderNames(filename)
        nChannels = len(channels)

        x_names = ('e', 'x','easting')
        y_names = ('n', 'y', 'northing')
        z_names = ('alt', 'laser', 'bheight', 'height')
        e_names = ('dtm','dem_elev','dem_np','topo', 'elev', 'elevation')

        # Check for each aspect of the data file and the number of columns
        nCoordinates = 0
        indices = []
        for channel in channels:
            channel = channel.lower()
            if (channel in x_names):
                nCoordinates += 1
            elif (channel in y_names):
                nCoordinates += 1
            elif (channel in z_names):
                nCoordinates += 1
            elif(channel in e_names):
                nCoordinates += 1

        assert nCoordinates <= 4, Exception("File must contain columns for easting, northing, height. May also have an elevation column \n {}".format(self.fileInformation()))

        # Initialize a column identifier for x y z
        index = np.zeros(nCoordinates, dtype=np.int32)
        for j, channel in enumerate(channels):
            channel = channel.lower()
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
        indices = self.csv_column_indices(fileName)
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
        geobipy.plotting.Scatter2D : For additional keyword arguments you may use.

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


    @property
    def summary(self):
        """ Display a summary of the 3D Point Cloud """
        msg = ("3D Point Cloud: \n"
              "Number of Points: : {} \n {} {} {} {}").format(self.nPoints, self.x.summary, self.y.summary, self.z.summary, self.elevation.summary)
        return msg


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


    def createHdf(self, parent, name, withPosterior=True, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)
        self.x.createHdf(grp, 'x', withPosterior=withPosterior, fillvalue=fillvalue)
        self.y.createHdf(grp, 'y', withPosterior=withPosterior, fillvalue=fillvalue)
        self.z.createHdf(grp, 'z', withPosterior=withPosterior, fillvalue=fillvalue)
        self.elevation.createHdf(grp, 'e', withPosterior=withPosterior, fillvalue=fillvalue)

        return grp


    def writeHdf(self, parent, name, withPosterior=True):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        grp = parent[name]
        self.x.writeHdf(grp, 'x',  withPosterior=withPosterior)
        self.y.writeHdf(grp, 'y',  withPosterior=withPosterior)
        self.z.writeHdf(grp, 'z',  withPosterior=withPosterior)
        self.elevation.writeHdf(grp, 'elevation',  withPosterior=withPosterior)


    def fromHdf(self, grp):
        """ Reads the object from a HDF group """

        x = StatArray.StatArray().fromHdf(grp['x'])
        y = StatArray.StatArray().fromHdf(grp['y'])
        z = StatArray.StatArray().fromHdf(grp['z'])
        elevation = StatArray.StatArray().fromHdf(grp['e'])

        PointCloud3D.__init__(self, x, y, z, elevation)

        return self



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
