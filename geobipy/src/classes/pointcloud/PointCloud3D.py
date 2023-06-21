from copy import deepcopy
from numpy import arange, argpartition, argsort, argwhere, asarray, column_stack, cumsum, diff, empty
from numpy import float64, hstack, inf, int32, int64, isnan, mean, meshgrid, nan, nanmin, nanmax
from numpy import ravel_multi_index, size, sqrt, squeeze, unique, vstack, zeros
from ...classes.core.myObject import myObject
from pandas import DataFrame, read_csv
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

try:
    from pygmt import surface
    gmt = True
except:
    gmt = False


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
    single = Point

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

        # raise DeprecationWarning()

    def __deepcopy__(self, memo={}):
        result = type(self).__new__(type(self))
        result._nPoints = 0
        result.x = deepcopy(self.x)
        result.y = deepcopy(self.y)
        result.z = deepcopy(self.z)
        result.elevation = deepcopy(self.elevation)
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
        if not isinstance(i, slice):
            i = unique(i)

        if size(i) == 1:
            cls = self.single
        else:
            cls = type(self)

        out = cls()
        if self.x.size > 0: out.x = self.x[i]
        if self.y.size > 0: out.y = self.y[i]
        if self.z.size > 0: out.z = self.z[i]
        if self.elevation.size > 0: out.elevation = self.elevation[i]
        return out

    def _as_dict(self):
        return {self.x.name.replace(' ', '_'): self.x,
                self.y.name.replace(' ', '_'): self.y,
                self.z.name.replace(' ', '_'): self.z,
                self.elevation.name.replace(' ', '_'): self.elevation}, \
                [self.x.name.replace(' ', '_'), self.y.name.replace(' ', '_'), self.z.name.replace(' ', '_'), self.elevation.name.replace(' ', '_')]

    @property
    def x(self):
        if size(self._x) == 0:
            self._x = StatArray.StatArray(self.nPoints, "Easting", "m", dtype=float64)
        return self._x

    @x.setter
    def x(self, values):
        if (values is None):
            values = self.nPoints
        else:
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("x must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._x = deepcopy(values)
                return

        self._x = StatArray.StatArray(values, "Easting", "m", dtype=float64)

    @property
    def y(self):
        if size(self._y) == 0:
            self._y = StatArray.StatArray(self.nPoints, "Northing", "m", dtype=float64)
        return self._y

    @y.setter
    def y(self, values):
        if (values is None):
            values = self.nPoints
        else:
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("y must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._y = deepcopy(values)
                return

        self._y = StatArray.StatArray(values, "Northing", "m", dtype=float64)

    @property
    def z(self):
        if size(self._z) == 0:
            self._z = StatArray.StatArray(self.nPoints, "Height", "m", dtype=float64)
        return self._z

    @z.setter
    def z(self, values):
        if (values is None):
            values = self.nPoints
        else:
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("z must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._z = deepcopy(values)
                return

        self._z = StatArray.StatArray(values, "Height", "m", dtype=float64)

    @property
    def elevation(self):
        if size(self._elevation) == 0:
            self._elevation = StatArray.StatArray(self.nPoints, "Elevation", "m", dtype=float64)
        return self._elevation

    @elevation.setter
    def elevation(self, values):
        if (values is None):
            values = self.nPoints
        else:
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("elevation must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._elevation = deepcopy(values)
                return

        self._elevation = StatArray.StatArray(values, "Elevation", "m", dtype=float64)

    @property
    def nPoints(self):
        """Get the number of points"""
        return self._nPoints

    @nPoints.setter
    def nPoints(self, value):
        assert isinstance(value, (int, int32, int64)), TypeError("nPoints must be an integer")
        self._nPoints = value

    @property
    def n_posteriors(self):
        return self.x.nPosteriors + self.y.nPosteriors + self.z.nPosteriors

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
        self.x = hstack([self.x, other.x])
        self.y = hstack([self.y, other.y])
        self.z = hstack([self.z, other.z])
        self.elevation = hstack([self.elevation, other.elevation])


    @property
    def bounds(self):
        """Gets the bounding box of the data set """
        return asarray(self.x.bounds, self.y.bounds)

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
            x_grid = self.x.centred_grid_nodes(dx)
        if y_grid is None:
            y_grid = self.y.centred_grid_nodes(dy)

        ax = RectilinearMesh1D(edges=x_grid)
        ix = ax.cellIndex(self.x)
        ay = RectilinearMesh1D(edges=y_grid)
        iy = ay.cellIndex(self.y)

        return ravel_multi_index([ix, iy], (ax.nCells.item(), ay.nCells.item()))


    def block_mean(self, dx, dy, values=None):

        i_cell = self.block_indices(dx, dy)

        isort = argsort(i_cell)
        n_new = unique(i_cell).size
        cuts = hstack([0, squeeze(argwhere(diff(i_cell[isort]) > 0)), i_cell.size])

        if values is None:
            values = self.z

        x_new = empty(n_new)
        y_new = empty(n_new)
        z_new = empty(n_new)
        e_new = empty(n_new)

        for i in range(cuts.size-1):
            i_cut = isort[cuts[i]:cuts[i+1]]
            x_new[i] = mean(self.x[i_cut])
            y_new[i] = mean(self.y[i_cut])
            z_new[i] = mean(self.z[i_cut])
            e_new[i] = mean(self.elevation[i_cut])

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

        isort = argsort(i_cell)
        n_new = unique(i_cell).size

        cuts = squeeze(argwhere(diff(i_cell[isort]) > 0))
        if cuts[0] != 0:
            cuts = hstack([0, cuts])
        if cuts[-1] != i_cell.size:
            cuts = hstack([cuts, i_cell.size])

        if values is None:
            values = self.z

        i_new = empty(cuts.size-1, dtype=int64)
        for i in range(cuts.size-1):
            i_cut = isort[cuts[i]:cuts[i+1]]
            tmp = values[i_cut]
            i_new[i] = i_cut[argpartition(tmp, tmp.size // 2)[tmp.size // 2]]

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

        x_grid = self.x.centred_grid_nodes(dx)
        y_grid = self.y.centred_grid_nodes(dy)
        return RectilinearMesh2D(x_edges=x_grid, y_edges=y_grid)


    def interpolate(self, dx=None, dy=None, mesh=None, values=None , method='mc', mask = False, clip = True, i=None, block=False, **kwargs):
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

        values[values == inf] = nan
        mn = nanmin(values)
        mx = nanmax(values)

        values -= mn
        if (mx - mn) != 0.0:
            values = values / (mx - mn)

        kdtree = None
        if (not i is None):
            xTmp = self.x[i]
            yTmp = self.y[i]
            vTmp = values[i]
            XY = column_stack((xTmp, yTmp))
            if (mask or extrapolate):
                kdtree = cKDTree(XY)
        else:
            XY = column_stack((self.x, self.y))
            vTmp = values
            if (mask or extrapolate):
                self.setKdTree(nDims = 2)
                kdtree = self.kdtree

        # Create the CT function for interpolation
        f = CloughTocher2DInterpolator(XY, vTmp)

        query = hstack([mesh.x_centres.flatten(), mesh.y_centres.flatten()])

        # Interpolate to the grid
        vals = f(query).reshape(*mesh.shape)

        # Reshape to a 2D array
        vals = StatArray.StatArray(vals, name=cf.getName(values), units = cf.getUnits(values))

        if mask or extrapolate:
            if (kdtree is None):
                kdt = cKDTree(XY)
            else:
                kdt = kdtree

        # Use distance masking
        if mask:
            g = meshgrid(mesh.x.centres, mesh.y.centres)
            xi = _ndim_coords_from_arrays(tuple(g), ndim=XY.shape[1])
            dists, indexes = kdt.query(xi)
            vals[dists > mask] = nan

        # Truncate values to the observed values
        if (clip):
            minV = nanmin(values)
            maxV = nanmax(values)
            mask = ~isnan(vals)
            mask[mask] &= vals[mask] < minV
            vals[mask] = minV
            mask = ~isnan(vals)
            mask[mask] &= vals[mask] > maxV
            vals[mask] = maxV

        if (not extrapolate is None):
            assert isinstance(extrapolate,str), 'extrapolate must be a string. Choose [Nearest]'
            extrapolate = extrapolate.lower()
            if (extrapolate == 'nearest'):
                # Get the indices of the nans
                iNan = argwhere(isnan(vals))
                # Create Query locations from the nans
                xi =  zeros([iNan.shape[0],2])
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

        try:
            from pygmt import surface
        except Exception as e:
            print(repr(e))
            raise Exception(("\npygmt not installed correctly.  method='mc' can only be used when pygmt is present.\n"
                             "To install pygmt, you need to use conda environments. Installing instructions are here\n"
                             "https://www.pygmt.org/latest/install.html \n"
                             "After creating a new conda environment do\n"
                             "'pip install -c conda-forge numpy pandas xarray netcdf4 packaging gmt pygmt'\n"
                             "Then install geobipy and its dependencies to that environment."))

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

        values[values == inf] = nan
        mn = nanmin(values)
        mx = nanmax(values)

        values -= mn
        if (mx - mn) != 0.0:
            values = values / (mx - mn)

        dx = mesh.x.widths[0]
        dy = mesh.x.widths[1]

        if clip:
            clip_min = kwargs.pop('clip_min', nanmin(values))
            clip_max = kwargs.pop('clip_max', nanmax(values))
            xr = surface(x=x, y=y, z=values, spacing=(dx, dy), region=mesh.centres_bounds, N=iterations, T=tension, C=accuracy, Ll=[clip_min], Lu=[clip_max])
        else:
            xr = surface(x=x, y=y, z=values, spacing=(dx, dy), region=mesh.centres_bounds, N=iterations, T=tension, C=accuracy)

        vals = xr.values

        if (mx - mn) != 0.0:
            vals = vals * (mx - mn)
        vals += mn

        # Use distance masking
        if mask:
            kdt = cKDTree(column_stack((x, y)))
            xi = _ndim_coords_from_arrays(tuple(meshgrid(mesh.x.centres, mesh.y.centres)), ndim=2)
            dists, indexes = kdt.query(xi)
            vals[dists > mask] = nan

        vals = StatArray.StatArray(vals, name=cf.getName(values), units = cf.getUnits(values))

        out = Model(mesh=mesh, values=vals.T)

        return out, kwargs

    def map(self, dx, dy, i=None, **kwargs):
        """ Create a map of a parameter """

        values = kwargs.pop('values', self.z)
        mask = kwargs.pop('mask', None)
        clip = kwargs.pop('clip', True)
        method = kwargs.pop('method', "mc").lower()

        values, _ = self.interpolate(dx, dy, values=values, mask=mask, method=method, i=i, clip=clip, **kwargs)

        return values.pcolor(**kwargs)

    def nearest(self, x, k=1, eps=0, p=2, radius=inf):
        """Obtain the k nearest neighbours

        See Also
        --------
        See scipy.spatial.cKDTree.query for argument descriptions and return types

        """

        assert (not self.kdtree is None), TypeError('kdtree has not been set, use self.setKdTree()')
        return self.kdtree.query(x, k, eps, p, distance_upper_bound=radius)

    def plot(self, values, x='index', **kwargs):
        """Line plot of values against a co-ordinate.

        Parameters
        ----------
        values : array_like
            Values to plot against a co-ordinate
        axis : str
            If axis is 'index', returns numpy.arange(self.nPoints)
            If axis is 'x', returns self.x
            If axis is 'y', returns self.y
            If axis is 'z', returns self.z
            If axis is 'r2d', returns cumulative distance along the line in 2D using x and y.
            If axis is 'r3d', returns cumulative distance along the line in 3D using x, y, and z.

        Returns
        -------
        ax : matplotlib.axes
            Plot axes handle

        See Also
        --------
        geobipy.plotting.plot : For additional keyword arguments

        """
        x = self.axis(x)
        ax = cP.plot(x, values, **kwargs)
        return ax

    # @property
    def pyvista_mesh(self):
        import pyvista as pv

        out = pv.PolyData(vstack([self.x, self.y, self.z]).T)
        out['height'] = self.z
        out["elevation"] = self.elevation

        return out

    # def pyvista_plotter(self, plotter=None, **kwargs):

    #     import pyvista as pv

    #     if plotter is None:
    #         plotter = pv.Plotter()
    #     plotter.add_mesh(self.pyvista_mesh())

    #     labels = dict(xlabel=self.x.label, ylabel=self.y.label, zlabel=self.z.label)
    #     plotter.add_axes(**labels)

    #     return plotter

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
        nPoints = empty(nSystems, dtype=int64)
        for i in range(nSystems):
            nPoints[i] = fIO.getNlines(filename[i], 1)
        for i in range(1, nSystems):
            assert nPoints[i] == nPoints[0], Exception('Number of data points {} in file {} does not match {} in file {}'.format(nPoints[i], filename[i], nPoints[0], filename[0]))
        return nPoints[0]

    @staticmethod
    def _csv_n_points(filename):
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
        # if isinstance(filename, str):
        #     filename = [filename]

        # nPoints = asarray([fIO.getNlines(df, 1) for df in filename])

        # if nPoints.size > 1:
        #     assert all(diff(nPoints) == 0), Exception('Number of data points must match in all data files')
        # return nPoints[0]
        nPoints = fIO.getNlines(filename, 1)

        # if nPoints.size > 1:
            # assert all(diff(nPoints) == 0), Exception('Number of data points must match in all data files')
        return nPoints

    @staticmethod
    def _csv_channels(filename):
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
        channels = fIO.get_column_name(filename)
        nChannels = len(channels)

        x_names = ('e', 'x','easting')
        y_names = ('n', 'y', 'northing')
        z_names = ('alt', 'altitude', 'laser', 'bheight', 'height')
        e_names = ('dtm','dem_elev','dem_np','topo', 'elev', 'elevation')

        n = 0
        labels = [None]*3

        for channel in channels:
            cTmp = channel.lower()
            if (cTmp in x_names):
                n += 1
                labels[0] = channel
            elif (cTmp in y_names):
                n += 1
                labels[1] = channel
            elif (cTmp in z_names):
                n += 1
                labels[2] = channel
            elif(cTmp in e_names):
                labels.append(channel)

        assert not any([x is None for x in labels[:3]]), Exception("File must contain columns for easting, northing, height. May also have an elevation column \n {}".format(PointCloud3D.fileInformation()))
        assert n == 3 and len(labels) <= 4, Exception("File must contain columns for easting, northing, height. May also have an elevation column \n {}".format(PointCloud3D.fileInformation()))
        return PointCloud3D._csv_n_points(filename), labels

    @classmethod
    def read_csv(cls, filename, **kwargs):
        """Reads x y z co-ordinates from an ascii csv file.

        Parameters
        ----------
        filename : str
            Path to the file to read from.

        """
        nPoints, channels = PointCloud3D._csv_channels(filename)

        try:
            df = read_csv(filename, index_col=False, usecols=channels, skipinitialspace = True)
        except:
            df = read_csv(filename, index_col=False, usecols=channels, delim_whitespace=True, skipinitialspace = True)
        df = df.replace('NaN',nan)

        self = cls(**kwargs)
        self.x = df[channels[0]].values
        self.y = df[channels[1]].values
        self.z = df[channels[2]].values
        if size(channels) > 3:
            self.elevation = df[channels[3]].values

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
            tmp = column_stack((self.x, self.y))
        elif (nDims == 3):
            tmp = column_stack((self.x, self.y, self.z))
        self.kdtree = cKDTree(tmp)


    @property
    def summary(self):
        """Summary of self """
        msg =  "{}\n".format(type(self).__name__)
        msg += "x:\n{}\n".format("|   "+(self.x.summary.replace("\n", "\n|   "))[:-4])
        msg += "y:\n{}\n".format("|   "+(self.y.summary.replace("\n", "\n|   "))[:-4])
        msg += "z:\n{}\n".format("|   "+(self.z.summary.replace("\n", "\n|   "))[:-4])
        msg += "elevation:\n{}\n".format("|   "+(self.elevation.summary.replace("\n", "\n|   "))[:-4])

        return msg


#    def interpRBF(self, values, nDims, dx = None, function = None, epsilon = None, smooth = None, norm = None, **kwargs):
#        """ Interpolate values to a grid using radial basis functions """
#        # Get the bounding box
#        self.getBounds()
#        # Get the discretization
#        if (dx is None):
#            tmp = min((self.bounds[1]-self.bounds[0], self.bounds[3]-self.bounds[2]))
#            dx = 0.01 * tmp
#        assert dx > 0.0, ValueError("Interpolation cell size must be positive!")
#
#        if (nDims == 2):
#            z = ones(self.N)
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

        nodes = vstack([self.x, self.y, self.z]).T

        vtk = VtkData(UnstructuredGrid(nodes, vertex=arange(self._nPoints)))
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
                    assert size(p) == self.nPoints, ValueError("pointData entries must have size {}".format(self.nPoints))
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
        self.elevation.createHdf(grp, 'elevation', withPosterior=withPosterior, fillvalue=fillvalue)

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

    @classmethod
    def fromHdf(cls, grp, **kwargs):
        """ Reads the object from a HDF group """

        if kwargs.get('index') is not None:
            return cls.single.fromHdf(grp, **kwargs)

        x = StatArray.StatArray.fromHdf(grp['x'])
        y = StatArray.StatArray.fromHdf(grp['y'])
        z = StatArray.StatArray.fromHdf(grp['z'])
        elevation = StatArray.StatArray.fromHdf(grp['elevation'])

        return cls(x=x, y=y, z=z, elevation=elevation, **kwargs)

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
        return type(self)(x=x, y=y, z=z, elevation=e)

    def Isend(self, dest, world):
        self.x.Isend(dest, world)
        self.y.Isend(dest, world)
        self.z.Isend(dest, world)
        self.elevation.Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world, **kwargs):
        x = StatArray.StatArray.Irecv(source, world)
        y = StatArray.StatArray.Irecv(source, world)
        z = StatArray.StatArray.Irecv(source, world)
        e = StatArray.StatArray.Irecv(source, world)
        return cls(x=x, y=y, z=z, elevation=e, **kwargs)

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
        x = self.x.Scatterv(starts, chunks, world, root=root)
        y = self.y.Scatterv(starts, chunks, world, root=root)
        z = self.z.Scatterv(starts, chunks, world, root=root)
        e = self.elevation.Scatterv(starts, chunks, world, root=root)
        return type(self)(x=x, y=y, z=z, elevation=e)
