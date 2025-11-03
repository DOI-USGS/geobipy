from copy import deepcopy
from numpy import arange, argpartition, argsort, argwhere, asarray, column_stack, cumsum, diff, empty
from numpy import float64, hstack, inf, int32, int64, isnan, maximum, mean, meshgrid, minimum, nan, nanmin, nanmax
from numpy import ravel_multi_index, size, sqrt, squeeze, unique, vstack, zeros
from numpy.linalg import norm
from ...classes.core.myObject import myObject
from pandas import read_csv
from matplotlib.figure import Figure
from matplotlib.pyplot import gcf
from ..core.DataArray import DataArray
from ..statistics.StatArray import StatArray
from ...base import fileIO as fIO
from ...base import utilities as cf
from ...base import plotting as cP
from ...base.interpolation import sibson
# from .Point import Point
from ..mesh.RectilinearMesh1D import RectilinearMesh1D
from ..mesh.RectilinearMesh2D import RectilinearMesh2D
from..model.Model import Model
from ..statistics.Histogram import Histogram
from ..statistics.Distribution import Distribution
from scipy.spatial import cKDTree
from scipy.interpolate import CloughTocher2DInterpolator
# from scipy.interpolate.interpnd import _ndim_coords_from_arrays

try:
    from pyvtk import VtkData, Scalars, PolyData, PointData, UnstructuredGrid
except:
    pass

try:
    from pygmt import surface
    gmt = True
except:
    gmt = False


class Point(myObject):
    """3D Point Cloud with x,y,z co-ordinates

    Point(N, x, y, z)

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
    __slots__ = ('_nPoints', '_x', '_y', '_z', '_elevation', '_kdtree')

    def __init__(self, x=None, y=None, z=None, elevation=None, **kwargs):
        """ Initialize the class """

        # # Number of points in the cloud
        self._nPoints = 0
        self._x = StatArray(self._nPoints, "Easting", "m")
        self._y = StatArray(self._nPoints, "Northing", "m")
        self._z = StatArray(self._nPoints, "Height", "m")
        self._elevation = StatArray(self._nPoints, "Elevation", "m")

        self.x = x
        self.y = y
        self.z = z
        self.elevation = elevation

        self._kdtree = None

    def __add__(self, other):
        """ Add two points together """
        out = deepcopy(self)
        out._x += other.x
        out._y += other.y
        out._z += other.z
        out._elevation += other.elevation
        return out

    def __deepcopy__(self, memo={}):
        out = type(self).__new__(type(self))

        out._nPoints = self.nPoints
        out._x = deepcopy(self.x, memo=memo)
        out._y = deepcopy(self.y, memo=memo)
        out._z = deepcopy(self.z, memo=memo)
        out._elevation = deepcopy(self.elevation, memo=memo)
        return out

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

        return type(self)(x=self.x[i],
                          y=self.y[i],
                          z=self.z[i],
                          elevation=self.elevation[i])

    def __sub__(self, other):
        """ Subtract two points """
        out = deepcopy(self)
        out._x -= other.x
        out._y -= other.y
        out._z -= other.z
        out._elevation -= other.elevation
        return out

    def _as_dict(self):
        return {self.x.name.replace(' ', '_'): self.x,
                self.y.name.replace(' ', '_'): self.y,
                self.z.name.replace(' ', '_'): self.z,
                self.elevation.name.replace(' ', '_'): self.elevation}, \
                [self.x.name.replace(' ', '_'),
                 self.y.name.replace(' ', '_'),
                 self.z.name.replace(' ', '_'),
                 self.elevation.name.replace(' ', '_')]

    @property
    def addressof(self):
        msg =  '{}: {}\n'.format(type(self).__name__, hex(id(self)))
        msg += "x:\n{}".format(("|   "+self.x.addressof.replace("\n", "\n|   "))[:-4])
        msg += "y:\n{}".format(("|   "+self.y.addressof.replace("\n", "\n|   "))[:-4])
        msg += "z:\n{}".format(("|   "+self.z.addressof.replace("\n", "\n|   "))[:-4])
        msg += "elevation:\n{}".format(("|   "+self.elevation.addressof.replace("\n", "\n|   "))[:-4])
        return msg

    @property
    def address(self):
        out = asarray([hex(id(self))])
        for x in [self.x, self.y, self.z, self.elevation]:
            out = hstack([out, squeeze(x.address)])
        return out

    @property
    def hasPosterior(self):
        return (self.x.hasPosterior + self.y.hasPosterior + self.z.hasPosterior) > 0

    @property
    def probability(self):
        """Evaluate the probability for the EM data point given the specified attached priors

        Parameters
        ----------
        rEerr : bool
            Include the relative error when evaluating the prior
        aEerr : bool
            Include the additive error when evaluating the prior
        height : bool
            Include the elevation when evaluating the prior
        calibration : bool
            Include the calibration parameters when evaluating the prior
        verbose : bool
            Return the components of the probability, i.e. the individually evaluated priors

        Returns
        -------
        out : float64
            The evaluation of the probability using all assigned priors

        Notes
        -----
        For each boolean, the associated prior must have been set.

        Raises
        ------
        TypeError
            If a prior has not been set on a requested parameter

        """
        probability = float64(0.0)

        for ax in (self.x, self.y, self.z):
            if ax.hasPrior:
                probability += ax.probability(log=True)

        return probability

    @property
    def kdtree(self):

        if self._kdtree is None:
            assert self.ndim > 1, ValueError("Points must have more than 1 dimension")
            self.set_kdtree(ndim=self.ndim)

        return self._kdtree

    @property
    def single(self):
        return Point

    @property
    def x(self):
        if self._x.size == 0:
            self._x = StatArray(self._nPoints, "Easting", "m")
        return self._x

    @x.setter
    def x(self, values):
        if values is not None: # Set a default array
            self.nPoints = size(values)
            if (self._x.size != self._nPoints):
                self._x = StatArray(values, "Easting", "m")
                return

            self._x[:] = values

    @property
    def y(self):
        if self._y.size == 0:
            self._y = StatArray(self._nPoints, "Northing", "m")
        return self._y

    @y.setter
    def y(self, values):
        if values is not None:
            self.nPoints = size(values)
            if (self._y.size != self._nPoints):
                self._y = StatArray(values, "Northing", "m")
                return
            self._y[:] = values

    @property
    def z(self):
        if self._z.size == 0:
            self._z = StatArray(self._nPoints, "Height", "m")
        return self._z

    @z.setter
    def z(self, values):
        if values is not None: # Set a default array
            self.nPoints = size(values)
            if (self._z.size != self._nPoints):
                self._z = StatArray(values, "Height", "m")
                return
            self._z[:] = values

    @property
    def elevation(self):
        if self._elevation.size == 0:
            self._elevation = StatArray(self._nPoints, "Elevation", "m")
        return self._elevation

    @elevation.setter
    def elevation(self, values):
        if values is not None: # Set a default array
            self.nPoints = size(values)
            if (self._elevation.size != self._nPoints):
                self._elevation = StatArray(values, "Elevation", "m", dtype=float64)
                return
            self._elevation[:] = values

    @property
    def ndim(self):
        return sum([size(t) == self._nPoints for t in (self.x, self.y, self.z)])

    @property
    def nPoints(self):
        """Get the number of points"""
        return self._nPoints

    @nPoints.setter
    def nPoints(self, value):
        if self._nPoints == 0 and value > 0:
            self._nPoints = int32(value)

    @property
    def n_posteriors(self):
        return self.x.n_posteriors + self.y.n_posteriors + self.z.n_posteriors

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
        return Point._csv_n_points(filename), labels


    def append(self, other):
        """Append pointclouds together

        Parameters
        ----------
        other : geobipy.PointCloud3D
            3D pointcloud

        """

        self._x = self.x.append(other.x)
        self._y = self.y.append(other.y)
        self._z = self.z.append(other.z)
        self._elevation = self.elevation.append(other.elevation)
        self._nPoints = self.nPoints + other.nPoints

        return self

    def axis(self, axis='x'):
        """Obtain the axis against which to plot values.

        Parameters
        ----------
        axis : str
            If axis is 'index', returns numpy.arange(self.nPoints)
            If axis is 'x', returns self.x
            If axis is 'y', returns self.y
            If axis is 'z', returns self.z
            If axis is 'r2d', returns cumulative distance along the line in 2D using x and y.
            If axis is 'r3d', returns cumulative distance along the line in 3D using x, y, and z.

        Returns
        -------
        out : array_like
            The requested axis.

        """
        assert axis in ['index', 'x', 'y', 'z', 'r2d', 'r3d'], Exception("axis must be either 'index', x', 'y', 'z', 'r2d', or 'r3d'")
        if axis == 'index':
            return DataArray(arange(self.x.size), name="Index")
        elif axis == 'x':
            return self.x
        elif axis == 'y':
            return self.y
        elif axis == 'z':
            return self.z
        elif axis == 'r2d':
            r = diff(self.x)**2.0 + diff(self.y)**2.0
            distance = DataArray(zeros(self.x.size), 'Distance', self.x.units)
            distance[1:] = cumsum(sqrt(r))
            return distance
        elif axis == 'r3d':
            r = diff(self.x)**2.0 + diff(self.y)**2.0 + diff(self.z)**2.0
            distance = DataArray(zeros(self.x.size), 'Distance', self.x.units)
            distance[1:] = cumsum(sqrt(r))
            return distance

    @property
    def bounds(self):
        """Gets the bounding box of the data set """
        return asarray([nanmin(self.x), nanmax(self.x), nanmin(self.y), nanmax(self.y)])

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

        return Point(x=x_new, y=y_new, z=z_new, elevation=e_new)


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

        x_grid = self.centred_grid_nodes(self.bounds[:2], dx)
        y_grid = self.centred_grid_nodes(self.bounds[2:], dy)
        return RectilinearMesh2D(x_edges=x_grid, y_edges=y_grid)


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
        return arange(bounds[0] - sp, bounds[1] + (2*sp), spacing)

    # def distance(self, other, **kwargs):
    #     """Get the Lp norm distance between two points. """
    #     return norm(asarray([self.x, self.y, self.z]) - asarray([other.x, other.y, other.z]), **kwargs)

    @property
    def distance_2d(self):
        r = diff(self.x)**2.0
        r += diff(self.y)**2.0
        distance = DataArray(zeros(self.x.size), 'Distance', self.x.units)
        distance[1:] = cumsum(sqrt(r))
        return distance

    @property
    def distance_3d(self):
        r = diff(self.x)**2.0
        r += diff(self.y)**2.0
        r += diff(self.z)**2.0
        distance = DataArray(zeros(self.x.size), 'Distance', self.x.units)
        distance[1:] = cumsum(sqrt(r))
        return distance

    def move(self, dx, dy, dz):
        """ Move the point by [dx,dy,dz] """
        self._x += dx
        self._y += dy
        self._z += dz
        self._elevation += dz
        return self

    def perturb(self):
        """Propose a new point given the attached propsal distributions
        """

        for c in [self.x, self.y, self.z]:
            if c.hasPosterior:
                c.perturb(imposePrior=True, log=True)
                # Update the mean of the proposed elevation
                c.proposal.mean = c

    # def point(self, index):
    #     """Get a point from the 3D Point Cloud

    #     Parameters
    #     ----------
    #     i : int
    #         The index of the point to return

    #     Returns
    #     -------
    #     out : geobipy.Point
    #         A point

    #     Raises
    #     ------
    #     ValueError : If i is not a single int

    #     """
    #     assert size(index) == 1, ValueError("i must be a single integer")
    #     assert 0 <= index <= self.nPoints, ValueError("Must have 0 <= i <= {}".format(self.nPoints))
    #     return Point(self.x[index], self.y[index], self.z[index], self.elevation[index])

    def x_axis(self, xAxis='x'):
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
            return DataArray(arange(self.x.size), name="Index")
        elif xAxis == 'x':
            return self.x
        elif xAxis == 'y':
            return self.y
        elif xAxis == 'z':
            return self.z
        elif xAxis == 'r2d':
            return self.distance_2d
        elif xAxis == 'r3d':
            return self.distance_3d


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
            * 'ct' uses Clough Tocher interpolation. Default
            * 'mc' uses Minimum curvature and requires pygmt to be installed.
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
            return self._interp_clough_tocher(mesh, values, mask=mask, clip=clip, i=i, **kwargs)
        elif method.lower() == 'mc':
            return self._interp_minimum_curvature(mesh, values, mask=mask, clip=clip, i=i, **kwargs)
        elif method.lower() == 'sibson':
            return self._interp_sibson(mesh, values, mask=mask, clip=clip, i=i, **kwargs)

    def _interp_sibson(self, mesh, values, mask = False, clip = None, i=None, **kwargs):

        vals = sibson(self.x, self.y, values, grid_x=mesh.x.edges, grid_y=mesh.y.edges, max_distance=mask)

        vals = DataArray(vals, name=cf.getName(values), units = cf.getUnits(values))
        out =  Model(mesh=mesh, values=vals.T)
        return out, kwargs

    def _interp_clough_tocher(self, mesh, values, mask = False, clip = None, i=None, **kwargs):
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
                self.set_kdtree(ndim = 2)
                kdtree = self.kdtree

        # Create the CT function for interpolation
        f = CloughTocher2DInterpolator(XY, vTmp)

        query = hstack([mesh.centres(axis=0).flatten(), mesh.centres(axis=1).flatten()])

        # Interpolate to the grid
        vals = f(query).reshape(*mesh.shape)

        # Reshape to a 2D array
        vals = DataArray(vals, name=cf.getName(values), units = cf.getUnits(values))

        if mask or extrapolate:
            if (kdtree is None):
                kdt = cKDTree(XY)
            else:
                kdt = kdtree

        # Use distance masking
        # if mask:
        #     g = meshgrid(mesh.x.centres, mesh.y.centres)
        #     xi = _ndim_coords_from_arrays(tuple(g), ndim=XY.shape[1])
        #     dists, indexes = kdt.query(xi)

        #     vals[dists > mask] = nan

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

    def _interp_minimum_curvature(self, mesh, values, mask=False, clip=True, i=None, operator=None, condition=None, **kwargs):

        try:
            from pygmt import surface
        except Exception as e:
            raise Exception(("{}\npygmt not installed correctly.  method='mc' can only be used when pygmt is present.\n"
                             "To install pygmt, you need to use conda environments. Installing instructions are here\n"
                             "https://www.pygmt.org/latest/install.html \n"
                             "After creating a new conda environment do\n"
                             "'pip install -c conda-forge numpy pandas xarray netcdf4 packaging gmt pygmt'\n"
                             "Then install geobipy and its dependencies to that environment.").format(e))

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
            xr = surface(x=x, y=y, z=values, spacing=(dx, dy), region=mesh.centres_bounds, N=iterations, tension=tension, convergence=accuracy, lower=[clip_min], upper=[clip_max])
        else:
            xr = surface(x=x, y=y, z=values, spacing=(dx, dy), region=mesh.centres_bounds, N=iterations, tension=tension, convergence=accuracy)

        vals = xr.values

        if (mx - mn) != 0.0:
            vals = vals * (mx - mn)
        vals += mn

        # Use distance masking
        if mask:
            kdt = cKDTree(column_stack((x, y)))
            xi = cf._ndim_coords_from_arrays(tuple(meshgrid(mesh.x.centres, mesh.y.centres)), ndim=2)
            dists, indexes = kdt.query(xi)
            vals[dists > mask] = nan

        vals = DataArray(vals, name=cf.getName(values), units = cf.getUnits(values))

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

        k = minimum(self.nPoints, k)

        return self.kdtree.query(x, k, eps, p, distance_upper_bound=radius)


    def plot(self, values, x='index', **kwargs):
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
        x = self.x_axis(x)
        ax = cP.plot(x, values, **kwargs)
        return ax

    def plot_data_elevation(self, **kwargs):
        self.plot(values = self.z + self.elevation, **kwargs)

    def set_priors(self, x_prior=None, y_prior=None, z_prior=None, **kwargs):

        if x_prior is None:
            if kwargs.get('solve_x', False):
                x_prior = Distribution('Uniform', self.x - kwargs['maximum_x_change'], self.x + kwargs['maximum_x_change'], prng=kwargs.get('prng'))

        if y_prior is None:
            if kwargs.get('solve_y', False):
                y_prior = Distribution('Uniform', self.y - kwargs['maximum_y_change'], self.y + kwargs['maximum_y_change'], prng=kwargs.get('prng'))

        if z_prior is None:
            if kwargs.get('solve_z', False):
                z_prior = Distribution('Uniform', self.z - kwargs['maximum_z_change'], self.z + kwargs['maximum_z_change'], prng=kwargs.get('prng'))

        self.x.prior = x_prior
        self.y.prior = y_prior
        self.z.prior = z_prior

    def set_proposals(self, x_proposal=None, y_proposal=None, z_proposal=None, **kwargs):

        if x_proposal is None:
            if kwargs.get('solve_x', False):
                x_proposal = Distribution('Normal', self.x.item(), kwargs['x_proposal_variance'], prng=kwargs.get('prng'))

        if y_proposal is None:
            if kwargs.get('solve_y', False):
                y_proposal = Distribution('Normal', self.y.item(), kwargs['y_proposal_variance'], prng=kwargs.get('prng'))

        if z_proposal is None:
            if kwargs.get('solve_z', False):
                z_proposal = Distribution('Normal', self.z.item(), kwargs['z_proposal_variance'], prng=kwargs.get('prng'))

        self.x.proposal = x_proposal
        self.y.proposal = y_proposal
        self.z.proposal = z_proposal

    def reset_posteriors(self):
        self.x.reset_posteriors()
        self.y.reset_posteriors()
        self.z.reset_posteriors()

    def set_posteriors(self):

        self.set_x_posterior()
        self.set_y_posterior()
        self.set_z_posterior()

    def set_x_posterior(self):
        """

        """
        if self.x.hasPrior:
            mesh = RectilinearMesh1D(edges = DataArray(self.x.prior.bins(), name=self.x.name, units=self.x.units), relative_to=self.x)
            self.x.posterior = Histogram(mesh=mesh)

    def set_y_posterior(self):
        """

        """
        if isinstance(self.y, StatArray):
            if self.y.hasPrior:
                mesh = RectilinearMesh1D(edges = DataArray(self.y.prior.bins(), name=self.y.name, units=self.y.units), relative_to=self.y)
                self.y.posterior = Histogram(mesh=mesh)

    def set_z_posterior(self):
        """

        """
        if isinstance(self.z, StatArray):
            if self.z.hasPrior:
                mesh = RectilinearMesh1D(edges = DataArray(self.z.prior.bins(), name=self.z.name, units=self.z.units), relative_to=self.z)
                self.z.posterior = Histogram(mesh=mesh)

    def update_posteriors(self):
        self.x.update_posterior()
        self.y.update_posterior()
        self.z.update_posterior()

    def _init_posterior_plots(self, gs=None):
        """Initialize axes for posterior plots
        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split
        """
        n_posteriors = self.x.hasPosterior + self.y.hasPosterior + self.z.hasPosterior
        if n_posteriors == 0:
            return []

        if gs is None:
            gs = Figure()

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        splt = gs.subgridspec(n_posteriors, 1, wspace=0.3, hspace=1.0)

        ax = []
        i = 0
        for c in [self.x, self.y, self.z]:
            if c.hasPosterior:
                ax.append(c._init_posterior_plots(splt[i]))
                i += 1

        return ax

    def plot_posteriors(self, axes=None, **kwargs):

        n_posteriors = self.x.hasPosterior + self.y.hasPosterior + self.z.hasPosterior
        if n_posteriors == 0:
            return

        if axes is None:
            axes = kwargs.pop('fig', gcf())

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes)

        assert len(axes) == n_posteriors, ValueError("Must have length {} list of axes for the posteriors. self._init_posterior_plots can generate them.".format(n_posteriors))

        x_kwargs = kwargs.pop('x_kwargs', {})
        y_kwargs = kwargs.pop('y_kwargs', {})
        z_kwargs = kwargs.pop('z_kwargs', {})

        overlay = kwargs.pop('overlay', None)
        # if not overlay is None:
        #     x_kwargs['overlay'] = overlay.x
        #     y_kwargs['overlay'] = overlay.y
        #     z_kwargs['overlay'] = overlay.z

        if (not self.x.hasPosterior) & (not self.y.hasPosterior) & self.z.hasPosterior:
            z_kwargs['transpose'] = z_kwargs.get('transpose', True)

        i = 0
        for c, kw in zip([self.x, self.y, self.z], [x_kwargs, y_kwargs, z_kwargs]):
            if c.hasPosterior:
                c.plot_posteriors(ax = axes[i], **kw)
                i += 1

        if overlay is not None:
            axes = self.overlay_on_posteriors(overlay, axes)

    def overlay_on_posteriors(self, overlay, axes, x_kwargs={}, y_kwargs={}, z_kwargs={}, **kwargs):

        if (not self.x.hasPosterior) & (not self.y.hasPosterior) & self.z.hasPosterior:
            z_kwargs['transpose'] = z_kwargs.get('transpose', True)

        i = 0
        for s, o, kw in zip([self.x, self.y, self.z], [overlay.x, overlay.y, overlay.z], [x_kwargs, y_kwargs, z_kwargs]):
            if s.hasPosterior:
                s.posterior.plot_overlay(value = o, ax = axes[i], **kw, **kwargs)
                i += 1
        return axes


    def pyvista_mesh(self):
        import pyvista as pv

        out = pv.PolyData(vstack([self.x, self.y, self.z]).T)
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

        assert not any([x is None for x in labels[:3]]), Exception("File must contain columns for easting, northing, height. May also have an elevation column \n {}".format(self.fileInformation()))
        assert n == 3 and len(labels) <= 4, Exception("File must contain columns for easting, northing, height. May also have an elevation column \n {}".format(self.fileInformation()))
        return Point._csv_n_points(filename), labels

    @classmethod
    def read_csv(cls, filename, **kwargs):
        """Reads x y z co-ordinates from an ascii csv file.

        Parameters
        ----------
        filename : str
            Path to the file to read from.

        """
        nPoints, channels = Point._csv_channels(filename)

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


    def set_kdtree(self, ndim):
        """Creates a k-d tree of the point co-ordinates

        Parameters
        ----------
        nDims : int
            Either 2 or 3 to exclude or include the vertical co-ordinate

        """
        if (ndim == 2):
            self._kdtree = cKDTree(column_stack((self.x, self.y)))
        elif (ndim == 3):
            self._kdtree = cKDTree(column_stack((self.x, self.y, self.z)))

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
            assert isinstance(pointData, (DataArray, list)), TypeError("pointData must a geobipy.StatArray or a list of them.")
            if isinstance(pointData, list):
                for p in pointData:
                    assert isinstance(p, DataArray), TypeError("pointData entries must be a geobipy.StatArray")
                    assert size(p) == self.nPoints, ValueError("pointData entries must have size {}".format(self.nPoints))
                    assert p.hasLabels(), ValueError("StatArray needs a name")
                    vtk.point_data.append(Scalars(p, p.getNameUnits()))
            else:
                assert pointData.size == self.nPoints, ValueError("pointData entries must have sizd {}".format(self.nPoints))
                assert pointData.hasLabels(), ValueError("StatArray needs a name")
                vtk.point_data.append(Scalars(pointData, pointData.getNameUnits()))

        vtk.tofile(fileName, format=format)


    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)

        if self.x.size == 0:
            self.x = None
        self.x.createHdf(grp, 'x', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        if self.y.size == 0:
            self.y = None
        self.y.createHdf(grp, 'y', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        if self.z.size == 0:
            self.z = None
        self.z.createHdf(grp, 'z', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        if self.elevation.size == 0:
            self.elevation = None
        self.elevation.createHdf(grp, 'elevation', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

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
        self.z.writeHdf(grp, 'z',  withPosterior=withPosterior, index=index)
        self.elevation.writeHdf(grp, 'elevation',  withPosterior=withPosterior, index=index)

    @classmethod
    def fromHdf(cls, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        out = cls(**kwargs)
        if 'x' in grp:
            out.x = StatArray.fromHdf(grp['x'], index=index)
        if 'y' in grp:
            out.y = StatArray.fromHdf(grp['y'], index=index)
        if 'z' in grp:
            out.z = StatArray.fromHdf(grp['z'], index=index)
        if 'elevation' in grp:
            out.elevation = StatArray.fromHdf(grp['elevation'], index=index)

        return out

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
        # if self.x.size == 0:
        #     self.x = None
        self.x.Isend(dest, world)
        # if self.y.size == 0:
        #     self.y = None
        self.y.Isend(dest, world)
        # if self.z.size == 0:
        #     self.z = None
        self.z.Isend(dest, world)
        # if self.elevation.size == 0:
        #     self.elevation = None
        self.elevation.Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world, **kwargs):
        out = cls(**kwargs)
        out._x = DataArray.Irecv(source, world)
        out._y = DataArray.Irecv(source, world)
        out._z = DataArray.Irecv(source, world)
        out._elevation = DataArray.Irecv(source, world)

        return out

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
