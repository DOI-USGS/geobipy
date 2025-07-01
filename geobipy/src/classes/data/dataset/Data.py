""" @Data_Class
Module describing a Data Set where values are associated with an xyz co-ordinate
"""
from copy import copy, deepcopy

from numpy import allclose, any, arange, asarray, atleast_1d, atleast_2d, cumsum, diff, float64, full
from numpy import hstack, int32, isnan, nan, ndim
from numpy import ones, r_, s_, shape, size, sqrt, sum, unique
from numpy import vstack, where, zeros
from numpy import all as npall

from numpy.ma import MaskedArray

from pandas import DataFrame, read_csv
from ...core.DataArray import DataArray
from ...statistics.StatArray import StatArray
from ....base import fileIO as fIO
from ....base import utilities as cf
from ....base import plotting as cP
from ...pointcloud.Point import Point
from ..datapoint.DataPoint import DataPoint
from ....base import MPI as myMPI
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

try:
    from pyvtk import Scalars
except:
    pass


class Data(Point):
    """Class defining a set of Data.


    Data(channels_per_system, x, y, z, data, std, predicted_data, dataUnits, channel_names)

    Parameters
    ----------
    nPoints : int
        Number of points in the data.
    channels_per_system : int or array_like
        Number of data channels in the data
        * If int, a single acquisition system is assumed.
        * If array_like, each item describes the number of points per acquisition system.
    x : geobipy.StatArray or array_like, optional
        The x co-ordinates. Default is zeros of size nPoints.
    y : geobipy.StatArray or array_like, optional
        The y co-ordinates. Default is zeros of size nPoints.
    z : geobipy.StatArrayor array_like, optional
        The z co-ordinates. Default is zeros of size nPoints.
    data : geobipy.StatArrayor array_like, optional
        The values of the data.
        * If None, zeroes are assigned
    std : geobipy.StatArrayor array_like, optional
        The uncertainty estimates of the data.
        * If None, ones are assigned if data is None, else 0.1*data
    predicted_data : geobipy.StatArrayor array_like, optional
        The predicted data.
        * If None, zeros are assigned.
    dataUnits : str
        Units of the data.
    channel_names : list of str, optional
        Names of each channel of length sum(channels_per_system)

    Returns
    -------
    out : Data
        Data class

    """
    __slots__ = ('_units', '_components', '_channel_names', '_channels_per_system', '_fiducial', '_file',
                 '_data_filename', '_line_number', '_data', '_predicted_data', '_std', '_relative_error', '_additive_error',
                 '_system', '_iC', '_iR', '_iT', '_iOffset', '_iData', '_iStd', '_iPrimary', '_channels')

    def __init__(self, components=None, channels_per_system=0, x=None, y=None, z=None, elevation=None, data=None, std=None, predicted_data=None, fiducial=None, line_number=None, units=None, channel_names=None, **kwargs):
        """ Initialize the Data class """

        # Number of Channels
        self.units = units
        self.components = components
        self._channels_per_system = atleast_1d(asarray(channels_per_system, dtype=int32)).copy()

        super().__init__(x, y, z, elevation)

        self._fiducial = DataArray(arange(self.nPoints, dtype=float64), "Fiducial")
        self._line_number = DataArray(self.nPoints, "Line number")

        shp = (self._nPoints, self.nChannels)
        self._data = DataArray(shp, "Data", self.units)
        self._predicted_data = DataArray(shp, "Predicted Data", self.units)
        self._std = DataArray(ones(shp), "std", self.units)

        shp = (self.nPoints, self.nSystems)
        self._relative_error = DataArray(full(shp, fill_value=0.01), "Relative error", "%")
        self._additive_error = DataArray(shp, "Additive error", self.units)

        self.fiducial = fiducial
        self.line_number = line_number
        self.data = data
        self.std = std
        self.predicted_data = predicted_data
        self.channel_names = channel_names
        # self.relative_error = None
        # self.additive_error = None

        # self.error_posterior = None

    def _reconcile_channels(self, channels):

        channels = super()._reconcile_channels(channels)
        for i, channel in enumerate(channels):
            channel = channel.lower()
            if(channel in ['line']):
                channels[i] = 'line'
            elif(channel in ['id', 'fid', 'fiducial']):
                channels[i] = 'fiducial'

        return channels

    def _as_dict(self):
        out, order = super()._as_dict()
        out[self.fiducial.name.replace(' ', '_')] = self.fiducial
        out[self.line_number.name.replace(' ', '_')] = self.line_number
        for i, name in enumerate(self.channel_names):
            out[name.replace(' ', '_')] = self.data[:, i]

        return out, [self.line_number.name.replace(' ', '_'),
                     self.fiducial.name.replace(' ', '_'), *order, *[x.replace(' ', '_') for x in self.channel_names]]

    @property
    def active(self):
        """Logical array whether the channel is active or not.

        An inactive channel is one where channel values are NaN for all points.

        Returns
        -------
        out : bools
            Indices of non-NaN columns.

        """
        out = ~isnan(self.data)
        out.name = "Active"
        out.units = None
        return out

    @property
    def channel_saturation(self):
        out = 100.0 * sum(self.active, axis=1) / self.nChannels
        out.name = '% of active channels'
        out.units = '%'
        return out

    @property
    def active_channel(self):
        return any(self.active, axis=0)

    @property
    def additive_error(self):
        """The data. """
        if size(self._additive_error, 0) == 0:
            self._additive_error = DataArray((self.nPoints, self.nSystems), "Additive error", self.units)
        return self._additive_error

    @additive_error.setter
    def additive_error(self, values):
        if values is not None:
            self.nPoints = size(values, 0)
            shp = (self.nPoints, self.nSystems)
            if not allclose(self._additive_error.shape, shp):
                self._additive_error = DataArray(values, "Additive error", self.units)
                return

            self._additive_error[:, :] = values

    @property
    def channel_names(self):
        return self._channel_names

    @channel_names.setter
    def channel_names(self, values):
        if values is None:
            self._channel_names = ['Channel {}'.format(i) for i in range(self.nChannels)]
        else:
            assert all((isinstance(x, str) for x in values))
            assert len(values) == self.nChannels, Exception("Length of channel_names must equal total number of channels {}".format(self.nChannels))
            self._channel_names = values

    def channel_index(self, channel, system):
        """Gets the data in the specified channel

        Parameters
        ----------
        channel : int
            Index of the channel to return
            * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
        system : int, optional
            The system to obtain the channel from.

        Returns
        -------
        out : int
            The index of the channel

        """
        assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
        assert npall(channel < self.channels_per_system[system]), ValueError("channel must be < {} for system {}".format(self.channels_per_system[system], system))
        return self.systemOffset[system] + channel

    @property
    def channels_per_system(self):
        return self._channels_per_system

    # @channels_per_system.setter
    # def channels_per_system(self, values):
    #     if values is None:
    #         values = zeros(1, dtype=int32)
    #     else:
    #         values = atleast_1d(asarray(values, dtype=int32))

    #     self._channels_per_system = values

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, values):

        if values is None:
            values = ['z']
        else:

            if isinstance(values, str):
                values = [values]

            assert all([isinstance(x, str) for x in values]), TypeError('components_per_channel must be list of str')

        self._components = values

    @property
    def data(self):
        """The data. """
        if size(self._data, 0) == 0 or (self._data.shape[0] != self.nPoints):
            self._data = DataArray((self.nPoints, self.nChannels), "Data", self.units)
        return self._data

    @data.setter
    def data(self, values):
        if values is not None:
            values = atleast_2d(values)
            self.nPoints, self.nChannels = size(values, 0), size(values, 1)

            shp = (self.nPoints, self.nChannels)
            if not allclose(self._data.shape, shp):
                self._data = DataArray(values, "Data", self.units)
                return

            self._data[:, :] = values


    @property
    def deltaD(self):
        r"""Get the difference between the predicted and observed data,

        .. math::
            \delta \mathbf{d} = \mathbf{d}^{pre} - \mathbf{d}^{obs}.

        Returns
        -------
        out : StatArray
            The residual between the active observed and predicted data.

        """
        return self.predicted_data - self.data

    @property
    def fiducial(self):
        if size(self._fiducial) == 0:
            self._fiducial = DataArray(arange(self.nPoints, dtype=float64), "Fiducial")
        return self._fiducial

    @fiducial.setter
    def fiducial(self, values):
        if values is not None:
            self.nPoints = size(values)
            if self._fiducial.size != self.nPoints:
                self._fiducial = DataArray(values.astype(float64), "Fiducial")
                return

            self._fiducial[:] = values

    @property
    def line_number(self):
        if size(self._line_number) == 0:
            self._line_number = DataArray(self.nPoints, "Line number")
        return self._line_number

    @line_number.setter
    def line_number(self, values):
        if values is not None:
            self.nPoints = size(values)
            if self._line_number.size != self.nPoints:
                self._line_number = DataArray(values, "Line number")
                return

            self._line_number[:] = values

    @property
    def nActiveChannels(self):
        return sum(self.active, axis=1)

    @property
    def nChannels(self):
        return sum(self.channels_per_system)

    @nChannels.setter
    def nChannels(self, value):
        if (sum(self.channels_per_system) == 0) and (value > 0):
            self._channels_per_system = int32(value)

    @property
    def n_components(self):
        return size(self.components)

    @property
    def n_data_channels(self):
        return self.nChannels

    @property
    def nLines(self):
        return unique(self.line_number).size

    @property
    def n_posteriors(self):
        return super().n_posteriors + self.relative_error.n_posteriors + self.additive_error.n_posteriors + self.receiver.n_posteriors + self.transmitter.n_posteriors

    @property
    def nSystems(self):
        return size(self.channels_per_system)

    @property
    def predicted_data(self):
        """The predicted data. """
        if size(self._predicted_data, 0) == 0:
            self._predicted_data = DataArray((self.nPoints, self.nChannels), "Predicted Data", self.units)
        return self._predicted_data

    @predicted_data.setter
    def predicted_data(self, values):

        if values is not None:
            values = atleast_2d(values)
            self.nPoints, self.nChannels = size(values, 0), size(values, 1)

            shp = (self.nPoints, self.nChannels)
            if not allclose(self._predicted_data.shape, shp):
                self._predicted_data = DataArray(values, "Predicted Data", self.units)
                return

            self._predicted_data[:, :] = values

    @property
    def relative_error(self):
        """The data. """
        if size(self._relative_error, 0) == 0:
            self._relative_error = DataArray(full((self.nPoints, self.nSystems), fill_value=0.01), "Relative error", "%")
        return self._relative_error

    @relative_error.setter
    def relative_error(self, values):
        if values is not None:
            self.nPoints = size(values, 0)
            shp = (self.nPoints, self.nSystems)
            if not allclose(self._relative_error.shape, shp):
                self._relative_error = DataArray(values, "Relative error", "%")
                return

            self._relative_error[:, :] = values

    @property
    def std(self):
        shp = (self.nPoints, self.nChannels)
        if not allclose(self._std.shape, shp):
            self._std = DataArray(shp, "Standard deviation", self.units)

        relative_error = self.relative_error * self.data
        self._std[:, :] = sqrt((relative_error**2.0) + (self.additive_error**2.0))

        return self._std

    @std.setter
    def std(self, values):
        if values is not None:
            values = atleast_2d(values)
            self.nPoints, self.nChannels = size(values, 0), size(values, 1)

            shp = (self.nPoints, self.nChannels)
            if not allclose(self._std.shape, shp):
                self._std = DataArray(values, "Std", self.units)
                return

            self._std[:, :] = values

    @property
    def summary(self):
        """ Display a summary of the Data """
        msg = super().summary
        names = copy(self.channel_names)
        j = arange(5, self.nChannels, 5)
        for i in range(j.size):
            names.insert(j[i]+i, '\n')

        msg += "channel names:\n{}\n".format("|   "+(', '.join(names).replace("\n,", "\n|  ")))
        msg += "data:\n{}\n".format("|   "+(self.data[self.active].summary.replace("\n", "\n|   "))[:-4])
        msg += "predicted data:\n{}\n".format("|   "+(self.predicted_data[self.active].summary.replace("\n", "\n|   "))[:-4])
        msg += "std:\n{}\n".format("|   "+(self.std[self.active].summary.replace("\n", "\n|   "))[:-4])
        msg += "line number:\n{}\n".format("|   "+(self.line_number.summary.replace("\n", "\n|   "))[:-4])
        msg += "fiducial:\n{}\n".format("|   "+(self.fiducial.summary.replace("\n", "\n|   "))[:-4])
        msg += "relative error:\n{}\n".format("|   "+(self.relative_error.summary.replace("\n", "\n|   "))[:-4])
        msg += "additive error:\n{}\n".format("|   "+(self.additive_error.summary.replace("\n", "\n|   "))[:-4])

        return msg

    @property
    def systemOffset(self):
        return r_[0, cumsum(self.channels_per_system)]

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, values):
        if values is None:
            self._units = None
        else:
            assert isinstance(values, str)
            self._units = values

    @property
    def shape(self):
        return (self.nPoints, self.nChannels)

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out._fiducial = deepcopy(self.fiducial, memo)
        out._line_number = deepcopy(self.line_number, memo)
        out._channel_names = deepcopy(self.channel_names, memo)
        out._components = deepcopy(self.components, memo)
        out._data = deepcopy(self.data, memo)
        out._std = deepcopy(self.std, memo)
        out._predicted_data = deepcopy(self.predicted_data, memo)
        out._relative_error = deepcopy(self.relative_error, memo)
        out._additive_error = deepcopy(self._additive_error, memo)
        return out

    def addToVTK(self, vtk, prop=['data', 'predicted', 'std'], system=None):
        """Adds a member to a VTK handle.

        Parameters
        ----------
        vtk : pyvtk.VtkData
            vtk handle returned from self.vtkStructure()
        prop : str or list of str, optional
            List of the member to add to a VTK handle, either "data", "predicted", or "std".
        system : int, optional
            The system for which to add the data

        """


        if isinstance(prop, str):
            prop = [prop]

        for p in prop:
            assert p in ['data', 'predicted', 'std'], ValueError("prop must be either 'data', 'predicted' or 'std'.")
            if p == "data":
                tmp = self.data
            elif p == "predicted":
                tmp = self.predicted_data
            elif p == "std":
                tmp = self.std

            if system is None:
                r = range(self.nChannels)
            else:
                assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
                r = range(self.systemOffset[system], self.systemOffset[system+1])

            for i in r:
                vtk.point_data.append(Scalars(tmp[:, i], "{} {}".format(self.channel_names[i], tmp.getNameUnits())))

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


        # Get the column headers of the data file
        channels = fIO.get_column_name(filename)
        nChannels = len(channels)

        line_names = ('line', 'line_number', 'line_number')
        fiducial_names = ('fid', 'fiducial', 'id')

        n = 0
        labels = [None]*2

        for channel in channels:
            cTmp = channel.lower()
            if (cTmp in line_names):
                n += 1
                labels[0] = channel
            elif (cTmp in fiducial_names):
                n += 1
                labels[1] = channel

        assert n == 2, Exception("File {} must contain columns for line and fiducial. \n {}".format(filename, Data.fileInformation()))

        nPoints, ixyz = Point._csv_channels(filename)
        return nPoints, labels + ixyz

    def _open_csv_files(self, filename):

        channels = self.csv_channels(filename)
        try:
            df = read_csv(filename, index_col=False, usecols=channels, chunksize=1, skipinitialspace = True)
        except:
            df = read_csv(filename, index_col=False, usecols=channels, chunksize=1, sep=r"\s+", skipinitialspace = True)


        self._file = df
        self._data_filename = filename

        self._read_line_fiducial(filename)

    def close(self):
        self._file.close()

    def _read_line_fiducial(self, filename):

        _, channels = Data._csv_channels(filename)

        try:
            df = read_csv(filename, index_col=False, usecols=channels[:2], skipinitialspace = True)
        except:
            df = read_csv(filename, index_col=False, usecols=channels[:2], sep=r"\s+", skipinitialspace = True)

        df = df.replace('NaN',nan)
        self.line_number = df[channels[0]].values
        self.fiducial = df[channels[1]].values

    def _systemIndices(self, system=None):
        """The slice indices for the requested system.

        Parameters
        ----------
        system : int
            Requested system index.

        Returns
        -------
        out : numpy.slice
            The slice pertaining to the requested system.

        """

        if system is None:
            return [s_[self.systemOffset[x]:self.systemOffset[x+1]] for x in range(self.nSystems)]

        assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
        return s_[self.systemOffset[system]:self.systemOffset[system+1]]

    def append(self, other):

        self._relative_error = self.relative_error.append(other.relative_error)
        self._additive_error = self.additive_error.append(other.additive_error)

        super().append(other)
        self._fiducial = self._fiducial.append(other.fiducial)
        self._line_number = self._line_number.append(other.line_number)
        self._data = self._data.append(other.data, axis=0)
        self._predicted_data = self._predicted_data.append(other.predicted_data, axis=0)
        self._std = self._std.append(other.std, axis=0)

        return self

    def check_line_numbers(self):

        ln = unique(self.line_number)

        import numpy as np
        bad_lines = [l for l in ln if sum(diff(asarray(self.line_number == l, dtype=np.int8)) > 0) > 1]

        return bad_lines

    def data_misfit(self, squared=False):
        r"""Compute the :math:`L_{2}` norm squared misfit between the observed and predicted data

        .. math::
            \| \mathbf{W}_{d} (\mathbf{d}^{obs}-\mathbf{d}^{pre})\|_{2}^{2},

        where :math:`\mathbf{W}_{d}` are the reciprocal data errors.

        Parameters
        ----------
        squared : bool
            Return the squared misfit.

        Returns
        -------
        out : float64
            The misfit value.

        """
        x = MaskedArray((self.deltaD / self.std)**2.0, mask=~self.active)
        data_misfit = DataArray(sum(x, axis=1), "Data misfit")
        return data_misfit if squared else sqrt(data_misfit)

    def __getitem__(self, i):
        """ Define item getter for Data """
        if not isinstance(i, slice):
            i = unique(i)

        return type(self)(channels_per_system=self.channels_per_system,
                   x=self.x[i], y=self.y[i], z=self.z[i], elevation=self.elevation[i],
                   fiducial=self.fiducial[i], line_number=self.line_number[i],
                   data=self.data[i, :], std=self.std[i, :],
                   predicted_data=self.predicted_data[i, :],
                   channel_names=self.channel_names)

    # def dataChannel(self, channel, system=0):
    #     """Gets the data in the specified channel

    #     Parameters
    #     ----------
    #     channel : int
    #         Index of the channel to return
    #         * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
    #     system : int, optional
    #         The system to obtain the channel from.

    #     Returns
    #     -------
    #     out : geobipy.StatArray
    #         The data channel

    #     """
    #     assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
    #     assert channel < self.channels_per_system[system], ValueError("channel must be < {}".format(self.channels_per_system[system]))
    #     return self.data[:, self.systemOffset[system] + channel]


    def datapoint(self, i):
        """Get the ith data point from the data set

        Parameters
        ----------
        i : int
            The data point to get

        Returns
        -------
        out : geobipy.DataPoint
            The data point

        """
        assert size(i) == 1, ValueError("i must be a single integer")
        assert 0 <= i <= self.nPoints, ValueError("Must have 0 <= i <= {}".format(self.nPoints))
        return DataPoint(x=self.x[i], y=self.y[i], z=self.z[i], elevation=self.elevation[i],
                         data=self.data[i, :], std=self.std[i, :], predicted_data=self.predicted_data[i, :],
                         channel_names=self.channel_names)

    def _init_posterior_plots(self, gs):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        splt = gs.subgridspec(2, 2, wspace=0.3)
        ax = []
        # Height axis
        ax.append(plt.subplot(splt[0, 0]))
        # Data axis
        ax.append(plt.subplot(splt[0, 1], sharex=ax[0]))

        splt2 = splt[1, :].subgridspec(self.nSystems, 2, wspace=0.2)
        # Relative error axes
        ax.append([plt.subplot(splt2[i, 0], sharex=ax[0]) for i in range(self.nSystems)])
        # Additive Error axes
        ax.append([plt.subplot(splt2[i, 1], sharex=ax[0]) for i in range(self.nSystems)])

        return ax


    def line(self, line):
        """ Get the data from the given line number """
        if size(line) > 1:
            i = where(self.line_number == line[0])[0]
            for j in range(1, size(line)):
                i = hstack([i, where(self.line_number == line[j])[0]])
        else:
            i = where(self.line_number == line)[0]
        assert (i.size > 0), 'Could not get line with number {}'.format(line)
        return self[i]


    def nPointsPerLine(self):
        """Gets the number of points in each line.

        Returns
        -------
        out : ints
            Number of points in each line

        """
        nPoints = zeros(unique(self.line_number).size)
        lines = unique(self.line_number)
        for i, line in enumerate(lines):
            nPoints[i] = sum(self.line_number == line)
        return nPoints


    # def predicted_dataChannel(self, channel, system=None):
    #     """Gets the predicted data in the specified channel

    #     Parameters
    #     ----------
    #     channel : int
    #         Index of the channel to return
    #         * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
    #     system : int, optional
    #         The system to obtain the channel from.

    #     Returns
    #     -------
    #     out : geobipy.StatArray
    #         The predicted data channel

    #     """

    #     if system is None:
    #         return DataArray(self.predicted_data[:, channel], "Predicted data {}".format(self.channel_names[channel]), self.predicted_data.units)
    #     else:
    #         assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
    #         return DataArray(self.predicted_data[:, self.systemOffset[system] + channel], "Predicted data {}".format(self.channel_names[self.systemOffset[system] + channel]), self.predicted_data.units)


    # def stdChannel(self, channel, system=None):
    #     """Gets the uncertainty in the specified channel

    #     Parameters
    #     ----------
    #     channel : int
    #         Index of the channel to return
    #         * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
    #     system : int, optional
    #         The system to obtain the channel from.

    #     Returns
    #     -------
    #     out : geobipy.StatArray
    #         The uncertainty channel

    #     """

    #     if system is None:
    #         return DataArray(self.std[:, channel], "Std {}".format(self.channel_names[channel]), self.std.units)
    #     else:
    #         assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
    #         return DataArray(self.std[:, self.systemOffset[system] + channel], "Std {}".format(self.channel_names[self.systemOffset[system] + channel]), self.std.units)


    # def maketest(self, nPoints, nChannels):
    #     """ Create a test example """
    #     Data.__init__(self, nPoints, nChannels)   # Initialize the Data array
    #     # Use the PointCloud3D example creator
    #     PointCloud3D.maketest(self, nPoints)
    #     a = 1.0
    #     b = 2.0
    #     # Create different Rosenbrock functions as the test data
    #     for i in range(nChannels):
    #         tmp = cf.rosenbrock(self.x, self.y, a, b)
    #         # Put the tmp array into the data column
    #         self._data[:, i] = tmp[:]
    #         b *= 2.0


    def mapData(self, channel, system=None, *args, **kwargs):
        """Interpolate the data channel between the x, y co-ordinates.

        Parameters
        ----------
        channel : int
            Index of the channel to return
            * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
        system : int, optional
            The system to obtain the channel from.

        """

        if system is None:
            assert 0 <= channel < self.nChannels, ValueError('Requested channel must be 0 <= channel < {}'.format(self.nChannels))
        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            assert 0 <= channel < self.channels_per_system[system], ValueError('Requested channel must be 0 <= channel {}'.format(self.channels_per_system[system]))
            channel = self.systemOffset[system] + channel

        kwargs['values'] = self.data[:, channel]

        ax, _, _ = self.map(*args, **kwargs)

        ax.set_title(self.channel_names[channel])

        return ax


    def mapPredictedData(self, channel, system=None, *args, **kwargs):
        """Interpolate the predicted data channel between the x, y co-ordinates.

        Parameters
        ----------
        channel : int
            Index of the channel to return
            * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
        system : int, optional
            The system to obtain the channel from.

        """

        if system is None:
            assert 0 >= channel < self.nChannels, ValueError('Requested channel must be 0 <= channel < {}'.format(self.nChannels))
        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            assert 0 >= channel < self.nChannelsPerSystem[system], ValueError('Requested channel must be 0 <= channel {}'.format(self.nChannelsPerSystem[system]))
            channel = self.systemOffset[system] + channel

        kwargs['c'] = self.predicted_dataChannel(channel)

        self.map(*args, **kwargs)

        cP.title(self.channel_names[channel])


    def mapStd(self, channel, system=None, *args, **kwargs):
        """Interpolate the standard deviation channel between the x, y co-ordinates.

        Parameters
        ----------
        channel : int
            Index of the channel to return
            * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
        system : int, optional
            The system to obtain the channel from.

        """

        if system is None:
            assert 0 >= channel < self.nChannels, ValueError('Requested channel must be 0 <= channel < {}'.format(self.nChannels))
        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            assert 0 >= channel < self.nChannelsPerSystem[system], ValueError('Requested channel must be 0 <= channel {}'.format(self.nChannelsPerSystem[system]))
            channel = self.systemOffset[system] + channel

        kwargs['c'] = self.stdChannel(channel)

        self.map(*args, **kwargs)

        cP.title(self.channel_names[channel])


    def plot_data(self, x='index', channels=None, system=None, **kwargs):
        """Plots the specifed channels as a line plot.

        Plots the channels along a specified co-ordinate e.g. 'x'. A legend is auto generated.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'index', returns numpy.arange(self.nPoints)
            If xAxis is 'x', returns self.x
            If xAxis is 'y', returns self.y
            If xAxis is 'z', returns self.z
            If xAxis is 'r2d', returns cumulative distance along the line in 2D using x and y.
            If xAxis is 'r3d', returns cumulative distance along the line in 3D using x, y, and z.
        channels : ints, optional
            Indices of the channels to plot.  All are plotted if None
            * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
        values : arraylike, optional
            Specifies values to plot against the chosen axis. Takes precedence over channels.
        system : int, optional
            The system to obtain the channel from.
        legend : bool
            Attach a legend to the plot.  Default is True.

        Returns
        -------
        ax : matplotlib.axes
            Plot axes handle
        legend : matplotlib.legend.Legend
            The attached legend.

        See Also
        --------
        geobipy.plotting.plot : For additional keyword arguments

        """
        legend = kwargs.pop('legend', True)
        ax = kwargs.get('ax', plt.gca())
        ax.set_prop_cycle(None)

        if system is None:
            rTmp = s_[:] if channels is None else s_[channels]
        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            rTmp = self._systemIndices(system) if channels is None else channels + self._systemIndices(system).start

        if size(rTmp) == 1:
            rTmp = (rTmp)

        ax = super().plot(x=x, values=self.data[:, rTmp], **kwargs)

        if legend:
            # Put a legend to the right of the current axis
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
            leg.set_title(self._data.getNameUnits())
        else:
            leg = None

        return ax, leg

    def plot_posteriors(self, axes=None, height_kwargs={}, data_kwargs={}, rel_error_kwargs={}, add_error_kwargs={}, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', plt.gcf())

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes)

        assert len(axes) == 4, ValueError("axes must have length 4")
        # assert len(axes) == 4, ValueError("Must have length 3 list of axes for the posteriors. self.init_posterior_plots can generate them")

        self.z.plot_posteriors(ax = axes[0], **height_kwargs)

        self.plot(ax=axes[1], legend=False, **data_kwargs)
        self.plot_predicted(ax=axes[1], legend=False, **data_kwargs)

        self.relative_error.plot_posteriors(ax=axes[2], **rel_error_kwargs)
        self.additive_error.plot_posteriors(ax=axes[3], **add_error_kwargs)

        return axes

    def plot_predicted(self, xAxis='index', channels=None, system=None, **kwargs):
        """Plots the specifed predicted data channels as a line plot.

        Plots the channels along a specified co-ordinate e.g. 'x'. A legend is auto generated.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'index', returns numpy.arange(self.nPoints)
            If xAxis is 'x', returns self.x
            If xAxis is 'y', returns self.y
            If xAxis is 'z', returns self.z
            If xAxis is 'r2d', returns cumulative distance along the line in 2D using x and y.
            If xAxis is 'r3d', returns cumulative distance along the line in 3D using x, y, and z.
        channels : ints, optional
            Indices of the channels to plot.  All are plotted if None
            * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
        system : int, optional
            The system to obtain the channel from.
        noLegend : bool
            Do not attach a legend to the plot.  Default is False, a legend is attached.

        Returns
        -------
        ax : matplotlib.axes
            Plot axes handle
        legend : matplotlib.legend.Legend
            The attached legend.

        See Also
        --------
        geobipy.plotting.plot : For additional keyword arguments

        """

        legend = kwargs.pop('legend', True)
        kwargs['linestyle'] = kwargs.get('linestyle', '-.')

        ax = kwargs.get('ax', plt.gca())
        ax.set_prop_cycle(None)

        if system is None:
            rTmp = s_[:] if channels is None else s_[channels]
        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            rTmp = self._systemIndices(system) if channels is None else channels + self._systemIndices(system).start

        ax = super().plot(values=self.predicted_data[:, rTmp], xAxis=xAxis, label=self.channel_names[rTmp], **kwargs)

        if legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
            legend.set_title(self.predicted_data.getNameUnits())
        else:
            legend = None

        return ax, legend

    @classmethod
    def read_csv(cls, data_filename, **kwargs):
        """Reads the data and system parameters from file

        Parameters
        ----------
        dataFilename : str or list of str
            Time domain data file names
        systemFilename : str or list of str
            Time domain system file names

        Notes
        -----
        File Format

        The data columns are read in according to the column names in the first line.
        The header line should contain at least the following column names.
        Extra columns may exist, but will be ignored. In this description,
        the column name or its alternatives are given followed by what the name represents.
        Optional columns are also described.


        **Required columns**

        line
            Line number for the data point

        id or fid
            Id number of the data point, these be unique

        x or northing or n
            Northing co-ordinate of the data point

        y or easting or e
            Easting co-ordinate of the data point

        z or dtm or dem_elev or dem_np or topo
            Elevation of the ground at the data point

        alt or laser or bheight
            Altitude of the transmitter coil

        Off[0] to Off[nWindows-1]  (with the number and brackets)
           The measurements for each time specified in the accompanying system file under Receiver Window Times

        **Optional columns**

        If any loop orientation columns are omitted the loop is assumed to be horizontal.

        TxPitch
            Pitch of the transmitter loop
        TxRoll
            Roll of the transmitter loop
        TxYaw
            Yaw of the transmitter loop
        RxPitch
            Pitch of the receiver loop
        RxRoll
            Roll of the receiver loop
        RxYaw
            Yaw of the receiver loop

        OffErr[0] to ErrOff[nWindows-1]
            Error estimates for the data


        See Also
        --------
        INFORMATION ON TD SYSTEMS


        """

        self = cls(**kwargs)

        self._nPoints, iC,  = Data._csv_channels(data_filename)

        assert len(iData) == self.nChannels, Exception("Number of off time columns {} in {} does not match total number of times {} in system files \n {}".format(
            len(iData), data_filename, self.nChannels, self.fileInformation()))

        if len(iStd) > 0:
            assert len(iStd) == len(iData), Exception(
                "Number of Off time standard deviation estimates does not match number of Off time data columns in file {}. \n {}".format(data_filename, self.fileInformation()))

        # Get all readable column indices for the first file.
        channels = iC + iR + iT + iOffset + iData
        if len(iStd) > 0:
            channels += iStd

        # Read in the columns from the first data file
        try:
            df = read_csv(data_filename, usecols=channels, skipinitialspace = True)
        except:
            df = read_csv(data_filename, usecols=channels, delim_whitespace=True, skipinitialspace = True)
        df = df.replace('NaN', nan)

        # Assign columns to variables
        self.line_number = df[iC[0]].values
        self.fiducial = df[iC[1]].values
        self.x = df[iC[2]].values
        self.y = df[iC[3]].values
        self.z = df[iC[4]].values
        self.elevation = df[iC[5]].values

        self.check()

        return self


    def pyvista_mesh(self):
        import pyvista as pv

        out = super().pyvista_mesh()

        out[self.data.label] = self.data
        out[self.predicted_data.label] = self.predicted_data
        out[self.std.label] = self.std

        return out


    def createHdf(self, parent, myName, withPosterior=True, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = super().createHdf(parent, myName, withPosterior, fillvalue)

        self.fiducial.createHdf(grp, 'fiducial', fillvalue=fillvalue)
        self.line_number.createHdf(grp, 'line_number', fillvalue=fillvalue)
        self.data.createHdf(grp, 'data', withPosterior=withPosterior, fillvalue=fillvalue)
        self.std.createHdf(grp, 'std', withPosterior=withPosterior, fillvalue=fillvalue)
        self.predicted_data.createHdf(grp, 'predicted_data', withPosterior=withPosterior, fillvalue=fillvalue)

        self.relative_error.createHdf(grp, 'relative_error', withPosterior=withPosterior, fillvalue=fillvalue)
        self.additive_error.createHdf(grp, 'additive_error', withPosterior=withPosterior, fillvalue=fillvalue)

        return grp


    def writeHdf(self, parent, name, withPosterior=True):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        super().writeHdf(parent, name, withPosterior)

        grp = parent[name]

        self.fiducial.writeHdf(grp, 'fiducial')
        self.line_number.writeHdf(grp, 'line_number')

        self.data.writeHdf(grp, 'data',  withPosterior=withPosterior)
        self.std.writeHdf(grp, 'std',  withPosterior=withPosterior)
        self.predicted_data.writeHdf(grp, 'predicted_data',  withPosterior=withPosterior)

        self.relative_error.writeHdf(grp, 'relative_error',  withPosterior=withPosterior)
        self.additive_error.writeHdf(grp, 'additive_error',  withPosterior=withPosterior)

    @classmethod
    def fromHdf(cls, grp, **kwargs):
        """ Reads the object from a HDF group """

        self = super(Data, cls).fromHdf(grp, **kwargs)

        self.fiducial = DataArray.fromHdf(grp['fiducial'])
        self.line_number = DataArray.fromHdf(grp['line_number'])

        self.data = DataArray.fromHdf(grp['data'])
        self.std = DataArray.fromHdf(grp['std'])
        self.predicted_data = DataArray.fromHdf(grp['predicted_data'])

        self.relative_error = DataArray.fromHdf(grp['relative_error'])
        self.additive_error = DataArray.fromHdf(grp['additive_error'])

        return self

    def write_csv(self, filename, **kwargs):
        kwargs['na_rep'] = 'nan'
        kwargs['index'] = False

        d, order = self._as_dict()

        kwargs['columns'] = order
        df = DataFrame(data=d)

        df.to_csv(filename, **kwargs)


    def Bcast(self, world, root=0):
        """Broadcast a Data object using MPI

        Parameters
        ----------
        world : mpi4py.MPI.COMM_WORLD
            MPI communicator
        root : int, optional
            The MPI rank to broadcast from. Default is 0.

        Returns
        -------
        out : geobipy.Data
            Data broadcast to each core in the communicator

        """

        out = super().Bcast(world, root=root)

        out.components = world.bcast(self.components, root=root)
        out.channels_per_system = myMPI.Bcast(self.channels_per_system, world, root=root)

        out.fiducial = self.fiducial.Bcast(world, root=root)
        out.line_number = self.line_number.Bcast(world, root=root)
        out._data = self.data.Bcast(world, root=root)
        out._std = self.std.Bcast(world, root=root)
        out._predicted_data = self.predicted_data.Bcast(world, root=root)

        return out

    def Scatterv(self, starts, chunks, world, root=0):
        """Scatterv a Data object using MPI

        Parameters
        ----------
        starts : array of ints
            1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
        chunks : array of ints
            1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
        world : mpi4py.MPI.Comm
            The MPI communicator over which to Scatterv.
        root : int, optional
            The MPI rank to broadcast from. Default is 0.

        Returns
        -------
        out : geobipy.Data
            The Data distributed amongst ranks.

        """
        out = super().Scatterv(starts, chunks, world, root)

        out.channels_per_system = myMPI.Bcast(self.channels_per_system, world, root=root)
        out.fiducial = self.fiducial.Scatterv(starts, chunks, world, root=root)
        out.line_number = self.line_number.Scatterv(starts, chunks, world, root=root)
        out.data = self.data.Scatterv(starts, chunks, world, root=root)
        out.std = self.std.Scatterv(starts, chunks, world, root=root)
        out.predicted_data = self.predicted_data.Scatterv(starts, chunks, world, root=root)
        return out
