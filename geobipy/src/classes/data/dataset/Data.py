""" @Data_Class
Module describing a Data Set where values are associated with an xyz co-ordinate
"""
from copy import deepcopy
import numpy as np
from pandas import read_csv
from cached_property import cached_property
from ....classes.core import StatArray
from ....base import fileIO as fIO
from ....base import utilities as cf
from ....base import plotting as cP
from ...pointcloud.PointCloud3D import PointCloud3D
from ..datapoint.DataPoint import DataPoint
from ....classes.core.myObject import myObject
from ....base import MPI as myMPI
import matplotlib.pyplot as plt

try:
    from pyvtk import Scalars
except:
    pass


class Data(PointCloud3D):
    """Class defining a set of Data.


    Data(nPoints, nChannelsPerSystem, x, y, z, data, std, predictedData, dataUnits, channelNames)

    Parameters
    ----------
    nPoints : int
        Number of points in the data.
    nChannelsPerSystem : int or array_like
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
    predictedData : geobipy.StatArrayor array_like, optional
        The predicted data.
        * If None, zeros are assigned.
    dataUnits : str
        Units of the data.
    channelNames : list of str, optional
        Names of each channel of length sum(nChannelsPerSystem)

    Returns
    -------
    out : Data
        Data class

    """

    def __init__(self, channels_per_system=1, x=None, y=None, z=None, elevation=None, data=None, std=None, predictedData=None, fiducial=None, lineNumber=None, units=None, channelNames=None, **kwargs):
        """ Initialize the Data class """

        # Number of Channels
        self._channels_per_system = channels_per_system

        super().__init__(x, y, z, elevation)

        self.fiducial = fiducial
        self.lineNumber = lineNumber
        self.units = units
        self.data = data
        self.std = std
        self.predictedData = predictedData
        self.channelNames = channelNames

        self.error_posterior = None

    def _reconcile_channels(self, channels):

        channels = super()._reconcile_channels(channels)
        for i, channel in enumerate(channels):
            channel = channel.lower()
            if(channel in ['line']):
                channels[i] = 'line'
            elif(channel in ['id', 'fid', 'fiducial']):
                channels[i] = 'fiducial'

        return channels

    @property
    def active(self):
        """Logical array whether the channel is active or not.

        An inactive channel is one where channel values are NaN for all points.

        Returns
        -------
        out : bools
            Indices of non-NaN columns.

        """
        return ~np.isnan(self.data)

    @property
    def active_channel(self):
        return np.any(self.active, axis=0)

    @property
    def additive_error(self):
        """The data. """
        if np.size(self._additive_error, 0) == 0:
            self._additive_error = StatArray.StatArray((self.nPoints, self.nSystems), "Additive error", "%")
        return self._additive_error

    @additive_error.setter
    def additive_error(self, values):
        shp = (self.nPoints, self.nSystems)
        if values is None:
            self._additive_error = StatArray.StatArray(np.ones(shp), "Additive error", "%")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values, 0)
                shp = (self.nPoints, self.nSystems)
            assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("additive_error must have shape {}".format(shp))
            self._additive_error = StatArray.StatArray(values)

    @property
    def channelNames(self):
        return self._channelNames

    @channelNames.setter
    def channelNames(self, values):
        if values is None:
            self._channelNames = ['Channel {}'.format(i) for i in range(self.nChannels)]
        else:
            assert all((isinstance(x, str) for x in values))
            assert len(values) == self.nChannels, Exception("Length of channelNames must equal total number of channels {}".format(self.nChannels))
            self._channelNames = values

    @property
    def channels_per_system(self):
        return self._channels_per_system

    @channels_per_system.setter
    def channels_per_system(self, values):
        self._channels_per_system = np.asarray(values, dtype=np.int32)

    @property
    def data(self):
        """The data. """
        if np.size(self._data, 0) == 0:
            self._data = StatArray.StatArray((self.nPoints, self.nChannels), "Data", self.units)
        return self._data


    @data.setter
    def data(self, values):
        shp = (self.nPoints, self.nChannels)
        if values is None:
            self._data = StatArray.StatArray(shp, "Data", self.units)
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values, 0)
                shp = (self.nPoints, self.nChannels)
            assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("data must have shape {}".format(shp))
            self._data = StatArray.StatArray(values)

    @property
    def deltaD(self):
        """Get the difference between the predicted and observed data,

        .. math::
            \delta \mathbf{d} = \mathbf{d}^{pre} - \mathbf{d}^{obs}.

        Returns
        -------
        out : StatArray
            The residual between the active observed and predicted data.

        """
        return self.predictedData - self.data

    @property
    def fiducial(self):
        return self._fiducial

    @fiducial.setter
    def fiducial(self, values):
        if (values is None):
            self._fiducial = StatArray.StatArray(self.nPoints, "Fiducial")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("fiducial must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._fiducial = deepcopy(values)
            else:
                self._fiducial = StatArray.StatArray(values, "Fiducial")

    @property
    def lineNumber(self):
        return self._lineNumber

    @lineNumber.setter
    def lineNumber(self, values):
        if (values is None):
            self._lineNumber = StatArray.StatArray(self.nPoints, "Line number")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("lineNumber must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._lineNumber = deepcopy(values)
            else:
                self._lineNumber = StatArray.StatArray(values, "Line number")

    @property
    def nActiveChannels(self):
        return np.sum(self.active, axis=1)

    @property
    def nChannels(self):
        return np.sum(self.channels_per_system)

    @property
    def nLines(self):
        return np.unique(self.lineNumber).size

    @property
    def nSystems(self):
        return np.size(self.channels_per_system)

    @property
    def predictedData(self):
        """The predicted data. """
        if np.size(self._predictedData, 0) == 0:
            self._predictedData = StatArray.StatArray((self.nPoints, self.nChannels), "Predicted Data", self.units)
        return self._predictedData

    @predictedData.setter
    def predictedData(self, values):
        shp = (self.nPoints, self.nChannels)
        if values is None:
            self._predictedData = StatArray.StatArray(shp, "Predicted Data", self.units)
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values, 0)
                shp = (self.nPoints, self.nChannels)
            assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("predictedData must have shape {}".format(shp))
            self._predictedData = StatArray.StatArray(values)

    @property
    def relative_error(self):
        """The data. """
        if np.size(self._relative_error, 0) == 0:
            self._relative_error = StatArray.StatArray((self.nPoints, self.nSystems), "Relative error", "%")
        return self._relative_error

    @relative_error.setter
    def relative_error(self, values):
        shp = (self.nPoints, self.nSystems)
        if values is None:
            self._relative_error = StatArray.StatArray(np.ones(shp), "Relative error", "%")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values, 0)
                shp = (self.nPoints, self.nSystems)
            assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("relative_error must have shape {}".format(shp))
            self._relative_error = StatArray.StatArray(values)

    @property
    def std(self):
        """The data. """
        if np.size(self._std, 0) == 0:
            self._std = StatArray.StatArray((self.nPoints, self.nChannels), "Standard deviation", self.units)
        return self._std

    @std.setter
    def std(self, values):
        shp = (self.nPoints, self.nChannels)
        if values is None:
            self._std = StatArray.StatArray(np.ones(shp), "Standard deviation", self.units)
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values, 0)
                shp = (self.nPoints, self.nChannels)
            assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("std must have shape {}".format(shp))
            self._std = StatArray.StatArray(values)

    @property
    def summary(self):
        """ Display a summary of the Data """
        msg = ("{}"
              "Data:          : \n"
              "# of Channels: {} \n"
              "# of Total Data: {} \n"
              "{}\n {}\n {}\n").format(super().summary, self.nChannels, self.nPoints * self.nChannels, self.data.summary, self.std.summary, self.predictedData.summary)
        return msg

    @property
    def systemOffset(self):
        return np.r_[0, np.cumsum(self.channels_per_system)]

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
                tmp = self.predictedData
            elif p == "std":
                tmp = self.std

            if system is None:
                r = range(self.nChannels)
            else:
                assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
                r = range(self.systemOffset[system], self.systemOffset[system+1])

            for i in r:
                vtk.point_data.append(Scalars(tmp[:, i], "{} {}".format(self.channelNames[i], tmp.getNameUnits())))

    def _csv_channels(self, filename):
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

        line_names = ('line', 'linenumber')
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

        assert n == 2, Exception("File {} must contain columns for line and fiducial. \n {}".format(filename, self.fileInformation()))

        return super()._csv_channels(filename), labels

    def _open_csv_files(self, filename):
        self._file = []
        channels = self.csv_channels(filename)

        if isinstance(filename, str):
            filename = [filename]

        for i, f in enumerate(filename):
            try:
                df = read_csv(f, index_col=False, usecols=channels[i], chunksize=1, skipinitialspace = True)
            except:
                df = read_csv(f, index_col=False, usecols=channels[i], chunksize=1, delim_whitespace=True, skipinitialspace = True)

            self._file.append(df)

    def _read_csv_line_fiducial(self, filename=None):

        if isinstance(filename, str):
            filename = [filename]

        _, channels = Data._csv_channels(self, filename[0])

        try:
            df = read_csv(filename[0], index_col=False, usecols=channels, skipinitialspace = True)
        except:
            df = read_csv(filename[0], index_col=False, usecols=channels, delim_whitespace=True, skipinitialspace = True)

        df = df.replace('NaN',np.nan)
        return df[channels[0]].values, df[channels[1]].values

    def _systemIndices(self, system=0):
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

        assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))

        return np.s_[self.systemOffset[system]:self.systemOffset[system+1]]

    def append(self, other):
        super().append(other)
        self.fiducial = np.hstack([self.fiducial, other.fiducial])
        self.lineNumber = np.hstack([self.lineNumber, other.lineNumber])
        self.data = np.vstack([self.data, other.data])
        self.predictedData = np.vstack([self.predictedData, other.predictedData])
        self.std = np.vstack([self.std, other.std])

    def dataMisfit(self, squared=False):
        """Compute the :math:`L_{2}` norm squared misfit between the observed and predicted data

        .. math::
            \| \mathbf{W}_{d} (\mathbf{d}^{obs}-\mathbf{d}^{pre})\|_{2}^{2},

        where :math:`\mathbf{W}_{d}` are the reciprocal data errors.

        Parameters
        ----------
        squared : bool
            Return the squared misfit.

        Returns
        -------
        out : np.float64
            The misfit value.

        """
        x = np.ma.MaskedArray((self.deltaD / self.std)**2.0, mask=~self.active)
        dataMisfit = StatArray.StatArray(np.sum(x, axis=1), "Data misfit")
        return dataMisfit if squared else np.sqrt(dataMisfit)

    def __getitem__(self, i):
        """ Define item getter for Data """
        i = np.unique(i)
        out = Data(nChannelsPerSystem=self.nChannelsPerSystem,
                   x=self.x[i], y=self.y[i], z=self.z[i], elevation=self.elevation[i],
                   data=self.data[i, :], std=self.std[i, :],
                   predictedData=self.predictedData[i, :],
                   channelNames=self.channelNames)
        return out

    def dataChannel(self, channel, system=None):
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
        out : geobipy.StatArray
            The data channel

        """

        if system is None:
            return StatArray.StatArray(self.data[:, channel], self._channelNames[channel], self.data.units)
        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            return StatArray.StatArray(self._data[:, self.systemOffset[system] + channel], self._channelNames[self.systemOffset[system] + channel], self.data.units)


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
        assert np.size(i) == 1, ValueError("i must be a single integer")
        assert 0 <= i <= self.nPoints, ValueError("Must have 0 <= i <= {}".format(self.nPoints))
        return DataPoint(self.nChannelsPerSystem, self.x[i], self.y[i], self.z[i], self.elevation[i], self.data[i, :], self.std[i, :], self.predictedData[i, :], channelNames=self.channelNames)


    def line(self, line):
        """ Get the data from the given line number """
        i = np.where(self.lineNumber == line)[0]
        assert (i.size > 0), 'Could not get line with number {}'.format(line)
        return self[i]


    def nPointsPerLine(self):
        """Gets the number of points in each line.

        Returns
        -------
        out : ints
            Number of points in each line

        """
        nPoints = np.zeros(np.unique(self.lineNumber).size)
        lines = np.unique(self.lineNumber)
        for i, line in enumerate(lines):
            nPoints[i] = np.sum(self.lineNumber == line)
        return nPoints


    def predictedDataChannel(self, channel, system=None):
        """Gets the predicted data in the specified channel

        Parameters
        ----------
        channel : int
            Index of the channel to return
            * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
        system : int, optional
            The system to obtain the channel from.

        Returns
        -------
        out : geobipy.StatArray
            The predicted data channel

        """

        if system is None:
            return StatArray.StatArray(self.predictedData[:, channel], "Predicted data {}".format(self.channelNames[channel]), self.predictedData.units)
        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            return StatArray.StatArray(self.predictedData[:, self.systemOffset[system] + channel], "Predicted data {}".format(self.channelNames[self.systemOffset[system] + channel]), self.predictedData.units)


    def stdChannel(self, channel, system=None):
        """Gets the uncertainty in the specified channel

        Parameters
        ----------
        channel : int
            Index of the channel to return
            * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
        system : int, optional
            The system to obtain the channel from.

        Returns
        -------
        out : geobipy.StatArray
            The uncertainty channel

        """

        if system is None:
            return StatArray.StatArray(self.std[:, channel], "Std {}".format(self.channelNames[channel]), self.std.units)
        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            return StatArray.StatArray(self.std[:, self.systemOffset[system] + channel], "Std {}".format(self.channelNames[self.systemOffset[system] + channel]), self.std.units)


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

        kwargs['values'] = self.dataChannel(channel)

        self.mapPlot(*args, **kwargs)

        cP.title(self.channelNames[channel])


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

        kwargs['c'] = self.predictedDataChannel(channel)

        self.mapPlot(*args, **kwargs)

        cP.title(self.channelNames[channel])


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

        self.mapPlot(*args, **kwargs)

        cP.title(self.channelNames[channel])


    def plot(self, xAxis='index', channels=None, values=None, system=None, **kwargs):
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
        ax = kwargs.pop('ax', plt.gca())

        if not values is None:
            legend = False
            ax = super().plot(values=values, xAxis=xAxis, label=cf.getName(values), **kwargs)

        else:
            if system is None:
                rTmp = range(self.nChannels) if channels is None else channels
            else:
                assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
                rTmp = self._systemOffset[system] + channels

            for i in rTmp:
                super().plot(values=self.data[:, i], xAxis=xAxis, label=self.channelNames[i], **kwargs)

        leg = None
        if legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
            leg.set_title(self._data.getNameUnits())

        return ax, leg


    def plotPredicted(self, xAxis='index', channels=None, system=None, **kwargs):
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

        noLegend = kwargs.pop('noLegend', False)

        if system is None:
            rTmp = range(self.nChannels) if channels is None else channels
        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            rTmp = self._systemOffset[system] + channels

        ax = plt.gca()

        for i in rTmp:
            super().plot(values=self._predictedData[:, i], xAxis=xAxis, label=self.channelNames[i], **kwargs)

        legend = None
        if not noLegend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
            legend.set_title(self._predictedData.getNameUnits())

        return ax, legend


    # def read(self, fname, columnIndex, nHeaders=0, nChannels=0):
    #     """ Read the specified columns from an ascii file
    #     cols[0,1,2,...] should be the indices of the x,y,z co-ordinates """
    #     nCols = len(columnIndex)
    #     #if any([cols < 0]): err.Emsg("Please specify the columns to read the first three indices should be xyz")
    #     # Get the number of points
    #     nLines = fIO.getNlines(fname, nHeaders)

    #     # Get the names of the headers
    #     names = fIO.getHeaderNames(fname, columnIndex)

    #     # Get the number of Data if none was specified
    #     if (nChannels == 0):
    #         nChannels = nCols - 3
    #     # Initialize the Data
    #     Data.__init__(self, nChannels, channelNames=names[3:])

    #     self.x.name = names[0]
    #     self.y.name = names[1]
    #     self.z.name = names[2]
    #     # Read each line assign the values to the class
    #     with open(fname) as f:
    #         fIO.skipLines(f, nHeaders)  # Skip header lines
    #         for j, line in enumerate(f):  # For each line in the file
    #             values = fIO.getRealNumbersfromLine(line, columnIndex)  # grab the requested entries
    #             # Assign values into object
    #             self.x[j] = values[0]
    #             self.y[j] = values[1]
    #             self.z[j] = values[2]
    #             self._data[j, ] = values[3:]

    def updateErrors(self, relativeErr, additiveErr, system=None):
        """Updates the data errors

        Updates the standard deviation of the data errors using the following model

        .. math::
            \sqrt{(\mathbf{\epsilon}_{rel} \mathbf{d}^{obs})^{2} + \mathbf{\epsilon}^{2}_{add}},

        where :math:`\mathbf{\epsilon}_{rel}` is the relative error, a percentage fraction and :math:`\mathbf{\epsilon}_{add}` is the additive error.

        Parameters
        ----------
        relativeErr : float
            A fraction percentage that is multiplied by the observed data.
        additiveErr : float
            An absolute value of additive error.

        Raises
        ------
        ValueError
            If any relative or additive errors are <= 0.0
        """

        relativeErr = np.atleast_1d(relativeErr)
        additiveErr = np.atleast_1d(additiveErr)
        # For each system assign error levels using the user inputs
        assert all(relativeErr > 0.0), ValueError("relativeErr must be > 0.0")
        assert all(additiveErr > 0.0), ValueError("additiveErr must be > 0.0")

        if system is None:
            if np.size(relativeErr) == 1:
                self._std[:, :] = np.sqrt((relativeErr * self._data[:, :])**2.0 + additiveErr**2.0)
            else:
                assert np.size(relativeErr) == self.nSystems, ValueError("Size of relative error must equal nSystems {}".format(self.nSystems))
                assert np.size(additiveErr) == self.nSystems, ValueError("Size of additive error must equal nSystems {}".format(self.nSystems))

                for i in range(self.nSystems):
                    iSys = self._systemIndices(system)
                    self._std[:, iSys] = np.sqrt((relativeErr[i] * self._data[:, iSys])**2.0 + additiveErr[i]**2.0)

        else:
            assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
            iSys = self._systemIndices(system)

            self._std[:, iSys] = np.sqrt((relativeErr * self._data[:, iSys])**2.0 + additiveErr**2.0)


    def toVTK(self, fileName, prop=['data', 'predicted', 'std'], system=None, format='binary'):
        """Save to a VTK file.

        Parameters
        ----------
        fileName : str
            Filename to save to.
        prop : str or list of str, optional
            List of the members to add to a VTK handle, either "data", "predicted", or "std".
        # channels : ints, optional
        #     Indices of the channels to plot.  All are plotted if None
        #     * If system is None, 0 <= channel < self.nChannels else 0 <= channel < self.nChannelsPerSystem[system]
        system : int, optional
            The system to obtain the channel from.
        format : str, optional
            "ascii" or "binary" format. Ascii is readable, binary is not but results in smaller files.

        """

        vtk = super().vtkStructure()

        self.addToVTK(vtk, prop, system=system)

        vtk.tofile(fileName, format=format)


    def createHdf(self, parent, myName, withPosterior=True, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = super().createHdf(parent, myName, withPosterior, fillvalue)

        grp.create_dataset('channels_per_system', data=self.nChannelsPerSystem)

        self.fiducial.createHdf(grp, 'fiducial', fillvalue=fillvalue)
        self.lineNumber.createHdf(grp, 'line_number', fillvalue=fillvalue)
        self.elevation.createHdf(grp, 'e', withPosterior=withPosterior, fillvalue=fillvalue)
        self.data.createHdf(grp, 'd', withPosterior=withPosterior, fillvalue=fillvalue)
        self.std.createHdf(grp, 's', withPosterior=withPosterior, fillvalue=fillvalue)
        self.predictedData.createHdf(grp, 'p', withPosterior=withPosterior, fillvalue=fillvalue)

        if not self.errorPosterior is None:
            for i, x in enumerate(self.errorPosterior):
                x.createHdf(grp, 'joint_error_posterior_{}'.format(i), fillvalue=fillvalue)

        self.relErr.createHdf(grp, 'relErr', withPosterior=withPosterior, fillvalue=fillvalue)
        self.addErr.createHdf(grp, 'addErr', withPosterior=withPosterior, fillvalue=fillvalue)

        return grp


    def writeHdf(self, parent, name, withPosterior=True):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        super().writeHdf(parent, name, withPosterior, index)

        grp = parent[name]

        self.fiducial.writeHdf(grp, 'fiducial', index=index)
        self.lineNumber.writeHdf(grp, 'line_number', index=index)
        self.elevation.writeHdf(grp, 'e',  withPosterior=withPosterior, index=index)

        self.data.writeHdf(grp, 'd',  withPosterior=withPosterior, index=index)
        self.std.writeHdf(grp, 's',  withPosterior=withPosterior, index=index)
        self.predictedData.writeHdf(grp, 'p',  withPosterior=withPosterior, index=index)

        if not self.errorPosterior is None:
            for i, x in enumerate(self.errorPosterior):
                x.writeHdf(grp, 'joint_error_posterior_{}'.format(i), index=index)

        self.relative_error.writeHdf(grp, 'relErr',  withPosterior=withPosterior, index=index)
        self.additive_error.writeHdf(grp, 'addErr',  withPosterior=withPosterior, index=index)


    def fromHdf(self, grp, **kwargs):
        """ Reads the object from a HDF group """

        super.fromHdf(grp)

        self.errorPosterior = None

        if 'fiducial' in grp:
            self.fiducial = StatArray.StatArray.fromHdf(grp['fiducial'])

        if 'line_number' in grp:
            self.lineNumber = StatArray.StatArray.fromHdf(grp['line_number'])

        if 'channels_per_system' in grp:
            self._nChannelsPerSystem = np.asarray(grp['channels_per_system'])

        self._data = StatArray.StatArray.fromHdf(grp['d'])
        self._std = StatArray.StatArray.fromHdf(grp['s'])
        self._predictedData = StatArray.StatArray.fromHdf(grp['p'])

        if 'joint_error_posterior_0' in grp:
            i = 0
            self.errorPosterior = []
            while 'joint_error_posterior_{}'.format(i) in grp:
                self.errorPosterior.append(Histogram3D.fromHdf(grp['joint_error_posterior_{}'.format(i)]))
                i += 1

        self._relative_error = StatArray.StatArray.fromHdf(grp['relErr'])
        self._additive_error = StatArray.StatArray.fromHdf(grp['addErr'])

        return self


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

        pc3d = PointCloud3D.Bcast(self, world, root=root)
        nPoints = myMPI.Bcast(self.nPoints, world, root=root)
        ncps = myMPI.Bcast(self.nChannelsPerSystem, world, root=root)
        x = self.x.Bcast(world)
        y = self.y.Bcast(world)
        z = self.z.Bcast(world)
        e = self.elevation.Bcast(world)
        d = self._data.Bcast(world)
        s = self._std.Bcast(world)
        p = self._predictedData.Bcast(world)

        return Data(nPoints, ncps, x=x, y=y, z=z,elevation=e, data=d, std=s, predictedData=p)


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
        ncps = myMPI.Bcast(self.nChannelsPerSystem, world, root=root)
        x = self.x.Scatterv(starts, chunks, world, root=root)
        y = self.y.Scatterv(starts, chunks, world, root=root)
        z = self.z.Scatterv(starts, chunks, world, root=root)
        e = self.elevation.Scatterv(starts, chunks, world, root=root)
        d = self._data.Scatterv(starts, chunks, world, root=root)
        s = self._std.Scatterv(starts, chunks, world, root=root)
        p = self._predictedData.Scatterv(starts, chunks, world, root=root)
        return Data(chunks[world.rank], ncps, x=x, y=y, z=z, elevation=e, data=d, std=s, predictedData=p)
