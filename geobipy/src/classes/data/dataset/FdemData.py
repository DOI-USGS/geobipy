""" @FdemData_Class
Module describing an EMData Set where channels are associated with an xyz co-ordinate
"""
from copy import deepcopy

from numpy import any, arange, asarray, atleast_1d, float64, full
from numpy import hstack, int32, isnan, nan, nanmin
from numpy import size, sqrt, squeeze, sum, unique
from numpy import zeros
from numpy import all as npall

from pandas import read_csv
from .Data import Data
from ..datapoint.FdemDataPoint import FdemDataPoint
from ....base import utilities as cF
from ....base import plotting as cP
from ...core.DataArray import DataArray
from ...statistics.StatArray import StatArray
from ...system.FdemSystem import FdemSystem
from ...system.CircularLoop import CircularLoop
from ....base import fileIO as fIO
#from ....base import Error as Err
from ....base import MPI as myMPI
import matplotlib.pyplot as plt

try:
    from pyvtk import Scalars
except:
    pass


class FdemData(Data):
    """Class extension to geobipy.Data defining a Fourier domain electro magnetic data set

    FdemData(nPoints, nFrequencies, system)

    Parameters
    ----------
    nPoints : int, optional
        Number of observations in the data set
    nFrequencies : int, optional
        Number of measurement frequencies
    system : str or geobipy.FdemSystem, optional
        * If str: Must be a file name from which to read FD system information.
        * If FdemSystem: A deepcopy is made.

    Returns
    -------
    out : FdemData
        Contains x, y, z, elevation, and data values for a frequency domain dataset.

    Notes
    -----
    FdemData.read() requires a data filename and a system class or system filename to be specified.
    The data file is structured using columns with the first line containing header information.
    The header should contain the following entries
    Line [ID or FID] [X or N or northing] [Y or E or easting] [Z or DTM or dem_elev] [Alt or Laser or bheight] [I Q] ... [I Q]
    Do not include brackets []
    [I Q] are the in-phase and quadrature values for each measurement frequency.

    If a system filename is given, it too is structured using columns with the first line containing header information
    Each subsequent row contains the information for each measurement frequency

    freq  tor  tmom  tx ty tz ror rmom  rx   ry rz
    378   z    1     0  0  0  z   1     7.93 0  0
    1776  z    1     0  0  0  z   1     7.91 0  0
    ...

    where tor and ror are the orientations of the transmitter/reciever loops [x or z].
    tmom and rmom are the moments of the loops.
    t/rx,y,z are the loop offsets from the observation locations in the data file.

    """

    single = FdemDataPoint

    __slots__ = ('_magnetic', '_powerline')

    def __init__(self, system=None, **kwargs):
        """Instantiate the FdemData class. """

        self.system = system

        kwargs['components'] = kwargs.get('components', self.components)
        kwargs['channels_per_system'] = kwargs.get('channels_per_system', 2*self.nFrequencies)
        # kwargs['components_per_channel'] = kwargs.get('components_per_channel', self.system[0].components)
        kwargs['units'] = kwargs.get('units', "ppm")

        # Data Class containing xyz and channel values
        super().__init__(**kwargs)

        self._powerline = DataArray(self._nPoints, "Powerline")
        self._magnetic = DataArray(self._nPoints, "Magnetic")

        # self.channel_names = kwargs.get('channel_names', None)

        self.powerline = kwargs.get('powerline', None)
        self.magnetic = kwargs.get('magnetic', None)


    @property
    def nFrequencies(self):
        return atleast_1d(self.system[0].nFrequencies)

    @property
    def channels_per_system(self):
        return 2 * self.nFrequencies

    @property
    def nSystems(self):
        return size(self.channels_per_system)

    @property
    def magnetic(self):
        if self._magnetic.size == 0:
            self._magnetic = DataArray(self._nPoints, "Magnetic", "nT")
        return self._magnetic

    @magnetic.setter
    def magnetic(self, values):
        if values is not None: # Set a default array
            if self._nPoints == 0: self.nPoints = size(values)
            if (self._magnetic.size != self._nPoints):
                self._magnetic = DataArray(values, "Magnetic", "nT")
                return

            self._magnetic[:] = values

    @property
    def powerline(self):
        if self._powerline.size == 0:
            self._powerline = DataArray(self._nPoints, "Powerline")
        return self._powerline

    @powerline.setter
    def powerline(self, values):
        if values is not None: # Set a default array
            if self._nPoints == 0: self.nPoints = size(values)
            if (self._powerline.size != self._nPoints):
                self._powerline = DataArray(values, "Powerline")
                return

            self._powerline[:] = values


    @Data.std.getter
    def std(self):
        if size(self._std, 0) == 0:
            self._std = DataArray((self.nPoints, self.nChannels), "Standard deviation", self.units)

        if self.relative_error.max() > 0.0:
            self._std[:, :] = sqrt((self.relative_error * self.data)**2 + (self.additive_error**2.0))

        return self._std

    @property
    def system(self):
        return self._system


    @system.setter
    def system(self, values):

        if values is None:
            self._system = None
            self.components = None
            return

        if isinstance(values, (str, FdemSystem)):
            values = [values]
        nSystems = len(values)
        # Make sure that list contains strings or TdemSystem classes
        assert all([isinstance(x, (str, FdemSystem)) for x in values]), TypeError("system must be str or list of either str or geobipy.FdemSystem")

        self._system = [None] * nSystems

        for i, s in enumerate(values):
            if isinstance(s, str):
                self._system[i] = FdemSystem.read(s)
            else:
                self._system[i] = s

        self.components = None

    @Data.channel_names.setter
    def channel_names(self, values):
        if values is None:
            self._channel_names = []
            for i in range(self.nSystems):
                # Set the channel names
                for ic in range(self.n_components):
                    for iFrequency in range(2*self.nFrequencies[i]):
                        self._channel_names.append('{} {}'.format(self.getMeasurementType(iFrequency, i), self.getFrequency(iFrequency, i)))
        else:
            assert all((isinstance(x, str) for x in values))
            assert len(values) == self.nChannels, Exception("Length of channel_names must equal total number of channels {}".format(self.nChannels))
            self._channel_names = values

    def check(self):
        if (nanmin(self.data) <= 0.0):
            print("Warning: Your data contains values that are <= 0.0")


    def fileInformation(self):
        """Description of the data file."""
        s =('\nThe data columns are read in according to the column names in the first line \n'
            'The header line should contain at least the following column names. Extra columns may exist, but will be ignored \n'
            'In this description, the column name or its alternatives are given followed by what the name represents \n'
            'Optional columns are also described \n'
            'Required columns'
            'line \n'
            '    Line number for the data point\n'
            'id or fid \n'
            '    Id number of the data point, these be unique\n'
            'x or northing or n \n'
            '    Northing co-ordinate of the data point, (m)\n'
            'y or easting or e \n'
            '    Easting co-ordinate of the data point, (m)\n'
            'z or alt or laser or bheight \n'
            '    Altitude of the transmitter coil above ground level (m)\n'
            'dtm or dem_elev or dem_np \n'
            '    Elevation of the ground at the data point (m)\n'
            'TxPitch \n'
            '    Pitch of the transmitter loop\n'
            'TxRoll \n'
            '    Roll of the transmitter loop\n'
            'TxYaw \n'
            '    Yaw of the transmitter loop\n'
            'RxPitch \n'
            '    Pitch of the receiver loop\n'
            'RxRoll \n'
            '    Roll of the receiver loop\n'
            'RxYaw \n'
            '    Yaw of the receiver loop\n'
            'Inphase[0] Quadrature[0] ... Inphase[nFrequencies] Quadrature[nFrequencies]  - with the number and square brackets\n'
            '    The measurements for each frequency specified in the accompanying system file. \n'
            'Optional columns\n'
            'InphaseErr[0] QuadratureErr[0] ... InphaseErr[nFrequencies] QuadratureErr[nFrequencies]\n'
            '    Estimates of standard deviation for each inphase and quadrature measurement.')
        return s


    @property
    def nActiveData(self):
        """Number of active data per data point.

        For each data point, counts the number of channels that are NOT nan.

        Returns
        -------
        out : int
            Number of active data

        """

        return sum(~isnan(self._data, 1))


    def append(self, other):

        super().append(other)

        self.powerline = self.powerline.append(other.powerline)
        self.magnetic = self.magnetic.append(other.magnetic)

        return self


    # def getChannel(self, channel):
    #     """Gets the data in the specified channel

    #     Parameters
    #     ----------
    #     channel : int
    #         A channel number less than 2 * number of frequencies

    #     Returns
    #     -------
    #     out : StatArray
    #         Contains the values of the requested channel

    #     """
    #     assert 0 <= channel < 2 * self.nFrequencies, ValueError('Requested channel must be 0 <= channel < {}'.format(2*self.nFrequencies))

    #     # if (channel < self.nFrequencies):
    #     #     tmp='InPhase - Frequency: {}'.format(self.system.frequencies[channel%self.nFrequencies])
    #     # else:
    #     #     tmp='Quadrature - Frequency: {}'.format(self.system.frequencies[channel%self.nFrequencies])

    #     tmp = StatArray(self._data[:, channel], self._channel_names[channel], self._data.units)

    #     return tmp


    def getMeasurementType(self, channel, system=0):
        """Returns the measurement type of the channel

        Parameters
        ----------
        channel : int
            Channel number
        system : int, optional
            System number

        Returns
        -------
        out : str
            Either "In-Phase " or "Quadrature "

        """
        return 'In_Phase' if channel < self.nFrequencies[system] else 'Quadrature'


    def getFrequency(self, channel, system=0):
        """Return the measurement frequency of the channel

        Parameters
        ----------
        channel : int
            Channel number
        system : int, optional
            System number

        Returns
        -------
        out : float
            The measurement frequency of the channel

        """
        return self.system[system].frequencies[channel%self.nFrequencies[system]]


    # def getLine(self, line):
    #     """Gets the data in the given line number

    #     Parameters
    #     ----------
    #     line : float
    #         A line number from the data file

    #     Returns
    #     -------
    #     out : geobipy.FdemData
    #         A data class containing only the data in the line

    #     """
    #     i = where(self.line == line)[0]
    #     assert (i.size > 0), 'Could not get line with number {}'.format(line)
    #     return self[i]


    def __getitem__(self, i):
        """Define item getter for Data

        Allows slicing into the data FdemData[i]

        """

        if not isinstance(i, slice):
            i = unique(i)

        return type(self)(self.system,
                       x=self.x[i],
                       y=self.y[i],
                       z=self.z[i],
                       elevation=self.elevation[i],
                       data=self.data[i, :],
                       std=self.std[i, :],
                       predicted_data=self.predicted_data[i, :],
                       line_number=self.line_number[i],
                       fiducial=self.fiducial[i],
                       powerline=self.powerline[i],
                       magnetic=self.magnetic[i])

    def datapoint(self, index=None, fiducial=None):
        """Get the ith data point from the data set

        Parameters
        ----------
        index : int, optional
            Index of the data point to get.
        fiducial : float, optional
            Fiducial of the data point to get.

        Returns
        -------
        out : geobipy.FdemDataPoint
            The data point.

        Raises
        ------
        Exception
            If neither an index or fiducial are given.

        """
        iNone = index is None
        fNone = fiducial is None

        assert not (iNone and fNone) ^ (not iNone and not fNone), Exception("Must specify either an index OR a fiducial.")

        if not fNone:
            index = self.fiducial.searchsorted(fiducial)

        return self.single(self.x[index],
                             self.y[index],
                             self.z[index],
                             self.elevation[index],
                             self.data[index, :],
                             self.std[index, :],
                             self.predicted_data[index, :],
                             system=self.system,
                             line_number=self.line_number[index],
                             fiducial=self.fiducial[index])


    # def mapChannel(self, channel, *args, system=0, **kwargs):
    #     """ Create a map of the specified data channel """

    #     assert channel < 2*self.nFrequencies[system], ValueError('Requested channel must be less than '+str(2*self.nFrequencies[system]))

    #     tmp = self.getChannel(system, channel)
    #     kwargs['c'] = tmp

    #     self.map(*args, **kwargs)

    #     cP.title(tmp.name)

    #     # Data.mapChannel(self, channel, *args, **kwargs)

    #     # cP.title(self._channel_names[channel])


    def plot_data(self, x='index', channels=None, **kwargs):
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
            The indices of the channels to plot. All are plotted if channels is None.
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

        ax, legend = super().plot_data(x=x, channels=channels, **kwargs)

        if kwargs.get('legend', True):
            legend.set_title('Frequency (Hz)')

        return ax, legend


    def plotLine(self, line, system=0, x='index', **kwargs):
        """ Plot the specified line """

        l = self.line(line)
        kwargs['log'] = kwargs.pop('log', None)

        x = self.axis(x)

        ax = plt.gca()
        ax1 = plt.subplot(211)

        for i in range(self.nFrequencies[system]):
            l._data[:, i].plot(x=x, label='{}'.format(self.getFrequency(i)), **kwargs)

        # ax.set_xticklabels([])
        # Shrink current axis by 20%
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
        leg.set_title('Frequency (Hz)')

        ylabel ='{}In_Phase ({})'.format(cF._logLabel(kwargs['log']), l._data.units)
        cP.ylabel(ylabel)

        ax = plt.subplot(212, sharex=ax1)
        for i in range(self.nFrequencies[system], 2*self.nFrequencies[system]):
            l._data[:, i].plot(x=x, label='{}'.format(self.getFrequency(i)), **kwargs)

        cP.suptitle("Line number {}".format(line))
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
        leg.set_title('Frequency (Hz)')

        # cP.xlabel(cF.getNameUnits(r))

        ylabel = '{}Quadrature ({})'.format(cF._logLabel(kwargs['log']), l._data.units)
        cP.ylabel(ylabel)

        return ax

    @classmethod
    def read_csv(cls, dataFilename, system):
        """Read in both the Fdem data and FDEM system files

        The data file is structured using columns with the first line containing header information.
        The header should contain the following entries
        Line [ID or FID] [X or N or northing] [Y or E or easting] [Z or DTM or dem_elev] [Alt or Laser or bheight] [I Q] ... [I Q]
        Do not include brackets []
        [I Q] are the in-phase and quadrature values for each measurement frequency.

        If a system filename is given, it too is structured using columns with the first line containing header information
        Each subsequent row contains the information for each measurement frequency

        freq  tor  tmom  tx ty tz ror rmom  rx   ry rz
        378   z    1     0  0  0  z   1     7.93 0  0
        1776  z    1     0  0  0  z   1     7.91 0  0
        ...

        where tor and ror are the orientations of the transmitter/reciever loops [x or z].
        tmom and rmom are the moments of the loops.
        t/rx,y,z are the loop offsets from the observation locations in the data file.

        """
        # Read in the EM System file

        # Initialize the EMData Class
        self = cls(system=system)

        # assert nDatafiles == self.nSystems, Exception("Number of data files must match number of system files.")
        nPoints, iC, iData, iStd, powerline, magnetic = FdemData._csv_channels(dataFilename)

        channels = iC + iData
        if len(iStd) > 0:
            channels += iStd

        # Read in the columns from the first data file
        try:
            df = read_csv(dataFilename, index_col=False, usecols=channels, skipinitialspace = True)
        except:
            df = read_csv(dataFilename, index_col=False, usecols=channels, sep=r'\s+', skipinitialspace = True)
        df = df.replace('NaN',nan)

        # Assign columns to variables
        self.line_number = df[iC[0]].values
        self.fiducial = df[iC[1]].values
        self.x = df[iC[2]].values
        self.y = df[iC[3]].values
        self.z = df[iC[4]].values
        self.elevation = df[iC[5]].values

        if not powerline is None:
            self.powerline = df[powerline].values
        else:
            self.powerline = None

        if not magnetic is None:
            self.magnetic = df[magnetic].values
        else:
            self.magnetic = None

        self.data = df[iData].values

        if len(iStd) > 0:
            self.std = df[iStd].values
        else:
            self.std = 0.1 * self.data

        self.check()

        return self


    # def _reconcile_channels(self, channels):

    #     for i, channel in enumerate(channels):
    #         channel = channel.lower()
    #         if(channel in ['line']):
    #             channels[i] = 'line'
    #         elif(channel in ['id', 'fid', 'fiducial']):
    #             channels[i] = 'fiducial'
    #         elif (channel in ['n', 'x','northing']):
    #             channels[i] = 'x'
    #         elif (channel in ['e', 'y', 'easting']):
    #             channels[i] = 'y'
    #         elif (channel in ['alt', 'laser', 'bheight', 'height']):
    #             channels[i] = 'height'
    #         elif(channel in ['z','dtm','dem_elev', 'dem', 'dem_np','topo', 'elev', 'elevation']):
    #             channels[i] = 'elevation'

    #     return channels

    def csv_channels(self, filename):
        self.nPoints, self._iC, self._iData, self._iStd, iP, iM = self._csv_channels(filename)

        self._channels = self._iC + self._iData
        if len(self._iStd) > 0:
            self._channels += self._iStd

        return self._channels

    @staticmethod
    def _csv_channels(data_filename):
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

        powerline = None
        magnetic = None

        nPoints, location_channels = Data._csv_channels(data_filename)

        # Get the column headers of the data file
        channels = fIO.get_column_name(data_filename)
        nChannels = len(channels)

        # To grab the EM data, skip the following header names. (More can be added to this)
        # Initialize a column identifier for x y z
        inPhase = []
        quadrature = []
        in_err = []
        quad_err = []

        import numpy as np

        for j, channel in enumerate(channels):
            cTmp = channel.lower()
            if cTmp in ['powerline']:
                powerline = 'powerline'

            elif cTmp in ['magnetic']:
                magnetic = 'magnetic'

            elif any([label in cTmp for label in ('cpi', 'i_', 'in_phase')]):

                if 'err' in cTmp:
                    in_err.append(channel)
                else:
                    inPhase.append(channel)

            elif any([label in cTmp for label in ('cpq', 'q_', 'quad')]):
                if 'err' in cTmp:
                    quad_err.append(channel)
                else:
                    quadrature.append(channel)

        data_channels = inPhase + quadrature

        error_channels = in_err + quad_err #if hasErrors else None

        return nPoints, location_channels, data_channels, error_channels, powerline, magnetic

    @classmethod
    def _initialize_sequential_reading(cls, data_filename, system_filename):
        """Special function to initialize a file for reading data points one at a time.

        Parameters
        ----------
        dataFileName : str
            Path to the data file
        systemFname : str
            Path to the system file

        """

        # Read in the EM System file
        self = cls(system_filename)
        self._data_filename = data_filename
        self._open_csv_files(data_filename)

        return self


    # def _open_csv_files(self, dataFilename):
    #     self._file = []
    #     for i, f in enumerate(dataFilename):
    #         try:
    #             df = read_csv(f, index_col=False, usecols=self._channels[i], chunksize=1, skipinitialspace = True)
    #         except:
    #             df = read_csv(f, index_col=False, usecols=self._channels[i], chunksize=1, delim_whitespace=True, skipinitialspace = True)

    #         self._file.append(df)

            # self._file.append(read_csv(f, usecols=self._indicesForFile[i], chunksize=1, delim_whitespace=True))


    # def _close_data_files(self):
    #     for f in self._file:
    #         if not f.closed:
    #             f.close()

    # def _read_line_fiducial(self, data_filename=None):
    #     try:
    #         df = read_csv(data_filename[0], index_col=False, usecols=self._iC[0][:2], skipinitialspace = True)
    #     except:
    #         df = read_csv(data_filename[0], index_col=False, usecols=self._iC[0][:2], delim_whitespace=True, skipinitialspace = True)

    #     df = df.replace('NaN',nan)
    #     return df[self._iC[0][0]].values, df[self._iC[0][1]].values

    def _read_record(self, record=None, mpi_enabled=False):
        """Reads a single data point from the data file.

        FdemData.__initLineByLineRead() must have already been run.

        """
        try:
            if record is None:
                df = self._file.get_chunk()
            else:
                df = self._file.get_chunk()
                if not mpi_enabled:
                    i = 1
                    while i <= record:
                        df = self._file.get_chunk()
                        i += 1

            df = df.replace('NaN',nan)
            endOfFile = False
        except:
            self._file.close()
            endOfFile = True

        if endOfFile:
            return None

        D = squeeze(df[self._iData].values)

        if len(self._iStd) > 0:
            S = squeeze(df[self._iStd].values)
        else:
            S = 0.1 * D

        return self.single(x=df[self._iC[2]].values,
                             y=df[self._iC[3]].values,
                             z=df[self._iC[4]].values,
                             elevation=df[self._iC[5]].values,
                             data=D, std=S, system=self.system,
                             line_number=df[self._iC[0]].values,
                             fiducial=df[self._iC[1]].values)


    def readAarhusFile(self, dataFilename):
        """Read in frequency domain data from an Aarhus workbench file.

        Parameters
        ----------
        dataFilename : str
            The data file.

        """

        # Read in the EM System file
        if (isinstance(dataFilename, str)):
            dataFilename = [dataFilename]

        nDatafiles = len(dataFilename)
        nSystems = nDatafiles

        system, nPoints, iC, iD, nHeaderLines, iP, iM = self._readAarhusHeader(dataFilename[0])

        # nPoints, iC, iD, iS = self.__readColumnIndices(dataFilename, self.system)

        # Get all readable column indices for the first file.
        tmp = [iC]
        tmp.append(iD)
        # if not iS[0] is None:
        #     tmp.append(iS[0])

        if not iP is None:
            tmp.append(iP)
        if not iM is None:
            tmp.append(iM)

        indicesForFile = hstack(tmp)


        # Initialize the EMData Class
        FdemData.__init__(self, nPoints, systems=system)

        values = fIO.read_columns(dataFilename[0], indicesForFile, nHeaderLines, nPoints)

        # Assign columns to variables
        self._line_number[:] = values[:, 0]
        self._fiducial[:] = values[:, 1]
        self.x[:] = values[:, 2]
        self.y[:] = values[:, 3]
        self.z[:] = values[:, 4]
        self.elevation[:] = values[:, 5]

        self._data[:, :] = values[:, 6:6+2*self.nFrequencies[0]]

        if not iP is None:
            self.powerline = DataArray(values[:, 6+2*self.nFrequencies[0]])
        if not iM is None:
            iM = 6+2*self.nFrequencies[0]
            if not iP is None:
                iM += 1
            self.magnetic = DataArray(values[:, iM])



    def _readAarhusHeader(self, dataFilename):
        """Read in the header information from an Aarhus workbench file.

        Parameters
        ----------
        dataFilename : str
            The data file.

        """

        with open(dataFilename, 'r') as f:
            go = True
            nHeaderLines = 0
            while go:
                line = f.readline().strip('/').lower()
                nHeaderLines += 1
                if "number of channels" in line:
                    line = f.readline().strip('/')
                    nHeaderLines += 1
                    nFrequencies = int32(line)
                if "frequencies" in line:
                    line = f.readline().strip('/').split()
                    nHeaderLines += 1
                    frequencies = asarray([float64(x) for x in line])
                if "coil configurations" in line:
                    pairs = f.readline().strip('/').split(')  (')
                    nHeaderLines += 1
                    transmitterLoops = DataArray(nFrequencies, dtype=CircularLoop)
                    receiverLoops = DataArray(nFrequencies, dtype=CircularLoop)
                    for i, pair in enumerate(pairs):
                        tmp = pair.split(',')
                        if 'VMD' in tmp[0]:
                            transmitterLoops[i] = CircularLoop()
                        else:
                            transmitterLoops[i] = CircularLoop(orient='x', moment=-1)

                        if 'VMD' in tmp[1]:
                            receiverLoops[i] = CircularLoop()
                        else:
                            receiverLoops[i] = CircularLoop(orient='x', moment=-1)

                if "coil separations" in line:
                    line = f.readline().strip('/').split()
                    nHeaderLines += 1
                    loopSeparation = asarray([float64(x) for x in line])
                    go = False
                    channels = f.readline().strip('/')
                    nHeaderLines += 1


        system = FdemSystem(nFrequencies, frequencies, transmitterLoops, receiverLoops, loopSeparation)


        _powerline = None
        _magnetic = None
        _columnIndex = zeros(6, dtype=int32)
        for j, channel in enumerate(channels.split()):
            channel = channel.lower()
            if(channel in ['line_no']):
                _columnIndex[0] = j
            elif(channel in ['fiducial']):
                _columnIndex[1] = j
            elif (channel in ['utmx']):
                _columnIndex[2] = j
            elif (channel in ['utmy']):
                _columnIndex[3] = j
            elif (channel in ['sensor_height']):
                _columnIndex[4] = j
            elif(channel in ['elevation']):
                _columnIndex[5] = j
            elif(channel == 'imag1'):
                tmp = arange(j, j + 2 * nFrequencies)
                _dataIndices = hstack((tmp[1::2], tmp[::2]))
            elif channel == 'powerline':
                _powerline = j
            elif channel == 'magnetic':
                _magnetic = j


        nPoints = self._readNpoints([dataFilename]) - nHeaderLines + 1

        return system, nPoints, _columnIndex, _dataIndices, nHeaderLines, _powerline, _magnetic

    def createHdf(self, parent, myName, withPosterior=True, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = super().createHdf(parent, myName, withPosterior, fillvalue)
        self.system[0].toHdf(grp, 'sys')
        return grp

    @classmethod
    def fromHdf(cls, grp, **kwargs):
        """ Reads the object from a HDF group """

        if kwargs.get('index') is not None:
            return cls.single.fromHdf(grp, **kwargs)

        system = FdemSystem.fromHdf(grp['sys'])
        return super(FdemData, cls).fromHdf(grp, system=system)

    def Bcast(self, world, root=0):
        """Broadcast the FdemData using MPI

        Parameters
        ----------
        world : mpi4py.MPI.COMM_WORLD
            MPI communicator

        Returns
        -------
        out : geobipy.FdemData
            A copy of the data on each core

        Examples
        --------
        >>> from mpi4py import MPI
        >>> from geobipy import FdemData

        >>> world = MPI.COMM_WORLD

        >>> rank = world.rank

        >>> if (rank == 0): # Only the master reads in the data
        >>>     D = FdemData()
        >>>     D.read(dataFile, systemFile)
        >>> else:
        >>>     D = FdemData() # Must instantiate an empty object to Bcast

        >>> D2 = D.Bcast(world)

        """

        out = super().Bcast(world, root=root)

        n_frequencies = myMPI.Bcast(self.nFrequencies, world, root=root)
        n_systems = myMPI.Bcast(self.nSystems, world, root=root)

        if world.rank == root:
            systems_null = self.system
        else:
            systems_null = [FdemSystem(None, None, None, n_frequencies = n_frequencies) for i in range(n_systems)]

        out.systems = [sys.Bcast(world, root=root) for sys in systems_null]

        return out


    def Scatterv(self, starts, chunks, world, root=0):
        """Distributes the FdemData between all cores using MPI

        Parameters
        ----------
        starts : array of ints
            1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
        chunks : array of ints
            1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
        world : mpi4py.MPI.COMM_WORLD
            The MPI communicator

        Returns
        -------
        out : geobipy.FdemData
            The data distributed amongst cores

        Examples
        --------
        >>> from mpi4py import MPI
        >>> from geobipy import FdemData
        >>> import numpy as np

        >>> world = MPI.COMM_WORLD

        >>> rank = world.rank

        >>> if (rank == 0): # Only the master reads in the data
        >>>     D = FdemData()
        >>>     D.read(dataFile, systemFile)
        >>> else:
        >>>     D = FdemData() # Must instantiate an empty object to Bcast

        >>> # In this example, assume there are 10 data and 4 cores
        >>> start = asarray([0, 2, 4, 6])
        >>> chunks = asarray([2, 2, 2, 4])

        >>> D2 = D.Scatterv(start, chunks, world)

        """

        out = super().Scatterv(starts, chunks, world, root=root)

        # npoints = myMPI.Bcast(self.nPoints, world, root=root)
        n_frequencies = myMPI.Bcast(self.nFrequencies, world, root=root)
        n_systems = myMPI.Bcast(self.nSystems, world, root=root)

        if world.rank == root:
            systems_null = self.system
        else:
            systems_null = [FdemSystem(None, None, None, n_frequencies = n_frequencies) for i in range(n_systems)]

        out.system = [sys.Bcast(world, root=root) for sys in systems_null]

        return out




    def write(self, fileNames, std=False, predicted_data=False):

        if isinstance(fileNames, str):
            fileNames = [fileNames]

        assert len(fileNames) == self.nSystems, ValueError("fileNames must have length equal to the number of systems {}".format(self.nSystems))

        import pandas as pd

        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)

        # for i, sys in enumerate(self.system):
        #     # Create the header
        #     header = "Line Fid Easting Northing Elevation Height "

        #     for x in sys.frequencies:
        #         header += "I_{0} Q_{0} ".format(x)

        #     if not self.powerline is None:
        #         header += 'Powerline '
        #     if not self.magnetic is None:
        #         header += 'Magnetic'

        #     d = empty(2*sys.nFrequencies)

        #     if std:
        #         for x in sys.frequencies:
        #             header += "I_{0}_Err Q_{0}_Err ".format(x)
        #         s = empty(2*sys.nFrequencies)

        #     with open(fileNames[i], 'w') as f:
        #         f.write(header+"\n")
        #         with printoptions(formatter={'float': '{: 0.15g}'.format}, suppress=True):
        #             for j in range(self.nPoints):

        #                 x = asarray([self.line_number[j], self.fiducial[j], self.x[j], self.y[j], self.elevation[j], self.z[j]])

        #                 if predicted_data:
        #                     d[0::2] = self.predicted_data[j, :sys.nFrequencies]
        #                     d[1::2] = self.predicted_data[j, sys.nFrequencies:]
        #                 else:
        #                     d[0::2] = self.data[j, :sys.nFrequencies]
        #                     d[1::2] = self.data[j, sys.nFrequencies:]

        #                 if std:
        #                     s[0::2] = self.std[j, :sys.nFrequencies]
        #                     s[1::2] = self.std[j, sys.nFrequencies:]
        #                     x = hstack([x, d, s])
        #                 else:
        #                     x = hstack([x, d])

        #                 if not self.powerline is None:
        #                     x = hstack([x, self.powerline[j]])
        #                 if not self.magnetic is None:
        #                     x = hstack([x, self.magnetic[j]])

        #                 y = ""
        #                 for a in x:
        #                     y += "{} ".format(a)

        #                 f.write(y+"\n")

    def create_synthetic_data(self, model, prng):

        ds = FdemData(system=self.system)

        ds.x = model.x.centres
        ds.y[:] = 0.0
        ds.z = full(model.x.nCells, fill_value=30.0)
        ds.elevation = zeros(model.x.nCells)
        ds.relative_error = full((model.x.nCells, 1), fill_value = 0.05)
        ds.additive_error = full((model.x.nCells, 1), fill_value = 5)

        dp = ds.datapoint(0)

        for k in range(model.x.nCells):
            mod = model[k]
            dp.forward(mod)
            ds.data[k, :] = dp.predicted_data

        ds_noisy = deepcopy(ds)

        ds_noisy._data += prng.normal(scale=ds.std, size=ds.data.shape)

        return ds, ds_noisy