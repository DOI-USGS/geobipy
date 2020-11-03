""" @FdemData_Class
Module describing an EMData Set where channels are associated with an xyz co-ordinate
"""
from .Data import Data
from ..datapoint.FdemDataPoint import FdemDataPoint
from ....base import customFunctions as cF
from ....base import customPlots as cP
from ....classes.core import StatArray
from ...system.FdemSystem import FdemSystem
from ...system.CircularLoop import CircularLoop
import numpy as np
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

    def __init__(self, systems=None, **kwargs):
        """Instantiate the FdemData class. """

        if (systems is None):
            return

        self.system = systems

        # Data Class containing xyz and channel values
        Data.__init__(self, nChannelsPerSystem=2*self.nFrequencies, units="ppm", **kwargs)

        # Assign data names
        self._data.name = 'Fdem Data'

        self.channelNames = kwargs.get('channel_names', None)

        self.powerline = kwargs.get('powerline', None)
        self.magnetic = kwargs.get('magnetic', None)


    @property
    def nFrequencies(self):
        return np.asarray([s.nFrequencies for s in self.system])


    @property
    def nSystems(self):
        return np.size(self.system)


    @property
    def magnetic(self):
        return self._magnetic


    @magnetic.setter
    def magnetic(self, values):
        if (values is None):
            self._magnetic = StatArray.StatArray(self.nPoints, "Magnetic", "nT")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("magnetic must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._magnetic = values.deepcopy()
            else:
                self._magnetic = StatArray.StatArray(values, "Magnetic", "nT")


    @property
    def powerline(self):
        return self._powerline


    @powerline.setter
    def powerline(self, values):
        if (values is None):
            self._powerline = StatArray.StatArray(self.nPoints, "Powerline")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("powerline must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._powerline = values.deepcopy()
            else:
                self._powerline = StatArray.StatArray(values, "Powerline")


    @property
    def system(self):
        return self._system


    @system.setter
    def system(self, values):

        if isinstance(values, (str, FdemSystem)):
            values = [values]
        nSystems = len(values)
        # Make sure that list contains strings or TdemSystem classes
        assert all([isinstance(x, (str, FdemSystem)) for x in values]), TypeError("system must be str or list of either str or geobipy.FdemSystem")

        self._system = np.ndarray(nSystems, dtype=FdemSystem)

        for i, s in enumerate(values):
            if isinstance(s, str):
                self._system[i] = FdemSystem().read(s)
            else:
                self._system[i] = s


    @property
    def channelNames(self):
        return self._channelNames


    @channelNames.setter
    def channelNames(self, values):
        if values is None:
            self._channelNames = []
            for i in range(self.nSystems):
                # Set the channel names
                if not self.system[i] is None:
                    for iFrequency in range(2*self.nFrequencies[i]):
                        self._channelNames.append('{} {} (Hz)'.format(self.getMeasurementType(iFrequency, i), self.getFrequency(iFrequency, i)))
        else:
            assert all((isinstance(x, str) for x in values))
            assert len(values) == self.nChannels, Exception("Length of channelNames must equal total number of channels {}".format(self.nChannels))
            self._channelNames = values


    def check(self):
        if (np.nanmin(self.data) <= 0.0):
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

        return np.sum(~np.isnan(self._data, 1))


    def append(self, other):

        super().append(other)

        self.powerline = np.hstack([self.powerline, other.powerline])
        self.magnetic = np.hstack([self.magnetic, other.magnetic])


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

    #     tmp = StatArray(self._data[:, channel], self._channelNames[channel], self._data.units)

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
        return 'In-Phase' if channel < self.nFrequencies[system] else 'Quadrature'


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
    #     i = np.where(self.line == line)[0]
    #     assert (i.size > 0), 'Could not get line with number {}'.format(line)
    #     return self[i]


    def __getitem__(self, i):
        """Define item getter for Data

        Allows slicing into the data FdemData[i]

        """

        if not isinstance(i, slice):
            i = np.unique(i)
        return FdemData(self.system,
                       x=self.x[i],
                       y=self.y[i],
                       z=self.z[i],
                       elevation=self.elevation[i],
                       data=self.data[i, :],
                       std=self.std[i, :],
                       predictedData=self.predictedData[i, :],
                       lineNumber=self.lineNumber[i],
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

        return FdemDataPoint(self.x[index], self.y[index], self.z[index], self.elevation[index], self._data[index, :], self._std[index, :], system=self.system, lineNumber=self.lineNumber[index], fiducial=self.fiducial[index])


    # def mapChannel(self, channel, *args, system=0, **kwargs):
    #     """ Create a map of the specified data channel """

    #     assert channel < 2*self.nFrequencies[system], ValueError('Requested channel must be less than '+str(2*self.nFrequencies[system]))

    #     tmp = self.getChannel(system, channel)
    #     kwargs['c'] = tmp

    #     self.mapPlot(*args, **kwargs)

    #     cP.title(tmp.name)

    #     # Data.mapChannel(self, channel, *args, **kwargs)

    #     # cP.title(self._channelNames[channel])


    def plot(self, xAxis='index', channels=None, values=None, **kwargs):
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
        geobipy.customPlots.plot : For additional keyword arguments

        """

        kwargs['legend'] = kwargs.pop('legend', True)
        ax, legend = super().plot(xAxis, channels=channels, values=values, **kwargs)

        if not legend is None:
            legend.set_title('Frequency (Hz)')

        return ax, legend


    def plotLine(self, line, system=0, xAxis='index', **kwargs):
        """ Plot the specified line """

        l = self.line(line)
        kwargs['log'] = kwargs.pop('log', None)

        x = self.getXAxis(xAxis)

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

        ylabel ='{}In-Phase ({})'.format(cF._logLabel(kwargs['log']), l._data.getUnits())
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

        ylabel = '{}Quadrature ({})'.format(cF._logLabel(kwargs['log']), l._data.getUnits())
        cP.ylabel(ylabel)

        return ax


    def read(self, dataFilename, systemFilename):
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
        if (isinstance(dataFilename, str)):
            dataFilename = [dataFilename]
        if (isinstance(systemFilename, str)):
            systemFilename = [systemFilename]

        nDatafiles = len(dataFilename)
        nSystems = len(systemFilename)

        assert nDatafiles == nSystems, Exception("Number of data files must match number of system files.")

        self.system = systemFilename
        # self.readSystemFile(systemFilename)

        nPoints, iC, iD, iS, powerline, magnetic = self.__readColumnIndices(dataFilename, self.system)

        nBase = np.size(iC[0])

        # Get all readable column indices for the first file.
        tmp = [iC[0]]
        tmp.append(iD[0])
        if not iS[0] is None:
            tmp.append(iS[0])
        indicesForFile = np.hstack(tmp)


        # Initialize the EMData Class
        FdemData.__init__(self, systems=self.system)

        # Read in the columns from the first data file
        values = fIO.read_columns(dataFilename[0], indicesForFile, 1, nPoints)

        # Assign columns to variables
        self.lineNumber = values[:, 0]
        self.fiducial = values[:, 1]
        self.x = values[:, 2]
        self.y = values[:, 3]
        self.z = values[:, 4]
        self.elevation = values[:, 5]

        if not powerline is None:
            self.powerline = values[:, powerline]
        else:
            self.powerline = None

        if not magnetic is None:
            self.magnetic = values[:, magnetic]
        else:
            self.magnetic = None

        # EM data columns are in the following order
        # I1 Q1 I2 Q2 .... IN QN ErrI1 ErrQ1 ... ErrIN ErrQN
        # Reshuffle to the following
        # I1 I2 ... IN Q1 Q2 ... QN and
        # ErrI1 ErrI2 ... ErrIN ErrQ1 ErrQ2 ... ErrQN

        self.data = values[:, nBase:nBase + (2 * self.nFrequencies[0])]
        if (iS[0]):
            self.std = values[:, nBase + (2 * self.nFrequencies[0]):]
        else:
            self.std = 0.1 * self.data

        # # Read in the data for the other systems.  Only read in the data and, if available, the errors
        # for i in range(1, self.nSystems):
        #     # Get all readable column indices for the file.
        #     tmp = [iD[i]]
        #     if not iS[i] is None:
        #         tmp.append(iS[i])
        #     indicesForFile = np.hstack(tmp)

        #     # Read the columns
        #     values = fIO.read_columns(dataFilename[i], indicesForFile[i], 1, nPoints)
        #     # Assign the data
        #     iSys = self._systemIndices(i)
        #     self.data[:, iSys] = values[:, nBase:nBase + 2 * self.nFrequencies[i]]
        #     if (iS[i]):
        #         self.std[:, iSys] = values[:, nBase + 2 * self.nFrequencies[i]:]
        #     else:
        #         self.std[:, iSys] = 0.1 * self.data[:, iSys]

        self.check()

        return self


    # def readSystemFile(self, systemFilename):
    #     """ Reads in the system handler using the system file name """

    #     if isinstance(systemFilename, str):
    #         systemFilename = [systemFilename]

    #     nSys = len(systemFilename)
    #     self.system = np.ndarray(nSys, dtype=FdemSystem)

    #     for i in range(nSys):
    #         self.system[i] = FdemSystem()
    #         self.system[i].read(systemFilename[i])

    #     self.nSystems = nSys
    #     self.nChannelsPerSystem = np.asarray([np.int32(2*x.nFrequencies) for x in self.system])
    #     self._systemOffset = np.append(0, np.cumsum(self.nChannelsPerSystem))


    # Section contains routines for opening a data file, and reading data points one at a time
    # when requested.  These are used for a parallel implementation so that data points can be read
    # by a master rank and sent individually to worker ranks.  Removes the need to read the entire
    # dataset on all cores and minimizes RAM requirements.
    def __readColumnIndices(self, dataFilename, system):
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
        dataIndices = []
        errIndices = []

        if isinstance(dataFilename, str):
            dataFilename = [dataFilename]
        if isinstance(system, FdemSystem):
            system = [system]

        assert all(isinstance(s, FdemSystem) for s in system), TypeError("system must contain geobipy.FdemSystem classes.")

        nPoints = self._readNpoints(dataFilename)

        for k, f in enumerate(dataFilename):

            # Get the column headers of the data file
            channels = fIO.getHeaderNames(f)
            channels = [channel.lower() for channel in channels]
            nChannels = len(channels)

            # Check for each aspect of the data file and the number of columns
            nCoordinates = 0

            powerline = None
            magnetic = None
            for channel in channels:
                channel = channel.lower()
                if(channel in ['line']):
                    nCoordinates += 1
                elif(channel in ['id', 'fid']):
                    nCoordinates += 1
                elif (channel in ['n', 'x','northing']):
                    nCoordinates += 1
                elif (channel in ['e', 'y', 'easting']):
                    nCoordinates += 1
                elif (channel in ['alt', 'laser', 'bheight', 'height']):
                    nCoordinates += 1
                elif(channel in ['z','dtm','dem_elev','dem_np','topo', 'elev', 'elevation']):
                    nCoordinates += 1
                elif(channel in ['powerline']):
                    nCoordinates += 1
                elif(channel in ['magnetic']):
                    nCoordinates += 1

            assert nCoordinates >= 6, Exception("Data file must contain columns for easting, northing, height, elevation, line, and fid. \n {}".format(self.fileInformation()))

            nData = nChannels - nCoordinates

            if nData > 2*system[k].nFrequencies:
                _hasErrors = True
                assert nData == 4*system[k].nFrequencies, Exception("Data file must have {0} data channels and {0} uncertainty channels each for in-phase and quadrature data.".format(system[k].nFrequencies))

            else:
                _hasErrors = False
                assert nData == 2*system[k].nFrequencies, Exception("Data file must have {} data channels each for in-phase and quadrature data.".format(system[k].nFrequencies))


            # To grab the EM data, skip the following header names. (More can be added to this)
            # Initialize a column identifier for x y z
            _columnIndex = np.zeros(nCoordinates, dtype=np.int32)
            inPhase = []
            quadrature = []
            for j, channel in enumerate(channels):
                if(channel in ['line']):
                    _columnIndex[0] = j
                elif(channel in ['id', 'fid']):
                    _columnIndex[1] = j
                elif (channel in ['e', 'x', 'easting']):
                    _columnIndex[2] = j
                elif (channel in ['n', 'y','northing']):
                    _columnIndex[3] = j
                elif (channel in ['alt', 'laser', 'bheight', 'height']):
                    _columnIndex[4] = j
                elif(channel in ['z','dtm','dem_elev','dem_np','topo', 'elev', 'elevation']):
                    _columnIndex[5] = j
                elif channel in ['powerline']:
                    _columnIndex[6] = j
                    powerline = 6
                elif channel in ['magnetic']:
                    if nCoordinates == 6:
                        _columnIndex[6] = j
                        magnetic = 6
                    else:
                        _columnIndex[7] = j
                        magnetic = 7
                elif 'i_' in channel:
                    inPhase.append(j)
                elif 'q_' in channel:
                    quadrature.append(j)

            _dataIndices = np.hstack([inPhase, quadrature])

            _errIndices = np.hstack([inPhase + 2*system[k].nFrequencies, quadrature + 2*system[k].nFrequencies]) if _hasErrors else None

            indices.append(_columnIndex)
            dataIndices.append(_dataIndices)
            errIndices.append(_errIndices)

        return nPoints, indices, dataIndices, errIndices, powerline, magnetic


    def _initLineByLineRead(self, dataFilename, systemFilename):
        """Special function to initialize a file for reading data points one at a time.

        Parameters
        ----------
        dataFileName : str
            Path to the data file
        systemFname : str
            Path to the system file

        """

        # Read in the EM System file
        self.system = systemFilename
        # self.readSystemFile(systemFilename)
        self._nPoints, self._iC, self._iD, self._iS, iP, iM = self.__readColumnIndices(dataFilename, self.system)

        if isinstance(dataFilename, str):
            dataFilename = [dataFilename]

        self._openDatafiles(dataFilename)

        # Get all readable column indices for the first file.
        self._indicesForFile = []
        for i in range(self.nSystems):
            tmp = [self._iC[i]]
            tmp.append(self._iD[i])
            if not self._iS[i] is None:
                tmp.append(self._iS[i])
            self._indicesForFile.append(np.hstack(tmp))

        # Remap the indices for the different components of the file to make reading easier.
        for i in range(self.nSystems):
            offset = self._iC[i].size
            self._iC[i] = np.arange(offset)
            nTmp = self._iD[i].size
            self._iD[i] = np.arange(nTmp) + offset
            offset += nTmp
            if not self._iS[i] is None:
                nTmp = self._iS[i].size
                self._iS[i] = np.arange(nTmp) + offset


    def _openDatafiles(self, dataFilename):
        self._file = []
        for i, f in enumerate(dataFilename):
            self._file.append(open(f, 'r'))
            fIO.skipLines(self._file[i], nLines=1)


    def _closeDatafiles(self):
        for f in self._file:
            if not f.closed:
                f.close()


    def _readSingleDatapoint(self):
        """Reads a single data point from the data file.

        FdemData.__initLineByLineRead() must have already been run.

        """
        if self._file[0].closed:
            return None

        endOfFile = False
        values = []
        for i in range(self.nSystems):
            line = self._file[i].readline()
            try:
                values.append(fIO.getRealNumbersfromLine(line, self._indicesForFile[i]))
            except:
                self._file[i].close()
                endOfFile = True

        if endOfFile:
            return None

        D = np.empty(self.nChannels)
        S = np.empty(self.nChannels)
        for j in range(self.nSystems):
            iSys = self._systemIndices(j)
            D[iSys] = values[j][6:6 + 2*self.nFrequencies[j]]
            if self._iS[j] is None:
                S[iSys] = 0.1 * D[iSys]
            else:
                S[iSys] = values[j][6 + 2*self.nFrequencies[j] : 6 + 4*self.nFrequencies[j]]

        values = values[0]

        return FdemDataPoint(x=values[2], y=values[3], z=values[4], elevation=values[5], data=D, std=S, system=self.system, lineNumber=values[0], fiducial=values[1])


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

        indicesForFile = np.hstack(tmp)


        # Initialize the EMData Class
        FdemData.__init__(self, nPoints, systems=system)

        values = fIO.read_columns(dataFilename[0], indicesForFile, nHeaderLines, nPoints)

        # Assign columns to variables
        self._lineNumber[:] = values[:, 0]
        self._fiducial[:] = values[:, 1]
        self.x[:] = values[:, 2]
        self.y[:] = values[:, 3]
        self.z[:] = values[:, 4]
        self.elevation[:] = values[:, 5]

        self._data[:, :] = values[:, 6:6+2*self.nFrequencies[0]]

        if not iP is None:
            self.powerline = StatArray.StatArray(values[:, 6+2*self.nFrequencies[0]])
        if not iM is None:
            iM = 6+2*self.nFrequencies[0]
            if not iP is None:
                iM += 1
            self.magnetic = StatArray.StatArray(values[:, iM])



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
                    nFrequencies = np.int(line)
                if "frequencies" in line:
                    line = f.readline().strip('/').split()
                    nHeaderLines += 1
                    frequencies = np.asarray([np.float64(x) for x in line])
                if "coil configurations" in line:
                    pairs = f.readline().strip('/').split(')  (')
                    nHeaderLines += 1
                    transmitterLoops = StatArray.StatArray(nFrequencies, dtype=CircularLoop)
                    receiverLoops = StatArray.StatArray(nFrequencies, dtype=CircularLoop)
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
                    loopSeparation = np.asarray([np.float64(x) for x in line])
                    go = False
                    channels = f.readline().strip('/')
                    nHeaderLines += 1


        system = FdemSystem(nFrequencies, frequencies, transmitterLoops, receiverLoops, loopSeparation)


        _powerline = None
        _magnetic = None
        _columnIndex = np.zeros(6, dtype=np.int32)
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
                tmp = np.arange(j, j + 2 * nFrequencies)
                _dataIndices = np.hstack((tmp[1::2], tmp[::2]))
            elif channel == 'powerline':
                _powerline = j
            elif channel == 'magnetic':
                _magnetic = j


        nPoints = self._readNpoints([dataFilename]) - nHeaderLines + 1

        return system, nPoints, _columnIndex, _dataIndices, nHeaderLines, _powerline, _magnetic


    def fromHdf(self, grp, **kwargs):
        """ Reads the object from a HDF group """

        s = grp['d/data'].shape

        item = grp.get('sys')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        system = obj.fromHdf(item)

        tmp = FdemData(nPoints=s[0], nFrequencies=np.int(0.5*s[1]), systems=system)

        item = grp.get('x')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._x = obj.fromHdf(item)

        item = grp.get('y')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._y = obj.fromHdf(item)

        item = grp.get('z')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._z = obj.fromHdf(item)

        item = grp.get('e')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._elevation = obj.fromHdf(item)

        item = grp.get('d')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._data = obj.fromHdf(item)

        item = grp.get('s')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._std = obj.fromHdf(item)

        item = grp.get('p')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._predictedData = obj.fromHdf(item)

        item = grp.get('relErr')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._relErr = obj.fromHdf(item)

        item = grp.get('addErr')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._addErr = obj.fromHdf(item)

        item = grp.get('calibration')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp.calibration = obj.fromHdf(item)

        # tmp.getActiveChannels()
        return tmp

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

        npoints = myMPI.Bcast(self.nPoints, world, root=root)
        nf = myMPI.Bcast(self.nFrequencies, world, root=root)
        ns = myMPI.Bcast(self.nSystems, world, root=root)
        if world.rank != root:
            sys = np.ndarray(ns, dtype=FdemSystem)
            for i in range(ns):
                sys[i] = FdemSystem()
        else:
            sys = self.system

        sysTmp = []
        for i in range(ns):
            sysTmp.append(sys[i].Bcast(world, root=root))

        out = FdemData(npoints, nf, sysTmp)
        out._x = self.x.Bcast(world, root=root)
        out._y = self.y.Bcast(world, root=root)
        out._z = self.z.Bcast(world, root=root)
        out._elevation = self.elevation.Bcast(world, root=root)
        out._data = self._data.Bcast(world, root=root)
        out._std = self._std.Bcast(world, root=root)
        out._predictedData = self._predictedData.Bcast(world, root=root)
        out._fiducial = self.fiducial.Bcast(world, root=root)
        out._lineNumber = self.lineNumber.Bcast(world, root=root)
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
        >>> start = np.asarray([0, 2, 4, 6])
        >>> chunks = np.asarray([2, 2, 2, 4])

        >>> D2 = D.Scatterv(start, chunks, world)

        """

        nf = myMPI.Bcast(self.nFrequencies, world, root=root)
        ns = myMPI.Bcast(self.nSystems, world, root=root)
        if world.rank != root:
            sys = np.ndarray(ns, dtype=FdemSystem)
            for i in range(ns):
                sys[i] = FdemSystem()
        else:
            sys = self.system

        sysTmp = []
        for i in range(ns):
            sysTmp.append(sys[i].Bcast(world, root=root))

        out = FdemData(chunks[world.rank], nf, sysTmp)
        out._x = self.x.Scatterv(starts, chunks, world, root=root)
        out._y = self.y.Scatterv(starts, chunks, world, root=root)
        out._z = self.z.Scatterv(starts, chunks, world, root=root)
        out._elevation = self.elevation.Scatterv(starts, chunks, world, root=root)
        out._data = self._data.Scatterv(starts, chunks, world, root=root)
        out._std = self._std.Scatterv(starts, chunks, world, root=root)
        out._predictedData = self._predictedData.Scatterv(starts, chunks, world, root=root)
        out._fiducial = self.fiducial.Scatterv(starts, chunks, world, root=root)
        out._lineNumber = self.lineNumber.Scatterv(starts, chunks, world, root=root)

        return out


    def write(self, fileNames, std=False, predictedData=False):

        if isinstance(fileNames, str):
            fileNames = [fileNames]

        assert len(fileNames) == self.nSystems, ValueError("fileNames must have length equal to the number of systems {}".format(self.nSystems))

        for i, sys in enumerate(self.system):
            # Create the header
            header = "Line Fid Easting Northing Elevation Height "

            for x in sys.frequencies:
                header += "I_{0} Q_{0} ".format(x)

            if not self.powerline is None:
                header += 'Power_line '
            if not self.magnetic is None:
                header += 'Magnetic'

            d = np.empty(2*sys.nFrequencies)

            if std:
                for x in sys.frequencies:
                    header += "I_{0}_Err Q_{0}_Err ".format(x)
                s = np.empty(2*sys.nFrequencies)

            with open(fileNames[i], 'w') as f:
                f.write(header+"\n")
                with np.printoptions(formatter={'float': '{: 0.15g}'.format}, suppress=True):
                    for j in range(self.nPoints):

                        x = np.asarray([self.lineNumber[j], self.fiducial[j], self.x[j], self.y[j], self.elevation[j], self.z[j]])

                        if predictedData:
                            d[0::2] = self.predictedData[j, :sys.nFrequencies]
                            d[1::2] = self.predictedData[j, sys.nFrequencies:]
                        else:
                            d[0::2] = self.data[j, :sys.nFrequencies]
                            d[1::2] = self.data[j, sys.nFrequencies:]

                        if std:
                            s[0::2] = self.std[j, :sys.nFrequencies]
                            s[1::2] = self.std[j, sys.nFrequencies:]
                            x = np.hstack([x, d, s])
                        else:
                            x = np.hstack([x, d])

                        if not self.powerline is None:
                            x = np.hstack([x, self.powerline[j]])
                        if not self.magnetic is None:
                            x = np.hstack([x, self.magnetic[j]])

                        y = ""
                        for a in x:
                            y += "{} ".format(a)

                        f.write(y+"\n")
