""" @FdemData_Class
Module describing an EMData Set where channels are associated with an xyz co-ordinate
"""
from .Data import Data
from ..datapoint.FdemDataPoint import FdemDataPoint
from ....base import customFunctions as cF
from ....base import customPlots as cP
from ....classes.core.StatArray import StatArray
from ...system.FdemSystem import FdemSystem
import numpy as np
from ....base import fileIO as fIO
#from ....base import Error as Err
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

    def __init__(self, nPoints=1, nFrequencies=1, systems=None):
        """Instantiate the FdemData class. """

        if (not systems is None):
            if isinstance(systems, (str, FdemSystem)):
                systems = [systems]
            nSystems = len(systems)
            # Make sure that list contains strings or TdemSystem classes
            assert all([isinstance(x, (str, FdemSystem)) for x in systems]), TypeError("systems must be str or list of either str or geobipy.FdemSystem")

            system = np.ndarray(nSystems, dtype=FdemSystem)

            nFrequencies = np.empty(nSystems, dtype=np.int32)
            for i, s in enumerate(systems):
                if isinstance(s, str):
                    system[i] = FdemSystem()
                    system[i].read(systems[i])
                else:
                    system[i] = systems[i]
                nFrequencies[i] = system[i].nFrequencies
        else:
            nFrequencies = np.int32(np.atleast_1d(nFrequencies))
            nSystems = nFrequencies.size
            system = None

        # Data Class containing xyz and channel values
        Data.__init__(self, nPoints=nPoints, nChannelsPerSystem=2*nFrequencies, dataUnits="ppm")
        # StatArray of the line number for flight line data
        self.line = StatArray(nPoints, 'Line Number')
        # StatArray of the id number
        self.id = StatArray(nPoints, 'ID Number')
        # StatArray of the elevation
        self.elevation = StatArray(nPoints, 'Elevation', 'm')
        # Assign data names
        self._data.name = 'Fdem Data'

        self.system = system
        self.nSystems = nSystems

        k = 0
        for i in range(self.nSystems):
            # Set the channel names
            if not self.system is None:
                for iFrequency in range(2*self.nFrequencies[i]):
                    self.channelNames[k] = '{} {} (Hz)'.format(self.getMeasurementType(iFrequency), self.getFrequency(iFrequency))
                    k += 1

        self.iActive = self.getActiveChannels()

    @property
    def nFrequencies(self):
        return np.int32(0.5 * self.nChannelsPerSystem)

    @property
    def data(self):
        """The data"""
        return self._data

    @property
    def predictedData(self):
        """The predicted data"""
        return self._predictedData

    @property
    def std(self):
        """The standard deviation"""
        return self._std

    @property
    def channelNames(self):
        return self._channelNames


    def fileInformation(self):
        """Description of the data file."""
        tmp = 'The data file is structured using columns with the first line containing a header line.\n'\
              'The header should contain the following entries \n'\
              'Line [ID or FID] [X or N or northing] [Y or E or easting] [Z or DTM or dem_elev] '\
              '[Alt or Laser or bheight] [I Q] ... [I Q] \n'\
              'Do not include brackets [], [I Q] are the in-phase and quadrature values for each measurement frequency.\n'
        return tmp
            

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


    def getLine(self, line):
        """Gets the data in the given line number 
        
        Parameters
        ----------
        line : float
            A line number from the data file

        Returns
        -------
        out : geobipy.FdemData
            A data class containing only the data in the line
        
        """
        i = np.where(self.line == line)[0]
        assert (i.size > 0), 'Could not get line with number {}'.format(line)
        return self[i]


    def __getitem__(self, i):
        """Define item getter for Data 

        Allows slicing into the data FdemData[i]        
        
        """
        tmp = FdemData(np.size(i), self.nFrequencies)
        tmp.x[:] = self.x[i]
        tmp.y[:] = self.y[i]
        tmp.z[:] = self.z[i]
        tmp._data[:, :] = self._data[i, :]
        tmp._std[:, :] = self._std[i, :]
        tmp._predictedData[:, :] = self._predictedData[i, :]
        tmp.line[:] = self.line[i]
        tmp.id[:] = self.id[i]
        tmp.elevation[:] = self.elevation[i]
        tmp.system = self.system
        tmp.nSystems = self.nSystems
        # tmp.nChannelsPerSystem = self.nChannelsPerSystem
        return tmp


    def getDataPoint(self, i):
        """Get the ith data point from the data set 
        
        Parameters
        ----------
        i : int
            The data point to get
            
        Returns
        -------
        out : geobipy.FdemDataPoints
            The data point
            
        """
        return FdemDataPoint(self.x[i], self.y[i], self.z[i], self.elevation[i], self._data[i, :], self._std[i, :], self.system)


    # def mapChannel(self, channel, *args, system=0, **kwargs):
    #     """ Create a map of the specified data channel """

    #     assert channel < 2*self.nFrequencies[system], ValueError('Requested channel must be less than '+str(2*self.nFrequencies[system]))

    #     tmp = self.getChannel(system, channel)
    #     kwargs['c'] = tmp

    #     self.mapPlot(*args, **kwargs)

    #     cP.title(tmp.name)

    #     # Data.mapChannel(self, channel, *args, **kwargs)

    #     # cP.title(self._channelNames[channel])


    def plot(self, xAxis='index', channels=None, **kwargs):
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
        geobipy.customPlots.plot : For additional keyword arguments

        """

        noLegend = kwargs.pop('noLegend', False)
        kwargs['noLegend'] = noLegend
        ax, legend = super().plot(xAxis, channels, **kwargs)

        if not noLegend:
            legend.set_title('Frequency (Hz)')

        return ax, legend


    def plotLine(self, line, system=0, xAxis='index', **kwargs):
        """ Plot the specified line """

        l = self.getLine(line)
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

        self.readSystemFile(systemFilename)

        nPoints, iC, iD, iS = self.__readColumnIndices(dataFilename, self.system)

        # Get all readable column indices for the first file.
        tmp = [iC[0]]
        tmp.append(iD[0])
        if not iS[0] is None:
            tmp.append(iS[0])
        indicesForFile = np.hstack(tmp)

        # Initialize the EMData Class
        FdemData.__init__(self, nPoints, systems=self.system)

        # Read in the columns from the first data file
        values = fIO.read_columns(dataFilename[0], indicesForFile, 1, nPoints)

        # Assign columns to variables
        self.line[:] = values[:, 0]
        self.id[:] = values[:, 1]
        self.x[:] = values[:, 2]
        self.y[:] = values[:, 3]
        self.z[:] = values[:, 4]
        self.elevation[:] = values[:, 5]

        # EM data columns are in the following order
        # I1 Q1 I2 Q2 .... IN QN ErrI1 ErrQ1 ... ErrIN ErrQN
        # Reshuffle to the following
        # I1 I2 ... IN Q1 Q2 ... QN and
        # ErrI1 ErrI2 ... ErrIN ErrQ1 ErrQ2 ... ErrQN
        iSys = self._systemIndices(0)
        self._data[:, iSys] = values[:, 6:6+2*self.nFrequencies[0]]
        if (iS[0]):
            self._std[:, iSys] = values[:, 6+2*self.nFrequencies[0]:]
        else:
            self._std[:, iSys] = 0.1 * self._data[:, iSys]
                    
        # Read in the data for the other systems.  Only read in the data and, if available, the errors
        for i in range(1, self.nSystems):
            # Get all readable column indices for the file.
            tmp = [iD[i]]
            if not iS[i] is None:
                tmp.append(iS[i])
            indicesForFile = np.hstack(tmp)

            # Read the columns
            values = fIO.read_columns(dataFilename[i], indicesForFile[i], 1, nPoints)
            # Assign the data
            iSys = self._systemIndices(i)
            self._data[:, iSys] = values[:, 6:6 + 2*self.nFrequencies[i]]
            if (iS[i]):
                self._std[:, iSys] = values[:, 6+2*self.nFrequencies[i]:]
            else:
                self._std[:, iSys] = 0.1 * self._data[:, iSys]

        self.iActive = self.getActiveChannels()

    def readSystemFile(self, systemFilename):
        """ Reads in the system handler using the system file name """

        if isinstance(systemFilename, str):
            systemFilename = [systemFilename]

        nSys = len(systemFilename)
        self.system = np.ndarray(nSys, dtype=FdemSystem)

        for i in range(nSys):
            self.system[i] = FdemSystem()
            self.system[i].read(systemFilename[i])
        
        self.nSystems = nSys
        self.nChannelsPerSystem = np.asarray([np.int32(2*x.nFrequencies) for x in self.system])
        self._systemOffset = np.append(0, np.cumsum(self.nChannelsPerSystem))


    # def toVTK(self, fileName, prop=['data', 'predicted', 'std'], format='binary'):

    #     super().toVTK(fileName, prop=prop, format=format)


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
        
        nSystems = len(system) if isinstance(system, list) else 1
        assert all(isinstance(s, FdemSystem) for s in system), TypeError("system must contain geobipy.FdemSystem classes.")

        # First get the number of points in each data file. They should be equal.
        nPoints = np.empty(nSystems, dtype=np.int64)
        for i in range(nSystems):
            nPoints[i] = fIO.getNlines(dataFilename[i], 1)
        for i in range(1, nSystems):
            assert nPoints[i] == nPoints[0], Exception('Number of data points {} in file {} does not match {} in file {}'.format(nPoints[i], dataFilename[i], nPoints[0], dataFilename[0]))
        nPoints = nPoints[0]


        for k, f in enumerate(dataFilename):

            # Get the column headers of the data file
            channels = fIO.getHeaderNames(f)
            channels = [channel.lower() for channel in channels]
            nChannels = len(channels)

            # Check for each aspect of the data file and the number of columns
            nCoordinates = 0

            for channel in channels:
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
                elif(channel in ['z','dtm','dem_elev','dem_np','topo', 'elev']):
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
            _columnIndex = np.zeros(6, dtype=np.int32)
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
                elif(channel in ['z','dtm','dem_elev','dem_np','topo', 'elev']):
                    _columnIndex[5] = j

            i1 = nCoordinates + 2*system[k].nFrequencies
            i2 = nCoordinates + 4*system[k].nFrequencies
            tmp = np.arange(nCoordinates, i1)
            _dataIndices = np.hstack((tmp[::2], tmp[1::2]))
            tmp = np.arange(i1, i2)
            _errIndices = np.hstack((tmp[::2], tmp[1::2])) if _hasErrors else None

            indices.append(_columnIndex)
            dataIndices.append(_dataIndices)
            errIndices.append(_errIndices)

        return nPoints, indices, dataIndices, errIndices


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
        self.readSystemFile(systemFilename)
                

        self._nPoints, self._iC, self._iD, self._iS = self.__readColumnIndices(dataFilename, self.system)

        if isinstance(dataFilename, str):
            dataFilename = [dataFilename]

        self._file = []
        for f in dataFilename:
            self._file.append(open(f, 'r'))
        for f in self._file:
            fIO.skipLines(f, nLines=1)

        # Get all readable column indices for the first file.
        self._indicesForFile = []
        for i in range(self.nSystems):
            tmp = [self._iC[i]]
            tmp.append(self._iD[i])
            if not self._iS[i] is None:
                tmp.append(self._iS[i])
            self._indicesForFile.append(np.hstack(tmp))


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
                S[iSys] = values[6 + 2*self.nFrequencies[j] : 6 + 4*self.nFrequencies[j]]

        values = values[0]

        return FdemDataPoint(x=values[2], y=values[3], z=values[4], elevation=values[5], data=D, std=S, system=self.system)
        

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

        dat = None
        dat = Data.Bcast(self, world, root=root)
        this = FdemData(dat.nPoints, int(dat.nChannels/2))
        this.x = dat.x
        this.y = dat.y
        this.z = dat.z
        this.set[0] = dat.set[0]
        this.id = self.id.Bcast(world, root=root)
        this.line = self.line.Bcast(world, root=root)
        this.elevation = self.elevation.Bcast(world, root=root)
        this.system = self.system.Bcast(world, root=root)
        return this


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

        dat = None
        dat = Data.Scatterv(self, starts, chunks, world, root=root)
        this = FdemData(dat.nPoints, dat.nFrequencies)
        this.x = dat.x
        this.y = dat.y
        this.z = dat.z
        this.set = dat.set
        this.id = self.id.Scatterv(starts, chunks, world, root=root)
        this.line = self.line.Scatterv(starts, chunks, world, root=root)
        this.elevation = self.elevation.Scatterv(starts, chunks, world, root=root)
        this.system = self.system.Bcast(world, root=root)
        return this
