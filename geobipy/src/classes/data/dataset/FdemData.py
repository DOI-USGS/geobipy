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
        * If str: Must be a file name from which to read FD system information from
        * If FdemSystem: A deepcopy is made

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

    def __init__(self, nPoints=1, nFrequencies=1, system=None):
        """Instantiate the FdemData class. """
        # Data Class containing xyz and channel values
        Data.__init__(self, nPoints, 2*nFrequencies, "ppm")
        # StatArray of the line number for flight line data
        self.line = StatArray(nPoints, 'Line Number')
        # StatArray of the id number
        self.id = StatArray(nPoints, 'ID Number')
        # StatArray of the elevation
        self.e = StatArray(nPoints, 'Elevation', 'm')
        # Assign data names
        self.D.name='Fdem Data'

        if (not system is None):
            if isinstance(system, FdemSystem):
                self.sys = system.deepcopy()
            else:
                # Instantiate the system
                self.sys = FdemSystem()
                self.sys.read(system)
        else:
            self.sys = FdemSystem()

        if (not self.sys is None):
            # Set the channel names
            self.names = [None]*self.nChannels
            for i in range(2 * self.sys.nFreq):
                self.names[i] = self.getMeasurementType(i) + str(self.getFrequency(i))+' (Hz)'


    def read(self, dataFname, systemFname):
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
        sys = FdemSystem()
        sys.read(systemFname)

        # Get the column headers of the data file
        channels = fIO.getHeaderNames(dataFname)
        nChannels = len(channels)
        # Get the number of points
        nPoints = fIO.getNlines(dataFname, 1)
        # To grab the EM data, skip the following header names. (More can be added to this)
        # Initialize a column identifier for x y z
        tmp = [0, 0, 0]
        i = -1
        for j, line in enumerate(channels):
            line = line.lower()
            if (line in ['e', 'x', 'easting']):
                i += 1
                tmp[0] = i
            elif (line in ['n', 'y','northing']):
                i += 1
                tmp[1] = i
            elif (line in ['alt', 'laser', 'bheight']):
                i += 1
                tmp[2] = i
            elif(line in ['z', 'dtm','dem_elev']):
                i += 1
                iElev = i
            elif(line in ['line']):
                i += 1
                iLine = i
            elif(line in ['id', 'fid']):
                i += 1
                iID = i
        assert i == 5, ('Cannot determine data columns. \n' + self.dataFileStructure())
        # Initialize column identifiers
        cols = np.zeros(nChannels - i + 2, dtype=int)
        for j, k in enumerate(tmp):
            cols[j] = k
        tmp = range(i + 1, nChannels)
        nData = len(tmp)
        cols[3:] = tmp[:]

        # Check that the number of channels is even. EM data has two values, inphase and quadrature.
        # Therefore the number of data channels in the file must be even
        assert nData % 2 == 0, "Total number of in-phase + quadrature channels must be even"

        # Check that the file has errors listed, they may be missing.
        hasErrors = False
        if (nData > 2 * sys.nFreq):
            hasErrors = True
            # If the file has errors, make sure there are enough columns to
            # match the data
            assert (nData == 4 * sys.nFreq), "Number of error columns must 2 times # of Frequencies"

        # Initialize the EMData Class
        self.__init__(nPoints, sys.nFreq)
        self.sys = sys
        # Read in the data, extract the appropriate columns
        tmp = fIO.read_columns(dataFname, cols, 1, nPoints)
        # Assign the co-ordinates
        self.x[:] = tmp[:, 0]
        self.y[:] = tmp[:, 1]
        self.z[:] = tmp[:, 2]
        # EM data columns are in the following order
        # I1 Q1 I2 Q2 .... IN QN ErrI1 ErrQ1 ... ErrIN ErrQN
        # Reshuffle to the following
        # I1 I2 ... IN Q1 Q2 ... QN and
        # ErrI1 ErrI2 ... ErrIN ErrQ1 ErrQ2 ... ErrQN
        for i in range(self.sys.nFreq):
            i1 = (2 * i) + 3
            q1 = i1 + 1
            self.D[:, i] = tmp[:, i1]
            self.D[:, i + self.sys.nFreq] = tmp[:, q1]
            if (hasErrors):
                ei = i1 + 2 * self.sys.nFreq
                eq = q1 + 2 * self.sys.nFreq
                self.Std[:, i] = tmp[:, ei]
                self.Std[:, i + self.sys.nFreq] = tmp[:, eq]

        # Read the line numbers from the file
        tmp = fIO.read_columns(dataFname, [iLine], 1, nPoints)
        self.line[:] = tmp[:, 0]
        tmp = fIO.read_columns(dataFname, [iID], 1, nPoints)
        self.id[:] = tmp[:, 0]
        tmp = fIO.read_columns(dataFname, [iElev], 1, nPoints)
        self.e[:] = tmp[:, 0]

        # Set the channel names
        for i in range(2 * self.sys.nFreq):
            self.names[i] = self.getMeasurementType(i) + str(self.getFrequency(i))+' (Hz)'


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
        """Get the number of active data per data point.

        For each data point, counts the number of channels that are NOT nan.

        Returns
        -------
        out : int
            Number of active data

        """
        
        return np.sum(~np.isnan(self.D), 1)


    @property
    def nFrequencies(self):
        """Return the number of frequencies

        Returns
        -------
        nFrequencies : int
            Number of frequencies

        """
        return self.sys.nFreq


    def getChannel(self, channel):
        """Gets the data in the specified channel 
        
        Parameters
        ----------
        channel : int
            A channel number less than 2 * number of frequencies

        Returns
        -------
        out : StatArray
            Contains the values of the requested channel

        """
        assert channel < 2*self.sys.nFreq, 'Requested channel must be less than '+str(2*self.sys.nFreq)

        if (channel < self.sys.nFreq):
            tmp='InPhase - Frequency:'
        else:
            tmp='Quadrature - Frequency:'
        tmp += ' '+str(self.sys.freq[channel%self.sys.nFreq])

        tmp = StatArray(self.D[:, channel], tmp, self.D.units)

        return tmp


    def getMeasurementType(self, channel):
        """Returns the measurement type of the channel

        Parameters
        ----------
        channel : int
            Channel number

        Returns
        -------
        out : str
            Either "In-Phase " or "Quadrature "
        
        """
        return 'In-Phase ' if channel <self.sys.nFreq else 'Quadrature '


    def getFrequency(self, channel):
        """Return the measurement frequency of the channel

        Parameters
        ----------
        channel : int
            Channel number

        Returns
        -------
        out : float
            The measurement frequency of the channel

        """
        return self.sys.freq[channel%self.sys.nFreq]


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
        assert (i.size > 0), 'Could not get line with number '+str(line)
        return self[i]


    def __getitem__(self, i):
        """Define item getter for Data 

        Allows slicing into the data FdemData[i]        
        
        """
        tmp = FdemData(np.size(i), self.nFrequencies)
        tmp.x[:] = self.x[i]
        tmp.y[:] = self.y[i]
        tmp.z[:] = self.z[i]
        tmp.D[:, :] = self.D[i, :]
        tmp.Std[:, :] = self.Std[i, :]
        tmp.line[:] = self.line[i]
        tmp.id[:] = self.id[i]
        tmp.e[:] = self.e[i]
        tmp.sys = self.sys     
        tmp.names = self.names
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
        return FdemDataPoint(self.x[i], self.y[i], self.z[i], self.e[i], self.D[i, :], self.Std[i, :], self.sys)


    def mapChannel(self, channel, *args, **kwargs):
        """ Create a map of the specified data channel """

        assert channel < 2*self.sys.nFreq, ValueError('Requested channel must be less than '+str(2*self.sys.nFreq))

        Data.mapChannel(self, channel, *args, **kwargs)

        if (channel < self.sys.nFreq):
            tmp='InPhase - Frequency:'
        else:
            tmp='Quadrature - Frequency:'
        cP.title(tmp+' '+str(self.sys.freq[channel%self.sys.nFreq]))


    def plotLine(self, line, **kwargs):
        """ Plot the specified line """
        l = self.getLine(line)
        r = StatArray(np.sqrt(l.x**2.0 + l.y**2.0),'Distance',l.x.units)
        r -= np.min(r)

        log = kwargs.pop('log', None)

        xscale = kwargs.pop('xscale','linear')
        yscale = kwargs.pop('yscale','linear')

        ax = plt.gca()
        ax = plt.subplot(211)

        for i in range(self.sys.nFreq):
            tmp=str(self.getFrequency(i))+' (Hz)'
            dTmp = l.D[:,i]
            dTmp, dum = cF._logSomething(dTmp, log)
            #if (log):
            #    dTmp[dTmp < 0.0] = np.nan
            #    dTmp = np.log10(dTmp)
            iPositive = np.where(dTmp > 0.0)[0]
            ax.plot(r[iPositive],dTmp[iPositive],label=tmp,**kwargs)
        cP.title('in-phase')
        ax.set_xticklabels([])
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
        leg.set_title('Frequency')

        ylabel = 'In-Phase ('+l.D.getUnits()+')'
        if (log):
            dum, logLabel = cF._logSomething([10], log)
            ylabel = logLabel + ylabel
        cP.ylabel(ylabel)

        plt.xscale(xscale)
        plt.yscale(yscale)

        ax = plt.subplot(212)
        for i in range(self.sys.nFreq, 2*self.sys.nFreq):
            tmp=str(self.getFrequency(i))+' (Hz)'
            dTmp = l.D[:,i]
            dTmp, dum = cF._logSomething(dTmp, log)
            iPositive = np.where(dTmp > 0.0)[0]
            ax.plot(r[iPositive],dTmp[iPositive],label=tmp,**kwargs)

        cP.title('quadrature')
        cP.suptitle("Line number "+str(line))
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
        leg.set_title('Frequency')

        cP.xlabel(cF.getNameUnits(r))

        ylabel = 'Quadrature ('+l.D.getUnits()+')'
        if (log):
            dum, logLabel = cF._logSomething([10], log)
            ylabel = logLabel + ylabel
        cP.ylabel(ylabel)

        plt.xscale(xscale)
        plt.yscale(yscale)
        return ax

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
        this = FdemData(dat.N, int(dat.nChannels/2))
        this.x = dat.x
        this.y = dat.y
        this.z = dat.z
        this.set = dat.set
        this.id = self.id.Bcast(world, root=root)
        this.line = self.line.Bcast(world, root=root)
        this.e = self.e.Bcast(world, root=root)
        this.sys = self.sys.Bcast(world, root=root)
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
        this = FdemData(dat.N, dat.nFrequencies)
        this.x = dat.x
        this.y = dat.y
        this.z = dat.z
        this.set = dat.set
        this.id = self.id.Scatterv(starts, chunks, world, root=root)
        this.line = self.line.Scatterv(starts, chunks, world, root=root)
        this.e = self.e.Scatterv(starts, chunks, world, root=root)
        this.sys = self.sys.Bcast(world, root=root)
        return this
