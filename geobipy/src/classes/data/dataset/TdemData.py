"""
.. module:: StatArray
   :platform: Unix, Windows
   :synopsis: Time domain data set

.. moduleauthor:: Leon Foks

"""
from ...pointcloud.PointCloud3D import PointCloud3D
from .Data import Data
from ..datapoint.TdemDataPoint import TdemDataPoint
from ....classes.core import StatArray
from ...system.CircularLoop import CircularLoop
from ....base.customFunctions import safeEval
from ...system.TdemSystem import TdemSystem

import numpy as np
from ....base import fileIO as fIO
#from ....base import Error as Err
from ....base import customPlots as cP
from ....base import customFunctions as cF
from ....base import MPI as myMPI
import matplotlib.pyplot as plt
from os.path import join


class TdemData(Data):
    """Time domain electro magnetic data set

    A time domain data set with easting, northing, height, and elevation values. Each sounding in the data set can be given a receiver and transmitter loop.

    TdemData(nPoints=1, nTimes=[1], nSystems=1)

    Parameters
    ----------
    nPoints : int, optional
        Number of soundings in the data file
    nTimes : array of ints, optional
        Array of size nSystemsx1 containing the number of time gates in each system
    nSystem : int, optional
        Number of measurement systems

    Returns
    -------
    out : TdemData
        Time domain data set

    See Also
    --------
    :func:`~geobipy.src.classes.data.dataset.TdemData.TdemData.read`
        For information on file format

    """

    def __init__(self, nPoints=1, nTimes=1, systems=None):
        """ Initialize the TDEM data """

        # Systems are given
        if not systems is None:
            # if its a string, convert to a list
            if isinstance(systems, str):
                systems = [systems]
            nSystems = len(systems)
            # Make sure that list contains strings or TdemSystem classes
            assert all([isinstance(x, (str, TdemSystem)) for x in systems]), TypeError("systems must be str or list of either str or geobipy.TdemSystem")

            system = np.ndarray(nSystems, dtype=TdemSystem)

            nTimes = np.empty(nSystems, dtype=np.int32)
            for i, s in enumerate(systems):
                if isinstance(s, str):
                    system[i] = TdemSystem()
                    system[i].read(systems[i])
                else:
                    system[i] = systems[i]
                nTimes[i] = system[i].nwindows()
        else:
            nTimes = np.int32(np.atleast_1d(nTimes))
            nSystems = nTimes.size
            system = None

        # Data Class containing xyz and channel values
        Data.__init__(self, nPoints, nTimes, dataUnits=r"$\frac{V}{m^{2}}$")

        # StatArray of the line number for flight line data
        self.line = StatArray.StatArray(self.nPoints, 'Line Number')
        # StatArray of the id number
        self.id = StatArray.StatArray(self.nPoints, 'ID Number')
        # StatArray of the elevation
        self._elevation = StatArray.StatArray(self.nPoints, 'Elevation', 'm')

        # StatArray of Transmitter loops
        self.T = StatArray.StatArray(self.nPoints, 'Transmitter Loops', dtype=CircularLoop)
        # StatArray of Receiever loops
        self.R = StatArray.StatArray(self.nPoints, 'Receiver Loops', dtype=CircularLoop)

        self.system = system
        self.nSystems = nSystems
        # self.nTimes = nTimes

        k = 0
        for i in range(self.nSystems):
            # Set the channel names
            if not self.system is None:
                for iTime in range(self.nTimes[i]):
                    self.channelNames[k] = 'Time {:.3e} s'.format(self.system[i].windows.centre[iTime])
                    k += 1

        self.iActive = self.getActiveChannels()


    @property
    def nTimes(self):
        return self.nChannelsPerSystem


    def read(self, dataFilename, systemFilename):
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

        The data columns are read in according to the column names in the first line.  The header line should contain at least the following column names. Extra columns may exist, but will be ignored. In this description, the column name or its alternatives are given followed by what the name represents. Optional columns are also described.

        **Required columns**

        line
            Line number for the data point

        id or fid
            Id number of the data point, these be unique

        x or northing or n
            Northing co-ordinate of the data point

        y or easting or e
            Easting co-ordinate of the data point

        z or dtm or dem\_elev or dem\_np or topo
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

        # Get the number of systems to use
        if (isinstance(dataFilename, str)):
            dataFilename = [dataFilename]
        if (isinstance(systemFilename, str)):
            systemFilename = [systemFilename]

        nDatafiles = len(dataFilename)
        nSystems = len(systemFilename)

        assert nDatafiles == nSystems, Exception("Number of data files must match number of system files.")

        self.readSystemFile(systemFilename)
        nPoints, iC, iR, iT, iD, iS = self.__readColumnIndices(dataFilename, self.system)

        TdemData.__init__(self, nPoints, systems=self.system)

        # Get all readable column indices for the first file.
        tmp = [iC[0]]
        if not iR[0] is None:
            tmp.append(iR[0])
        if not iT[0] is None:
            tmp.append(iT[0])
        tmp.append(iD[0])
        if not iS[0] is None:
            tmp.append(iS[0])
        indicesForFile = np.hstack(tmp)

        # Read in the columns from the first data file
        values = fIO.read_columns(dataFilename[0], indicesForFile, 1, nPoints)

        # Assign columns to variables
        self.line[:] = values[:, 0]
        self.id[:] = values[:, 1]
        self.x[:] = values[:, 2]
        self.y[:] = values[:, 3]
        self.elevation[:] = values[:, 4]
        self.z[:] = values[:, 5]

        # Assign the orientations of the acquisistion loops

        i0 = 6
        if (not iR[0] is None):
            for i in range(nPoints):
                self.R[i] = CircularLoop(z=self.z[i], pitch=values[i, i0], roll=values[i, i0+1], yaw=values[i, i0+2], radius=self.system[0].loopRadius())
            i0 += 3
        else:
            for i in range(nPoints):
                self.R[i] = CircularLoop(z=self.z[i], radius=self.system[0].loopRadius())

        if (not iT[0] is None):
            for i in range(nPoints):
                self.T[i] = CircularLoop(z=self.z[i], pitch=values[i, i0], roll=values[i, i0+1], yaw=values[i, i0+2], radius=self.system[0].loopRadius())
            i0 += 3
        else:
            for i in range(nPoints):
                self.T[i] = CircularLoop(z=self.z[i], radius=self.system[0].loopRadius())

        # Assign the data values
        i1 = i0 + self.nTimes[0]
        iData = np.arange(i0, i1)

        # Get the data values
        iSys = self._systemIndices(0)
        self._data[:, iSys] = values[:, iData]
        # If the data error columns are given, assign them
        if (iS[0] is None):
            self._std[:, iSys] = 0.1 * self._data[:, iSys]
        else:
            i2 = i1 + self.nTimes[0]
            iStd = np.arange(i1, i2)
            self._std[:, iSys] = values[:, iStd]

        # Read in the data for the other systems.  Only read in the data and, if available, the errors
        for i in range(1, self.nSystems):
            # Assign the columns to read
            indicesForFile = iD[i]
            if (not iS[i] is None): # Append the error columns if they are available
                indicesForFile = np.append(indicesForFile, iS[i])

            # Read the columns
            values = fIO.read_columns(dataFilename[i], indicesForFile, 1, nPoints)
            # Assign the data
            iSys = self._systemIndices(i)
            self._data[:, iSys] = values[:, :self.nTimes[i]]
            if (iS[i] is None):
                self._std[:, iSys] = 0.1 * self._data[:, iSys]
            else:
                self._std[:, iSys] = values[:, self.nTimes[i]:]

        self.iActive = self.getActiveChannels()

        self.check()
  

    def readSystemFile(self, systemFilename):
        """ Reads in the C++ system handler using the system file name """

        if isinstance(systemFilename, str):
            systemFilename = [systemFilename]

        nSys = len(systemFilename)
        self.system = np.ndarray(nSys, dtype=TdemSystem)

        for i in range(nSys):
            self.system[i] = TdemSystem(systemFilename[i])
        
        self.nSystems = nSys
        self.nChannelsPerSystem = np.asarray([np.int32(x.nwindows()) for x in self.system])

        self._systemOffset = np.append(0, np.cumsum(self.nChannelsPerSystem))


    def __readColumnIndices(self, dataFilename, system):
        """Reads the column indices for the co-ordinates, loop orientations, and data from the TdemData file.
        
        Parameters
        ----------
        dataFilename : str or list of str
            Path to the data file(s)
        system : list of geobipy.TdemSystem
            System class for each time domain acquisition system.

        Returns
        -------
        indices : list of ints
            Size 6 indices to line, fid, easting, northing, height, and elevation.
        rLoopIndices : list of ints
            Size 3 indices to pitch, roll, and yaw, for the receiver loop.
        tLoopIndices : list of ints
            Size 3 indices to pitch, roll, and yaw, for the transmitter loop.
        offDataIndices : list of ints
            Indices to the off time data columns.  Size == number of time windows.
        offErrIndices : list of ints
            Indices to the off time uncertainty estimate columns.  Size == number of time windows.

        """

        indices = []
        rLoopIndices = []
        tLoopIndices = []
        offdataIndices = []
        offerrIndices = []

        if isinstance(dataFilename, str):
            dataFilename = [dataFilename]
        
        nSystems = len(system) if isinstance(system, list) else 1
        assert all(isinstance(s, TdemSystem) for s in system), TypeError("system must contain geobipy.TdemSystem classes.")

        # First get the number of points in each data file. They should be equal.
        nPoints = self._readNpoints(dataFilename)
        

        for k, f in enumerate(dataFilename):

            # Get the column headers of the data file
            channels = fIO.getHeaderNames(f)
            channels = [channel.lower() for channel in channels]

            # Check for each aspect of the data file and the number of columns
            nCoordinates = 0
            nOffData = 0
            nOffErr = 0
            nOnData = 0
            nOnErr = 0
            nRloop = 0
            nTloop = 0

            for channel in channels:
                if(channel in ['line']):
                    nCoordinates += 1
                elif(channel in ['id', 'fid']):
                    nCoordinates += 1
                elif (channel in ['e', 'x','easting']):
                    nCoordinates += 1
                elif (channel in ['n', 'y', 'northing']):
                    nCoordinates += 1
                elif (channel in ['alt', 'laser', 'bheight', 'height']):
                    nCoordinates += 1
                elif(channel in ['z','dtm','dem_elev','dem_np','topo', 'elev', 'elevation']):
                    nCoordinates += 1
                elif channel in ["rxpitch", "rxroll", "rxyaw"]:
                    nRloop += 1
                elif channel in ["txpitch", "txroll", "txyaw"]:
                    nTloop += 1
                elif "on[" in channel:
                    nOnData += 1
                elif "onerr[" in channel:
                    nOnErr += 1
                elif "off[" in channel:
                    nOffData += 1
                elif "offerr[" in channel:
                    nOffErr += 1

            assert nCoordinates >= 6, Exception("Data file must contain columns for easting, northing, height, elevation, line, and fid. \n {}".format(self.fileInformation()))

            if nRloop > 0:
                assert nRloop == 3, Exception('Must have all three RxPitch, RxRoll, and RxYaw headers in data file {} if reciever orientation is specified. \n {}'.format(f, self.fileInformation()))
            if nTloop > 0:
                assert nTloop == 3, Exception('Must have all three TxPitch, TxRoll, and TxYaw headers in data file {} if transmitter orientation is specified. \n {}'.format(f, self.fileInformation()))
            
            assert nOffData == system[k].windows.centre.size, Exception("Number of Off time columns in {} does not match number of times in system file {}. \n {}".format(f, system[k].fileName, self.fileInformation()))
            if nOffErr > 0:
                assert nOffErr == nOffData, Exception("Number of Off time standard deviation estimates does not match number of Off time data columns in file {}. \n {}".format(f, self.fileInformation()))

            _indices = np.zeros(6, dtype=np.int32)
            _rLoopIndices = None if nRloop == 0 else np.empty(3, dtype=np.int32)
            _tLoopIndices = None if nTloop == 0 else np.empty(3, dtype=np.int32)
            _offdataIndices = np.empty(nOffData, dtype=np.int32)
            _offerrIndices = None if nOffErr == 0 else np.empty(nOffErr, dtype=np.int32)

            i1 = -1
            i2 = -1
            
            for j, channel in enumerate(channels):
                if (channel in ['line']):
                    _indices[0] = j
                elif(channel in ['id', 'fid']):
                    _indices[1] = j
                elif (channel in ['e', 'x', 'easting']):
                    _indices[2] = j
                elif (channel in ['n', 'y', 'northing']):
                    _indices[3] = j
                elif(channel in ['z', 'dtm', 'dem_elev', 'dem_np', 'topo', 'elev', 'elevation']):
                    _indices[4] = j
                elif (channel in ['alt', 'laser', 'bheight', 'height']):
                    _indices[5] = j

                # Get the receiver loop orientation indices
                elif (channel == 'rxpitch'):
                    _rLoopIndices[0] = j
                elif (channel == 'rxroll'):
                    _rLoopIndices[1] = j
                elif (channel == 'rxyaw'):
                    _rLoopIndices[2] = j

                # Get the transmitter loop orientation indices
                elif (channel == 'txpitch'):
                    _tLoopIndices[0] = j
                elif (channel == 'txroll'):
                    _tLoopIndices[1] = j
                elif (channel == 'txyaw'):
                    _tLoopIndices[2] = j

                elif ('off[' in channel):
                    i1 += 1
                    _offdataIndices[i1] = j

                elif ('offerr[' in channel):
                    i2 += 1
                    _offerrIndices[i2] = j

            indices.append(_indices)
            rLoopIndices.append(_rLoopIndices)
            tLoopIndices.append(_tLoopIndices)
            offdataIndices.append(_offdataIndices)
            offerrIndices.append(_offerrIndices)

        return nPoints, indices, rLoopIndices, tLoopIndices, offdataIndices, offerrIndices

    
    def _readNpoints(self, dataFilename):
        """Read the number of points in a data file

        Parameters
        ----------
        dataFilename : list of str.
            Path to the data files.

        Returns
        -------
        nPoints : int
            Number of observations.

        """
        nSystems = len(dataFilename)
        nPoints = np.empty(nSystems, dtype=np.int64)
        for i in range(nSystems):
            nPoints[i] = fIO.getNlines(dataFilename[i], 1)
        for i in range(1, nSystems):
            assert nPoints[i] == nPoints[0], Exception('Number of data points {} in file {} does not match {} in file {}'.format(nPoints[i], dataFilename[i], nPoints[0], dataFilename[0]))
        return nPoints[0]


    def _initLineByLineRead(self, dataFileName, systemFilename):
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
        
        self._nPoints, self._iC, self._iR, self._iT, self._iD, self._iS = self.__readColumnIndices(dataFileName, self.system)

        if isinstance(dataFileName, str):
            dataFileName = [dataFileName]

        self._file = []
        for f in dataFileName:
            self._file.append(open(f, 'r'))
        for f in self._file:
            fIO.skipLines(f, nLines=1)

        # Get all readable column indices for the first file.
        self._indicesForFile = []
        
        for i in range(self.nSystems):
            tmp = [self._iC[i]]
            if not self._iR[i] is None:
                tmp.append(self._iR[i])
            if not self._iT[i] is None:
                tmp.append(self._iT[i])
            tmp.append(self._iD[i])
            if not self._iS[i] is None:
                tmp.append(self._iS[i])
            self._indicesForFile.append(np.hstack(tmp))

        # Remap the indices for the different components of the file to make reading easier.
        for i in range(self.nSystems):
            offset = self._iC[i].size
            self._iC[i] = np.arange(offset)
            if not self._iR[i] is None:
                nTmp = self._iR[i].size
                self._iR[i] = np.arange(nTmp) + offset
                offset += nTmp
            if not self._iT[i] is None:
                nTmp = self._iT[i].size
                self._iT[i] = np.arange(nTmp) + offset
                offset += nTmp
            nTmp = self._iD[i].size
            self._iD[i] = np.arange(nTmp) + offset
            offset += nTmp
            if not self._iS[i] is None:
                nTmp = self._iS[i].size
                self._iS[i] = np.arange(nTmp) + offset

        self._systemOffset = np.append(0, np.cumsum(self.nChannelsPerSystem))


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
            D[iSys] = values[j][self._iD[j]]
            if self._iS[j] is None:
                S[iSys] = 0.1 * D[iSys]
            else:
                S[iSys] = values[j][self._iS[j]]

        values = values[0]

        i0 = 6
        if (not self._iR[0] is None):
            R = CircularLoop(z=values[4], pitch=values[i0], roll=values[i0+1], yaw=values[i0+2], radius=self.system[0].loopRadius())
            i0 += 3
        else:
            R = CircularLoop(z=values[4], radius=self.system[0].loopRadius())

        if (not self._iT[0] is None):
            T = CircularLoop(z=values[4], pitch=values[i0], roll=values[i0+1], yaw=values[i0+2], radius=self.system[0].loopRadius())
            i0 += 3
        else:
            T = CircularLoop(z=values[4], radius=self.system[0].loopRadius())

        out = TdemDataPoint(x=values[2], y=values[3], z=values[5], elevation=values[4], data=D, std=S, system=self.system, T=T, R=R, lineNumber=values[0], fiducial=values[1])

        return out


    def _openDatafiles(self, dataFilename):
        self._file = []
        for i, f in enumerate(dataFilename):
            self._file.append(open(f, 'r'))
            fIO.skipLines(self._file[i], nLines=1)


    def _closeDatafiles(self):
        for f in self._file:
            if not f.closed:
                f.close()


    def check(self):
        if (np.any(self._data[~np.isnan(self._data)] <= 0.0)):
            print("Warning: Your data contains values that are <= 0.0")


    def estimateAdditiveError(self):
        """ Uses the late times after 1ms to estimate the additive errors and error bounds in the data. """
        for i in range(self.nSystems):
            h = 'System {} \n'.format(i)
            iS = self._systemIndices(i)
            D = self._data[:, iS]
            t = self.times(i)
            i1ms = t.searchsorted(1e-3)

            if (i1ms < t.size):
                lateD = D[:, i1ms:]
                if np.all(np.isnan(lateD)):
                    j = i1ms - 1
                    d = D[:, j]
                    while np.all(np.isnan(d)) and j > 0:
                        j -= 1
                        d = D[:, j]

                    h += 'All data values for times > 1ms are NaN \nUsing the last time gate with non-NaN values.\n'
                else:
                    d = lateD
                    h += 'Using {} time gates after 1ms\n'.format(self.nTimes[i] - i1ms)

            else:
                h = 'System {} has no time gates after 1ms \nUsing the last time gate with non-NaN values. \n'.format(i)

                j = -1
                d = D[:, j]
                while np.all(np.isnan(d)) and j > 0:
                    j -= 1
                    d = D[:, j]
            
            s = np.nanstd(d)
            h +=    '  Minimum: {} \n'\
                    '  Maximum: {} \n'\
                    '  Median:  {} \n'\
                    '  Mean:    {} \n'\
                    '  Std:     {} \n'\
                    '  4Std:    {} \n'.format(np.nanmin(d), np.nanmax(d), np.nanmedian(d), np.nanmean(d), s, 4.0*s)
            print(h)



    def getDataPoint(self, i):
        """ Get the ith data point from the data set """

        assert 0 <= i < self.nPoints, ValueError("Requested data point must have index (0, "+str(self.nPoints) + ']')

        return TdemDataPoint(self.x[i], self.y[i], self.z[i], self.elevation[i], self._data[i, :], self.std[i, :], self._predictedData[i, :], self.system, self.T[i], self.R[i], self.line[i], self.id[i])


    def getLine(self, line):
        """ Gets the data in the given line number """
        assert line in self.line, ValueError("No line available in data with number {}".format(line))
        i = np.where(self.line == line)[0]
        return self[i]


    def times(self, system=0):
        """ Obtain the times from the system file """
        assert 0 <= system < self.nSystems, ValueError('system must be in (0, {}]'.format(self.nSystems))
        return StatArray.StatArray(self.system[system].windows.centre, 'Time', 'ms')


    def __getitem__(self, i):
        """ Define item getter for TdemData """
        if not isinstance(i, slice):
            i = np.unique(i)
        tmp = TdemData(np.size(self.x[i]), self.nTimes, self.system)
        tmp.x[:] = self.x[i]
        tmp.y[:] = self.y[i]
        tmp.z[:] = self.z[i]
        tmp.line[:] = self.line[i]
        tmp.id[:] = self.id[i]
        tmp.elevation[:] = self.elevation[i]
        tmp.T[:] = self.T[i]
        tmp.R[:] = self.R[i]
        tmp.sys = np.ndarray(self.nSystems, dtype=TdemSystem)
        tmp._data[:, :] = self._data[i, :]
        tmp._std[:, :] = self._std[i, :]
        tmp._predictedData[:, :] = self._predictedData[i, :]
        tmp._channelNames = self._channelNames
        return tmp


    def fileInformation(self):
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
            '    Northing co-ordinate of the data point\n'
            'y or easting or e \n'
            '    Easting co-ordinate of the data point\n'
            'z or alt or laser or bheight \n'
            '    Altitude of the transmitter coil\n'
            'dtm or dem_elev or dem_np \n'
            '    Elevation of the ground at the data point\n'
            'Off[0] to Off[nWindows]  - with the number and brackets\n'
            '    The measurements for each time specified in the accompanying system file under Receiver Window Times \n'
            'Optional columns for loop orientation. If these are omitted, the loop is assumed horizontal \n'
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
            'Optional columns for off time uncertainty estimates. These should be estimates of the data standard deviation. \n'
            'OffErr[0] to Off[nWindows]\n'
            '    Uncertainty estimates for the data')
        return s


    def mapChannel(self, channel, system=0, *args, **kwargs):
        """ Create a map of the specified data channel """

        tmp = self.getChannel(system, channel)
        kwargs['c'] = tmp

        self.mapPlot(*args, **kwargs)

        cP.title(tmp.name)


    def plot(self, system=0, channels=None, xAxis='index', **kwargs):
        """ Plots the data 
        
        Parameters
        ----------
        system : int
            System to plot
        channels : sequence of ints
            Channels to plot
        
        """

        x = self.getXAxis(xAxis)

        ax = plt.gca()
        if channels is None:
            nCols = self.nTimes[system]
            for i in range(nCols):
                i1 = self._systemOffset[system] + i
                cP.plot(x, self._data[:, i1], label=self._channelNames[i1], **kwargs)
        else:
            channels = np.atleast_1d(channels)
            for j, i in enumerate(channels):
                cP.plot(x, self._data[:, i], label=self._channelNames[i], **kwargs)
    
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
        leg.set_title(self._data.getNameUnits())

        plt.xlabel(cF.getNameUnits(x))
        

    def plotLine(self, line, xAxis='index', **kwargs):

        line = self.getLine(line)

        x = self.getXAxis(xAxis)

        for i in range(self.nSystems):
            plt.subplot(2, 1, i + 1)
            line._data[:, self._systemIndices(i)].plot(x=x, **kwargs)


    def plotWaveform(self, **kwargs):
        for i in range(self.nSystems):
            plt.subplot(2, 1, i + 1)
            plt.plot(self.system[i].waveform.time, self.system[i].waveform.current, **kwargs)
            if (i == self.nSystems-1): cP.xlabel('Time (s)')
            cP.ylabel('Normalized Current (A)')
            plt.margins(0.1, 0.1)


    def pcolor(self, system=0, yAxis='index', **kwargs):
        """ Plot the data in the given system as a 2D array
        """
        D = self._data[:, self._systemIndices(system)]
        times = self.times(system)
        y = self.getXAxis(xAxis=yAxis)
        ax = D.pcolor(x=times, y=y, **kwargs)
        return ax


    def summary(self, out=False):
        """ Display a summary of the TdemData """
        msg = PointCloud3D.summary(self, out=out)
        msg = "Tdem Data: \n"
        msg += "Number of Systems: :" + str(self.nSystems) + '\n'
        msg += self.line.summary(True)
        msg += self.id.summary(True)
        msg += self.elevation.summary(True)
        if (out):
            return msg
        print(msg)


    def fromHdf(self, grp, **kwargs):
        """ Reads the object from a HDF group """

        assert ('sysPath' in kwargs), ValueError("missing 1 required argument 'sysPath', the path to directory containing system files")

        sysPath = kwargs.pop('sysPath', None)
        assert (not sysPath is None), ValueError("missing 1 required argument 'sysPath', the path to directory containing system files")

        nSystems = np.int(np.asarray(grp.get('nSystems')))
        systems = []
        for i in range(nSystems):
            # Get the system file name. h5py has to encode strings using utf-8, so decode it!
            systems.append(TdemSystem(join(sysPath, str(np.asarray(grp.get('System{}'.format(i))), 'utf-8'))))

        s = grp['d/data'].shape

        tmp = TdemData(nPoints=s[0], nTimes=s[1], systems=systems)

        item = grp.get('x')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp._x = obj.fromHdf(item)
        item = grp.get('y')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp._y = obj.fromHdf(item)
        item = grp.get('z')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp._z = obj.fromHdf(item)
        item = grp.get('e')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp._elevation = obj.fromHdf(item)

        item = grp.get('d')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp._data = obj.fromHdf(item)
        item = grp.get('s')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp._std = obj.fromHdf(item)
        item = grp.get('p')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp._predictedData = obj.fromHdf(item)
        item = grp.get('relErr')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp.relErr = obj.fromHdf(item)
        item = grp.get('addErr')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp.addErr = obj.fromHdf(item)
        item = grp.get('T')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp.T = obj.fromHdf(item)
        item = grp.get('R')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp.R = obj.fromHdf(item)

        tmp.iActive = tmp.getActiveChannels()

        return tmp


    def scatter2D(self, **kwargs):
        """Create a 2D scatter plot using the x, y coordinates.

        Can take any other matplotlib arguments and keyword arguments e.g. markersize etc.

        Parameters
        ----------
        c : 1D array_like or StatArray, optional
            Colour values of the points, default is the height of the points
        i : sequence of ints, optional
            Plot a subset of x, y, c, using the indices in i
            
        See Also
        --------
        geobipy.customPlots.Scatter2D : For additional keyword arguments you may use.

        """

        return Data.scatter2D(self, **kwargs)
        

    def _BcastSystem(self, world, root=0, system=None):
        """Broadcast the TdemSystems.

        The TD systems have a c++ backend.  The only way currently to instantiate a TdemSystem class is with a file that is read in.
        Therefore, to broadcast the systems, I have to broadcast the file names of the systems and have each worker read in the system file.
        However, if not system is None, I assume that system is a list of TdemSystem classes that already exists on each worker.
        If system is provided, simply assign them when broadcasting the TdemData.

        """

        # Since the Time Domain EM Systems are C++ objects on the back end, I can't Broadcast them through C++ (Currently a C++ Noob)
        # So instead, Broadcast the list of system file names saved in the TdemData Class and read the system files in on each worker.
        # This is cumbersome, but only done once at the beginning of the MPI
        # code.
 
        if system is None:
            if world.rank == root:
                sfnTmp = []
                for s in self.system:
                    sfnTmp.append(s.fileName)
            else:
                sfnTmp = None
            systemFilename = world.bcast(sfnTmp, root=root)
     
            nSystems = len(systemFilename)

            system = np.ndarray(nSystems, dtype=TdemSystem)
            for i in range(nSystems):
                    system[i] = TdemSystem()
                    system[i].read(systemFilename[i])

        return system
        


    def Bcast(self, world, root=0, system=None):
        """Broadcast the TdemData using MPI
        
        Parameters
        ----------

        """

        nPoints = myMPI.Bcast(self.nPoints, world, root=root)
        nTimes = myMPI.Bcast(self.nTimes, world, root=root)

        systems = self._BcastSystem(world, root=root, system=system)
        
        # Instantiate a new Time Domain Data set on each worker
        this = TdemData(nPoints, nTimes, systems)

        # Broadcast the Data point id, line numbers and elevations
        this.id = self.id.Bcast(world, root=root)
        this.line = self.line.Bcast(world, root=root)
        this._x = self.x.Bcast(world, root=root)
        this._y = self.y.Bcast(world, root=root)
        this._z = self.z.Bcast(world, root=root)
        this._elevation = self.elevation.Bcast(world, root=root)
        this._data = self._data.Bcast(world, root=root)
        this._std = self._std.Bcast(world, root=root)
        this._predictedData = self._predictedData.Bcast(world, root=root)

        # Broadcast the Transmitter Loops.
        if (world.rank == 0):
            lTmp = [None] * self.nPoints
            for i in range(self.nPoints):
                lTmp[i] = str(self.T[i])
        else:
            lTmp = []
        lTmp = myMPI.Bcast_list(lTmp, world)
        if (world.rank == 0):
            this.T = self.T
        else:
            for i in range(this.nPoints):
                this.T[i] = eval(safeEval(lTmp[i]))

        # Broadcast the Reciever Loops.
        if (world.rank == 0):
            lTmp = [None] * self.nPoints
            for i in range(self.nPoints):
                lTmp[i] = str(self.R[i])
        else:
            lTmp = []
        lTmp = myMPI.Bcast_list(lTmp, world)
        if (world.rank == 0):
            this.R = self.R
        else:
            for i in range(this.nPoints):
                this.R[i] = eval(safeEval(lTmp[i]))

        this.iActive = this.getActiveChannels()

        return this


    def Scatterv(self, starts, chunks, world, root=0, system=None):
        """ Scatterv the TdemData using MPI """

        nTimes = myMPI.Bcast(self.nTimes, world, root=root)

        systems = self._BcastSystem(world, root=root, system=system)

        # Instantiate a new Time Domain Data set on each worker
        this = TdemData(chunks[world.rank], nTimes, systems)

        # Broadcast the Data point id, line numbers and elevations
        this.id = self.id.Scatterv(starts, chunks, world, root=root)
        this.line = self.line.Scatterv(starts, chunks, world, root=root)
        this._x = self.x.Scatterv(starts, chunks, world, root=root)
        this._y = self.y.Scatterv(starts, chunks, world, root=root)
        this._z = self.z.Scatterv(starts, chunks, world, root=root)
        this._elevation = self.elevation.Scatterv(starts, chunks, world, root=root)
        this._data = self._data.Scatterv(starts, chunks, world, root=root)
        this._std = self._std.Scatterv(starts, chunks, world, root=root)
        this._predictedData = self._predictedData.Scatterv(starts, chunks, world, root=root)

        # Scatterv the Transmitter Loops.
        if (world.rank == 0):
            lTmp = [None] * self.nPoints
            for i in range(self.nPoints):
                lTmp[i] = str(self.T[i])
        else:
            lTmp = []
        lTmp = myMPI.Scatterv_list(lTmp, starts, chunks, world)
        if (world.rank == 0):
            this.T[:] = self.T[:chunks[0]]
        else:
            for i in range(this.nPoints):
                this.T[i] = eval(lTmp[i])

        # Scatterv the Reciever Loops.
        if (world.rank == 0):
            lTmp = [None] * self.nPoints
            for i in range(self.nPoints):
                lTmp[i] = str(self.R[i])
        else:
            lTmp = []
        lTmp = myMPI.Scatterv_list(lTmp, starts, chunks, world)
        if (world.rank == 0):
            this.R[:] = self.R[:chunks[0]]
        else:
            for i in range(this.nPoints):
                this.R[i] = eval(lTmp[i])

        this.iActive = this.getActiveChannels()

        return this


    def write(self, fileNames, std=False, predictedData=False):

        if isinstance(fileNames, str):
            fileNames = [fileNames]

        assert len(fileNames) == self.nSystems, ValueError("fileNames must have length equal to the number of systems {}".format(self.nSystems))

        for i in range(self.nSystems):

            iSys = self._systemIndices(i)
            # Create the header
            header = "Line Fid Easting Northing Elevation Height "

            for x in range(self.nTimes[i]):
                header += "Off[{}] ".format(x)

            d = np.empty(self.nTimes[i])

            if std:
                for x in range(self.nTimes[i]):
                    header += "OffErr[{}] ".format(x)
                s = np.empty(self.nTimes[i])

            with open(fileNames[i], 'w') as f:
                f.write(header+"\n")
                with np.printoptions(formatter={'float': '{: 0.15g}'.format}, suppress=True):
                    for j in range(self.nPoints):

                        x = np.asarray([self.line[j], self.id[j], self.x[j], self.y[j], self.elevation[j], self.z[j]])

                        if predictedData:
                            d[:] = self.predictedData[j, iSys]
                        else:
                            d[:] = self.data[j, iSys]
                        
                        if std:
                            s[:] = self.std[j, iSys]
                            x = np.hstack([x, d, s])
                        else:
                            x = np.hstack([x, d])

                        y = ""
                        for a in x:
                            y += "{} ".format(a)

                        f.write(y+"\n")
