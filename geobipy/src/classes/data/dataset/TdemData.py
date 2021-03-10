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

    def __init__(self, systems=None, **kwargs):
        """ Initialize the TDEM data """

        if systems is None:
            return

        self.system = systems

        # Data Class containing xyz and channel values
        super().__init__(nChannelsPerSystem=self.nTimes, units=r"$\frac{V}{m^{2}}$", **kwargs)

        # StatArray of Transmitter loops
        self.transmitter = kwargs.get('transmitter', None)
        # StatArray of Receiever loops
        self.receiver = kwargs.get('receiver', None)
        # Loop Offsets
        self.loopOffset = kwargs.get('loopOffset', None)


        self.channelNames = kwargs.get('channel_names', None)


    @property
    def channelNames(self):
        return self._channelNames


    @channelNames.setter
    def channelNames(self, values):
        if values is None:
            self._channelNames = []
            for i in range(self.nSystems):
                # Set the channel names
                for iTime in range(self.nTimes[i]):
                    self._channelNames.append('Time {:.3e} s'.format(self.system[i].windows.centre[iTime]))
        else:
            assert all((isinstance(x, str) for x in values))
            assert len(values) == self.nChannels, Exception("Length of channelNames must equal total number of channels {}".format(self.nChannels))
            self._channelNames = values

    @property
    def loopOffset(self):
        return self._loopOffset


    @loopOffset.setter
    def loopOffset(self, values):
        if (values is None):
            self._loopOffset = StatArray.StatArray((self.nPoints, 3), "Loop Offset")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.all(np.shape(values) == (self.nPoints, 3)), ValueError("loopOffset must have shape {}".format((self.nPoints, 3)))
            if (isinstance(values, StatArray.StatArray)):
                self._loopOffset = values.deepcopy()
            else:
                self._loopOffset = StatArray.StatArray(values, "Loop Offset")


    @property
    def receiver(self):
        return self._receiver


    @receiver.setter
    def receiver(self, values):

        if (values is None):
            self._receiver = StatArray.StatArray(self.nPoints, 'Receiver loops', dtype=CircularLoop)
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("receiver must have shape {}".format(self.nPoints))
            # if (isinstance(values, StatArray.StatArray)):
            #     self._receiver = values.deepcopy()
            # else:
            self._receiver = values #StatArray.StatArray(values, 'Receiver loops', dtype=CircularLoop)


    @property
    def system(self):
        return self._system


    @system.setter
    def system(self, values):

        if isinstance(values, (str, TdemSystem)):
            values = [values]
        nSystems = len(values)
        # Make sure that list contains strings or TdemSystem classes
        assert all([isinstance(x, (str, TdemSystem)) for x in values]), TypeError("system must be str or list of either str or geobipy.TdemSystem")

        self._system = np.ndarray(nSystems, dtype=TdemSystem)

        for i, s in enumerate(values):
            if isinstance(s, str):
                self._system[i] = TdemSystem().read(s)
            else:
                self._system[i] = s

    @property
    def transmitter(self):
        return self._transmitter


    @transmitter.setter
    def transmitter(self, values):

        if (values is None):
            self._transmitter = StatArray.StatArray(self.nPoints, 'Transmitter loops', dtype=CircularLoop)
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values)
            assert np.size(values) == self.nPoints, ValueError("transmitter must have shape {}".format(self.nPoints))
            # if (isinstance(values, StatArray.StatArray)):
            #     self._transmitter = values.deepcopy()
            # else:
            self._transmitter = values #StatArray.StatArray(self.nPoi, 'Transmitter loops', dtype=CircularLoop)


    def append(self, other):

        super().append(self, other)

        self.loopOffset = np.hstack([self.loopOffset, other.loopOffset])
        self.T = np.hstack([self.T, other.T])
        self.R = np.hstack(self.R, other.R)


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

        self.system = systemFilename
        nPoints, iC, iR, iT, iOffset, iD, iS = self.__readColumnIndices(dataFilename, self.system)

        # Get all readable column indices for the first file.
        tmp = [iC[0], iR[0], iT[0], iOffset[0], iD[0]]
        if not iS[0] is None:
            tmp.append(iS[0])
        indicesForFile = np.hstack(tmp)

        # Read in the columns from the first data file
        values = fIO.read_columns(dataFilename[0], indicesForFile, 1, nPoints)

        # Assign columns to variables
        lineNumber = values[:, 0]
        fiducial = values[:, 1]
        x = values[:, 2]
        y = values[:, 3]
        elevation = values[:, 4]
        z = values[:, 5]

        self.__init__(lineNumber=lineNumber,
                      fiducial=fiducial,
                      x=x,
                      y=y,
                      z=z,
                      elevatin=elevation,
                      systems=self.system)

        # Assign the orientations of the acquisistion loops
        i0 = 6
        self.receiver = None
        for i in range(nPoints):
            self.receiver[i] = CircularLoop(z=self.z[i], pitch=values[i, i0], roll=values[i, i0+1], yaw=values[i, i0+2], radius=self.system[0].loopRadius())
        i0 += 3

        self.transmitter = None
        for i in range(nPoints):
            self.transmitter[i] = CircularLoop(z=self.z[i], pitch=values[i, i0], roll=values[i, i0+1], yaw=values[i, i0+2], radius=self.system[0].loopRadius())
        i0 += 3

        self.loopOffset = values[:, i0:i0+3]
        i0 += 3

        # Assign the data values
        i1 = i0 + self.nTimes[0]
        iData = np.arange(i0, i1)

        # Get the data values
        iSys = self._systemIndices(0)

        self.data[:, iSys] = values[:, iData]
        # If the data error columns are given, assign them
        if (iS[0] is None):
            self.std[:, iSys] = 0.1 * self.data[:, iSys]
        else:
            i2 = i1 + self.nTimes[0]
            iStd = np.arange(i1, i2)
            self.std[:, iSys] = values[:, iStd]

        # Read in the data for the other systems.  Only read in the data and, if available, the errors
        for i in range(1, self.nSystems):
            # Assign the columns to read
            indicesForFile = iD[i]
            if (not iS[i] is None): # Append the error columns if they are available
                indicesForFile = np.concatenate(indicesForFile, iS[i])

            # Read the columns
            values = fIO.read_columns(dataFilename[i], indicesForFile, 1, nPoints)
            # Assign the data
            iSys = self._systemIndices(i)
            self.data[:, iSys] = values[:, :self.nTimes[i]]
            if (iS[i] is None):
                self.std[:, iSys] = 0.1 * self.data[:, iSys]
            else:
                self.std[:, iSys] = values[:, self.nTimes[i]:]

        # self.iActive = self.getActiveChannels()

        self.check()

        return self


    # def readSystemFile(self, systemFilename):
    #     """ Reads in the C++ system handler using the system file name """

    #     if isinstance(systemFilename, str):
    #         systemFilename = [systemFilename]

    #     nSys = len(systemFilename)
    #     self.system = np.ndarray(nSys, dtype=TdemSystem)

    #     for i in range(nSys):
    #         self.system[i] = TdemSystem().read(systemFilename[i])

    #     # self.nSystems = nSys
    #     self.nChannelsPerSystem = np.asarray([np.int32(x.nwindows()) for x in self.system])

    #     self._systemOffset = np.append(0, np.cumsum(self.nChannelsPerSystem))


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
        offsetIndices = []
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
            nOffset = 0

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
                elif channel in ['txrx_dx', 'txrx_dy', 'txrx_dz']:
                    nOffset += 1
                elif "on[" in channel:
                    nOnData += 1
                elif "onerr[" in channel:
                    nOnErr += 1
                elif "off[" in channel:
                    nOffData += 1
                elif "offerr[" in channel:
                    nOffErr += 1

            assert nCoordinates >= 6, Exception("Data file must contain columns for easting, northing, height, elevation, line, and fid. \n {}".format(self.fileInformation()))

            assert nRloop == 3, Exception('Must have all three RxPitch, RxRoll, and RxYaw headers in data file {} if reciever orientation is specified. \n {}'.format(f, self.fileInformation()))
            assert nTloop == 3, Exception('Must have all three TxPitch, TxRoll, and TxYaw headers in data file {} if transmitter orientation is specified. \n {}'.format(f, self.fileInformation()))
            assert nOffset == 3, Exception('Must have all three txrx_dx, txrx_dy, and txrx_dz headers in data file {} if transmitter-reciever loop separation is specified. \n {}'.format(f, self.fileInformation()))

            assert nOffData == system[k].windows.centre.size, Exception("Number of Off time columns {} in {} does not match number of times {} in system file {}. \n {}".format(nOffData, f, system[k].fileName, system[k].windows.centre.size, self.fileInformation()))
            if nOffErr > 0:
                assert nOffErr == nOffData, Exception("Number of Off time standard deviation estimates does not match number of Off time data columns in file {}. \n {}".format(f, self.fileInformation()))

            _indices = np.zeros(6, dtype=np.int32)
            _rLoopIndices = None if nRloop == 0 else np.empty(3, dtype=np.int32)
            _tLoopIndices = None if nTloop == 0 else np.empty(3, dtype=np.int32)
            _offsetIndices = None if nOffset == 0 else np.empty(3, dtype=np.int32)
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

                # Get the transmitter loop orientation indices
                elif (channel == 'txrx_dx'):
                    _offsetIndices[0] = j
                elif (channel == 'txrx_dy'):
                    _offsetIndices[1] = j
                elif (channel == 'txrx_dz'):
                    _offsetIndices[2] = j

                elif ('off[' in channel):
                    i1 += 1
                    _offdataIndices[i1] = j

                elif ('offerr[' in channel):
                    i2 += 1
                    _offerrIndices[i2] = j

            indices.append(_indices)
            rLoopIndices.append(_rLoopIndices)
            tLoopIndices.append(_tLoopIndices)
            offsetIndices.append(_offsetIndices)
            offdataIndices.append(_offdataIndices)
            offerrIndices.append(_offerrIndices)

        return nPoints, indices, rLoopIndices, tLoopIndices, offsetIndices, offdataIndices, offerrIndices


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
        self.system = systemFilename

        self._nPoints, self._iC, self._iR, self._iT, self._iOffset, self._iD, self._iS = self.__readColumnIndices(dataFileName, self.system)

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
            tmp = [self._iC[i], self._iR[i], self._iT[i], self._iOffset[i], self._iD[i]]
            if not self._iS[i] is None:
                tmp.append(self._iS[i])
            self._indicesForFile.append(np.hstack(tmp))

        # Remap the indices for the different components of the file to make reading easier.
        for i in range(self.nSystems):
            offset = self._iC[i].size
            self._iC[i] = np.arange(offset)
            self._iR[i] = np.arange(3) + offset
            offset += 3
            self._iT[i] = np.arange(3) + offset
            offset += 3
            self._iOffset[i] = np.arange(3) + offset
            offset += 3

            nTmp = self._iD[i].size
            self._iD[i] = np.arange(nTmp) + offset
            offset += nTmp
            if not self._iS[i] is None:
                nTmp = self._iS[i].size
                self._iS[i] = np.arange(nTmp) + offset

    @property
    def nSystems(self):
        return np.size(self.system)

    @property
    def nChannelsPerSystem(self):
        return np.asarray([s.nwindows() for s in self.system])


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
        R = CircularLoop(z=values[4], pitch=values[i0], roll=values[i0+1], yaw=values[i0+2], radius=self.system[0].loopRadius())
        i0 += 3

        T = CircularLoop(z=values[4], pitch=values[i0], roll=values[i0+1], yaw=values[i0+2], radius=self.system[0].loopRadius())
        i0 += 3

        loopOffset = values[i0:i0+3]
        i0 += 3

        out = TdemDataPoint(x=values[2], y=values[3], z=values[5], elevation=values[4], data=D, std=S, system=self.system, transmitter_loop=T, receiver_loop=R, loopOffset=loopOffset, lineNumber=values[0], fiducial=values[1])

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

        i = index
        return TdemDataPoint(self.x[i], self.y[i], self.z[i], self.elevation[i], self.data[i, :], self.std[i, :], self.predictedData[i, :], self.system, self.transmitter[i], self.receiver[i], self.loopOffset[i, :], self.lineNumber[i], self.fiducial[i])


    # def getLine(self, line):
    #     """ Gets the data in the given line number """
    #     assert line in self.lineNumber, ValueError("No line available in data with number {}".format(line))
    #     i = np.where(self.lineNumber == line)[0]
    #     return self[i]


    def times(self, system=0):
        """ Obtain the times from the system file """
        assert 0 <= system < self.nSystems, ValueError('system must be in (0, {}]'.format(self.nSystems))
        return StatArray.StatArray(self.system[system].windows.centre, 'Time', 'ms')


    def __getitem__(self, i):
        """ Define item getter for TdemData """
        if not isinstance(i, slice):
            i = np.unique(i)
        return TdemData(self.system,
                        x = self.x[i],
                        y = self.y[i],
                        z = self.z[i],
                        elevation = self.elevation[i],
                        lineNumber = self.lineNumber[i],
                        fiducial = self.fiducial[i],
                        transmitter = self.transmitter[i],
                        receiver = self.receiver[i],
                        loopOffset = self.loopOffset[i, :],
                        data = self.data[i, :],
                        std = self.std[i, :],
                        predictedData = self.predictedData[i, :],
                        channelNames = self.channelNames)


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
            '    Northing co-ordinate of the data point, (m)\n'
            'y or easting or e \n'
            '    Easting co-ordinate of the data point, (m)\n'
            'z or alt or laser or bheight \n'
            '    Altitude of the transmitter coil above ground level (m)\n'
            'dtm or dem_elev or dem_np \n'
            '    Elevation of the ground at the data point (m)\n'
            'txrx_dx \n'
            '    Distance in x between transmitter and reciever (m)\n'
            'txrx_dy \n'
            '    Distance in y between transmitter and reciever (m)\n'
            'txrx_dz \n'
            '    Distance in z between transmitter and reciever (m)\n'
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
            'Off[0] Off[1] ... Off[nWindows]  - with the number and square brackets\n'
            '    The measurements for each time specified in the accompanying system file under Receiver Window Times \n'
            'Optional columns\n'
            'OffErr[0] OffErr[1] ... Off[nWindows]\n'
            '    Estimates of standard deviation for each off time measurement.')
        return s


    def mapChannel(self, channel, system=0, *args, **kwargs):
        """ Create a map of the specified data channel """

        tmp = self.getChannel(system, channel)
        kwargs['c'] = tmp

        self.mapPlot(*args, **kwargs)

        cP.title(tmp.name)


    def plot(self, system=0, channels=None, xAxis='x', **kwargs):
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
                i1 = self.systemOffset[system] + i
                cP.plot(x, self.data[:, i1], label=self.channelNames[i1], **kwargs)
        else:
            channels = np.atleast_1d(channels)
            for j, i in enumerate(channels):
                cP.plot(x, self.data[:, i], label=self.channelNames[i], **kwargs)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
        leg.set_title(self.data.getNameUnits())

        plt.xlabel(cF.getNameUnits(x))


    def plotLine(self, line, xAxis='index', **kwargs):

        line = self.line(line)

        x = self.getXAxis(xAxis)

        for i in range(self.nSystems):
            plt.subplot(2, 1, i + 1)
            line._data[:, self._systemIndices(i)].plot(x=x, **kwargs)


    def plotWaveform(self, **kwargs):
        for i in range(self.nSystems):
            plt.subplot(self.nSystems, 1, i + 1)
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

    @property
    def summary(self):
        """ Display a summary of the TdemData """
        msg = PointCloud3D.summary(self, out=out)
        msg = "Tdem Data: \n"
        msg += "Number of Systems: :" + str(self.nSystems) + '\n'
        msg += self.lineNumber.summary
        msg += self.id.summary
        msg += self.elevation.summary
        return msg


    def fromHdf(self, grp, **kwargs):
        """ Reads the object from a HDF group """

        assert ('system_file_path' in kwargs), ValueError("missing 1 required argument 'system_file_path', the path to directory containing system files")

        system_file_path = kwargs.pop('system_file_path', None)
        assert (not system_file_path is None), ValueError("missing 1 required argument 'system_file_path', the path to directory containing system files")

        nSystems = np.int(np.asarray(grp.get('nSystems')))
        systems = []
        for i in range(nSystems):
            # Get the system file name. h5py has to encode strings using utf-8, so decode it!
            filename = str(np.asarray(grp.get('System{}'.format(i))), 'utf-8')
            td = TdemSystem().read(system_file_path+"//"+filename)
            systems.append(td)


        super().fromHdf(grp)

        self.systems = systems

        self._receiver = StatArray.StatArray(self.nPoints, 'Receiver loops', dtype=CircularLoop)
        tmp = np.asarray(grp['R/data'])
        for i in range(self.nPoints):
            self._receiver[i] = CircularLoop(*tmp[i, :])

        self._transmitter = StatArray.StatArray(self.nPoints, 'Receiver loops', dtype=CircularLoop)
        tmp = np.asarray(grp['T/data'])
        for i in range(self.nPoints):
            self._transmitter[i] = CircularLoop(*tmp[i, :])

        return self


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
                    system[i] = TdemSystem(systemFilename[i])

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
        this._fiducial = self.fiducial.Bcast(world, root=root)
        this._lineNumber = self.lineNumber.Bcast(world, root=root)
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
                this.T[i] = eval(cF.safeEval(lTmp[i]))

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
                this.R[i] = eval(cF.safeEval(lTmp[i]))

        # this.iActive = this.getActiveChannels()

        return this


    def Scatterv(self, starts, chunks, world, root=0, system=None):
        """ Scatterv the TdemData using MPI """

        nTimes = myMPI.Bcast(self.nTimes, world, root=root)

        systems = self._BcastSystem(world, root=root, system=system)

        # Instantiate a new Time Domain Data set on each worker
        this = TdemData(chunks[world.rank], nTimes, systems)

        # Broadcast the Data point id, line numbers and elevations
        this._fiducial = self.fiducial.Scatterv(starts, chunks, world, root=root)
        this._lineNumber = self.lineNumber.Scatterv(starts, chunks, world, root=root)
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

        # this.iActive = this.getActiveChannels()

        return this


    def write(self, fileNames, std=False, predictedData=False):

        if isinstance(fileNames, str):
            fileNames = [fileNames]

        assert len(fileNames) == self.nSystems, ValueError("fileNames must have length equal to the number of systems {}".format(self.nSystems))

        for i in range(self.nSystems):

            iSys = self._systemIndices(i)
            # Create the header
            header = "Line Fid Easting Northing Elevation Height txrx_dx txrx_dy txrx_dz TxPitch TxRoll TxYaw RxPitch RxRoll RxYaw "

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

                        x = np.asarray([self.lineNumber[j], self.id[j], self.x[j], self.y[j], self.elevation[j], self.z[j],
                                        self.loopOffset[j, 0], self.loopOffset[j, 1], self.loopOffset[j, 2],
                                        self.T.pitch, self.T.roll, self.T.yaw,
                                        self.R.pitch, self.R.roll, self.R.yaw])

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

                        f.write(y + "\n")
