"""
.. module:: StatArray
   :platform: Unix, Windows
   :synopsis: Time domain data set

.. moduleauthor:: Leon Foks

"""
from copy import deepcopy
from pandas import read_csv
from ...pointcloud.PointCloud3D import PointCloud3D
from .Data import Data
from ..datapoint.TdemDataPoint import TdemDataPoint
from ....classes.core import StatArray
from ...system.CircularLoop import CircularLoop
from ...system.TdemSystem import TdemSystem

import numpy as np
from ....base import fileIO as fIO
#from ....base import Error as Err
from ....base import plotting as cP
from ....base import utilities as cF
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

        # if systems is None:
        #     return

        self.system = systems

        kwargs['channels_per_system'] = kwargs.get('channels_per_system', self.nTimes)
        # kwargs['components_per_channel'] = kwargs.get('components_per_channel', self.system[0].components)
        kwargs['units'] = r"$\frac{V}{m^{2}}$"

        # Data Class containing xyz and channel values
        super().__init__(**kwargs)

        # StatArray of Transmitter loops
        self.transmitter = kwargs.get('transmitter', None)
        # StatArray of Receiever loops
        self.receiver = kwargs.get('receiver', None)
        # Loop Offsets
        self.loopOffset = kwargs.get('loopOffset', None)

        self.channelNames = kwargs.get('channel_names', None)

    @Data.channelNames.setter
    def channelNames(self, values):
        if values is None:
            self._channelNames = []
            for i in range(self.nSystems):
                # Set the channel names
                for ic in range(self.n_components):
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
                self._loopOffset = deepcopy(values)
            else:
                self._loopOffset = StatArray.StatArray(values, "Loop Offset")

    @property
    def n_components(self):
        if self.system is None:
            return 0
        return self.system[0].n_components

    @property
    def nTimes(self):
        return self.channels_per_system

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

        if values is None:
            self._system = None
            self._channels_per_system = 0
            return

        if isinstance(values, (str, TdemSystem)):
            values = [values]
        nSystems = len(values)
        # Make sure that list contains strings or TdemSystem classes
        assert all([isinstance(x, (str, TdemSystem)) for x in values]), TypeError("system must be str or list of either str or geobipy.TdemSystem")

        self._system = [None] * nSystems

        for i, s in enumerate(values):
            if isinstance(s, str):
                self._system[i] = TdemSystem.read(s)
            else:
                self._system[i] = s

        self._channels_per_system = np.asarray([s.nTimes for s in self.system])

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

    def _component_indices(self, component=0, system=0):
        assert component < self.n_components, ValueError("component must be < {}".format(self.n_components))
        return np.s_[((self.nTimes*component)+(system*self.nChannels))[0]:(self.nTimes*(component+1)+(system*self.nChannels))[0]]

    @classmethod
    def read_csv(cls, dataFilename, systemFilename):
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

        self = cls(systems=systemFilename)

        self._nPoints, iC, iR, iT, iOffset, iD, iS = self._csv_channels(dataFilename)

        # Get all readable column indices for the first file.
        channels = iC[0] + iR[0] + iT[0] + iOffset[0] + iD[0]
        if not iS[0] is None:
            channels += iS[0]

        # Read in the columns from the first data file
        try:
            df = read_csv(dataFilename[0], usecols=channels, skipinitialspace = True)
        except:
            df = read_csv(dataFilename[0], usecols=channels, delim_whitespace=True, skipinitialspace = True)
        df = df.replace('NaN',np.nan)

        # Assign columns to variables
        self.lineNumber = df[iC[0][0]].values
        self.fiducial = df[iC[0][1]].values
        self.x = df[iC[0][2]].values
        self.y = df[iC[0][3]].values
        self.z = df[iC[0][4]].values
        self.elevation = df[iC[0][5]].values

        # Assign the orientations of the acquisistion loops
        self.receiver = None
        for i in range(self.nPoints):
            self.receiver[i] = CircularLoop(z=self.z[i], pitch=df[iR[0][0]].values[i], roll=df[iR[0][1]].values[i], yaw=df[iR[0][2]].values[i], radius=self.system[0].loopRadius())

        self.transmitter = None
        for i in range(self.nPoints):
            self.transmitter[i] = CircularLoop(z=self.z[i], pitch=df[iT[0][0]].values[i], roll=df[iT[0][1]].values[i], yaw=df[iT[0][2]].values[i], radius=self.system[0].loopRadius())

        self.loopOffset = df[iOffset[0]].values


        # Get the data values
        iSys = self._systemIndices(0)

        self.data[:, iSys] = df[iD[0]].values
        # If the data error columns are given, assign them

        if (iS[0] is None):
            self.std[:, iSys] = 0.1 * self.data[:, iSys]
        else:
            self.std[:, iSys] = df[iS[0]].values

        # Read in the data for the other systems.  Only read in the data and, if available, the errors
        for i in range(1, self.nSystems):
            # Assign the columns to read
            channels = iD[i]
            if (not iS[i] is None): # Append the error columns if they are available
                channels += iS[i]

            # Read the columns
            try:
                df = read_csv(dataFilename[i], usecols=channels, skipinitialspace = True)
            except:
                df = read_csv(dataFilename[i], usecols=channels, delim_whitespace=True, skipinitialspace = True)
            df = df.replace('NaN',np.nan)

            # Assign the data
            iSys = self._systemIndices(i)

            self.data[:, iSys] = df[iD[i]].values
            if (iS[i] is None):
                self.std[:, iSys] = 0.1 * self.data[:, iSys]
            else:
                self.std[:, iSys] = df[iS[i]].values


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

    def csv_channels(self, data_filename):

        self._nPoints, self._iC, self._iR, self._iT, self._iOffset, self._iD, self._iS = self._csv_channels(data_filename)

        self._channels = []
        for i in range(self.nSystems):
            channels = self._iC[i] + self._iR[i] + self._iT[i] + self._iOffset[i] + self._iD[i]
            if not self._iS[i] is None:
                channels += self._iS[i]
            self._channels.append(channels)

        return self._channels


    def _csv_channels(self, data_filename):
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

        location_channels = []
        rLoop_channels = []
        tLoop_channels = []
        offset_channels = []
        off_channels = []
        off_error_channels = []

        # First get the number of points in each data file. They should be equal.
        nPoints = self._csv_n_points(data_filename)

        rloop_names = ('rxpitch', 'rxroll', 'rxyaw')
        tloop_names = ('txpitch', 'txroll', 'txyaw')
        offset_names = ('txrx_dx', 'txrx_dy', 'txrx_dz')

        if isinstance(data_filename, str):
            data_filename = [data_filename]

        for k, f in enumerate(data_filename):
            # Get the column headers of the data file
            channels = fIO.get_column_name(f)
            nChannels = len(channels)

            ixyz, ilf = super()._csv_channels(f)
            loc = ilf + ixyz

            nr = nt = no = 0
            rloop = [None]*3
            tloop = [None]*3
            offset = [None]*3
            on = []
            on_error = []
            off = []
            off_error = []

            # Check for each aspect of the data file and the number of columns
            for channel in channels:
                cTmp = channel.lower()
                if cTmp in rloop_names:
                    nr +=1
                    rloop[rloop_names.index(cTmp)] = channel
                elif cTmp in tloop_names:
                    nt += 1
                    tloop[tloop_names.index(cTmp)] = channel
                elif cTmp in offset_names:
                    no += 1
                    offset[offset_names.index(cTmp)] = channel
                elif "on[" in cTmp:
                    on.append(channel)
                elif "onerr[" in cTmp:
                    on_error.append(channel)
                elif "off[" in cTmp:
                    off.append(channel)
                elif "offerr[" in cTmp:
                    off_error.append(channel)

            assert nr == 3, Exception('Must have all three RxPitch, RxRoll, and RxYaw headers in data file {} if reciever orientation is specified. \n {}'.format(f, self.fileInformation()))
            assert nt == 3, Exception('Must have all three TxPitch, TxRoll, and TxYaw headers in data file {} if transmitter orientation is specified. \n {}'.format(f, self.fileInformation()))
            assert no == 3, Exception('Must have all three txrx_dx, txrx_dy, and txrx_dz headers in data file {} if transmitter-reciever loop separation is specified. \n {}'.format(f, self.fileInformation()))

            assert len(off) == self.system[k].windows.centre.size, Exception("Number of Off time columns {} in {} does not match number of times {} in system file {}. \n {}".format(len(off), f, self.system[k].fileName, self.system[k].windows.centre.size, self.fileInformation()))

            if len(off_error) > 0:
                assert len(off_error) == len(off), Exception("Number of Off time standard deviation estimates does not match number of Off time data columns in file {}. \n {}".format(f, self.fileInformation()))
            else:
                off_error = None

            location_channels.append(loc)
            rLoop_channels.append(rloop)
            tLoop_channels.append(tloop)
            offset_channels.append(offset)
            off_channels.append(off)
            off_error_channels.append(off_error)

        return nPoints, location_channels, rLoop_channels, tLoop_channels, offset_channels, off_channels, off_error_channels

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

    @property
    def nSystems(self):
        return np.size(self.system)

    @property
    def nChannelsPerSystem(self):
        return np.asarray([s.nwindows() for s in self.system])

    def _read_record(self, record=None):
        """Reads a single data point from the data file.

        FdemData.__initLineByLineRead() must have already been run.

        """
        endOfFile = False
        dfs = []
        for i in range(self.nSystems):
            try:
                df = self._file[i].get_chunk()
                df = df.replace('NaN', np.nan)
                dfs.append(df)
            except:
                self._file[i].close()
                endOfFile = True

        if endOfFile:
            return None

        D = np.squeeze(np.hstack([dfs[i][self._iD[i]].values for i in range(self.nSystems)]))


        if self._iS[0] is None:
            S = 0.1 * D
        else:
            S = np.squeeze(np.hstack([dfs[i][self._iS[i]].values for i in range(self.nSystems)]))

        data = np.squeeze(dfs[0][self._iC[0]].values)

        rloop = np.squeeze(dfs[0][self._iR[0]].values)
        R = CircularLoop(z=data[4], pitch=rloop[0], roll=rloop[1], yaw=rloop[2], radius=self.system[0].loopRadius())
        tloop = np.squeeze(dfs[0][self._iT[0]].values)
        T = CircularLoop(z=data[4], pitch=tloop[0], roll=tloop[1], yaw=tloop[2], radius=self.system[0].loopRadius())

        loopOffset = np.squeeze(dfs[0][self._iOffset[0]].values)

        out = TdemDataPoint(x=data[2], y=data[3], z=data[4], elevation=data[5],
                            data=D, std=S,
                            system=self.system,
                            transmitter_loop=T, receiver_loop=R, loopOffset=loopOffset,
                            lineNumber=data[0], fiducial=data[1])

        return out

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

    @property
    def datapoint_type(self):
        return TdemDataPoint

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
        return TdemDataPoint(self.x[i], self.y[i], self.z[i], self.elevation[i],
                             self.data[i, :], self.std[i, :], self.predictedData[i, :],
                             self.system,
                             self.transmitter[i], self.receiver[i], self.loopOffset[i, :],
                             self.lineNumber[i], self.fiducial[i])


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
            line.data[:, self._systemIndices(i)].plot(x=x, **kwargs)


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


        super.fromHdf(grp)

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
                sfnTmp = [s.filename for s in self.system]
            else:
                sfnTmp = None
            systemFilename = world.bcast(sfnTmp, root=root)
            nSystems = len(systemFilename)

            system = [None] * nSystems
            for i in range(nSystems):
                system[i] = TdemSystem.read(systemFilename[i])

        return system

    def Bcast(self, world, root=0, system=None):
        """Broadcast the TdemData using MPI

        Parameters
        ----------

        """

        # nPoints = myMPI.Bcast(self.nPoints, world, root=root)
        # nTimes = myMPI.Bcast(self.nTimes, world, root=root)

        systems = self._BcastSystem(world, root=root, system=system)

        # Instantiate a new Time Domain Data set on each worker
        this = TdemData(systems)

        # Broadcast the Data point id, line numbers and elevations
        this._fiducial = self.fiducial.Bcast(world, root=root)
        this._lineNumber = self.lineNumber.Bcast(world, root=root)
        this._x = self.x.Bcast(world, root=root)
        this._y = self.y.Bcast(world, root=root)
        this._z = self.z.Bcast(world, root=root)
        this._elevation = self.elevation.Bcast(world, root=root)
        this._data = self.data.Bcast(world, root=root)
        this._std = self.std.Bcast(world, root=root)
        this._predictedData = self.predictedData.Bcast(world, root=root)

        # Broadcast the Transmitter Loops.
        if (world.rank == 0):
            lTmp = [str(self.transmitter[i]) for i in range(self.nPoints)]
        else:
            lTmp = []
        lTmp = myMPI.Bcast_list(lTmp, world)
        if (world.rank == 0):
            this.transmitter = self.transmitter
        else:
            for i in range(this.nPoints):
                this.transmitter[i] = eval(cF.safeEval(lTmp[i]))

        # Broadcast the Reciever Loops.
        if (world.rank == 0):
            lTmp = [str(self.receiver[i]) for i in range(self.nPoints)]
        else:
            lTmp = []
        lTmp = myMPI.Bcast_list(lTmp, world)
        if (world.rank == 0):
            this.receiver = self.receiver
        else:
            for i in range(this.nPoints):
                this.receiver[i] = eval(cF.safeEval(lTmp[i]))

        # this.iActive = this.getActiveChannels()

        return this


    def Scatterv(self, starts, chunks, world, root=0, system=None):
        """ Scatterv the TdemData using MPI """

        # nTimes = myMPI.Bcast(self.nTimes, world, root=root)

        systems = self._BcastSystem(world, root=root, system=system)

        # Instantiate a new Time Domain Data set on each worker
        this = TdemData(systems)

        # Broadcast the Data point id, line numbers and elevations
        this._fiducial = self.fiducial.Scatterv(starts, chunks, world, root=root)
        this._lineNumber = self.lineNumber.Scatterv(starts, chunks, world, root=root)
        this._x = self.x.Scatterv(starts, chunks, world, root=root)
        this._y = self.y.Scatterv(starts, chunks, world, root=root)
        this._z = self.z.Scatterv(starts, chunks, world, root=root)
        this._elevation = self.elevation.Scatterv(starts, chunks, world, root=root)
        this._data = self.data.Scatterv(starts, chunks, world, root=root)
        this._std = self.std.Scatterv(starts, chunks, world, root=root)
        this._predictedData = self.predictedData.Scatterv(starts, chunks, world, root=root)

        # Scatterv the Transmitter Loops.
        if (world.rank == 0):
            lTmp = [str(self.transmitter[i]) for i in range(self.nPoints)]
        else:
            lTmp = []
        lTmp = myMPI.Scatterv_list(lTmp, starts, chunks, world)
        if (world.rank == 0):
            this.transmitter = self.transmitter[:chunks[0]]
        else:
            for i in range(this.nPoints):
                this.transmitter[i] = eval(lTmp[i])

        # Scatterv the Reciever Loops.
        if (world.rank == 0):
            lTmp = [str(self.receiver[i]) for i in range(self.nPoints)]
        else:
            lTmp = []
        lTmp = myMPI.Scatterv_list(lTmp, starts, chunks, world)
        if (world.rank == 0):
            this.receiver = self.receiver[:chunks[0]]
        else:
            for i in range(this.nPoints):
                this.receiver[i] = eval(lTmp[i])

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
