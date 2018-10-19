"""
.. module:: StatArray
   :platform: Unix, Windows
   :synopsis: Time domain data set

.. moduleauthor:: Leon Foks

"""
from ...pointcloud.PointCloud3D import PointCloud3D
from .Data import DataSet
from .Data import Data
from ..datapoint.TdemDataPoint import TdemDataPoint
from ....classes.core.StatArray import StatArray
from ...system.CircularLoop import CircularLoop
from ....base.customFunctions import safeEval
from ...system.TdemSystem import TdemSystem

import numpy as np
from ....base import fileIO as fIO
#from ....base import Error as Err
from ....base import customPlots as cP
from ....base import MPI as myMPI
import matplotlib.pyplot as plt


class TdemData(PointCloud3D):
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

    def __init__(self, nPoints=1, nTimes=[1], nSystems=1, systemFnames=None):
        """ Initialize the TDEM data """
        nt = np.asarray(nTimes)
        assert nt.size == nSystems, "length of nTimes must equal nSys"
        # PointCloud3D of x y z locations
        PointCloud3D.__init__(self, nPoints)
        # Number of systems
        self.nSystems = np.int32(nSystems)
        if (not systemFnames is None):
            assert isinstance(systemFnames, (str, list)), TypeError("systemFnames must be str or list of str")
            if (isinstance(systemFnames, str)):
                self.nSystems = 1
            else:
                self.nSystems = len(systemFnames)

        # Number of channels
        self.nTimes = np.int32(nTimes)
        # ndarray for the number of system types
        self.set = np.ndarray(nSystems, dtype=DataSet)
        # DataSet array for each system
        for i in range(nSystems):
            self.set[i] = DataSet(self.N, nTimes[i], r'$\frac{V}{m^{2}}$')

        # StatArray of the line number for flight line data
        self.line = StatArray(self.N, 'Line Number', 'N/A')
        # StatArray of the id number
        self.id = np.zeros(self.N, dtype=np.int32)
        # StatArray of the elevation
        self.elevation = StatArray(self.N, 'Elevation', 'm')
        # StatArray of Transmitter loops
        self.T = StatArray(self.N, 'Transmitter Loops', dtype=CircularLoop)
        # StatArray of Receiever loops
        self.R = StatArray(self.N, 'Receiver Loops', dtype=CircularLoop)


    def read(self, dataFname, systemFname):
        """Reads the data and system parameters from file

        Parameters
        ----------
        dataFname : str or list of str
            Time domain data file names
        systemFname : str or list of str
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

        Off[0] to Off[nWindows]  (with the number and brackets)
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

        OffErr[0] to ErrOff[nWindows]
            Error estimates for the data


        See Also
        --------
        INFORMATION ON TD SYSTEMS


        """

        # Get the number of systems to use
        if (isinstance(dataFname,str)):
            dataFname=[dataFname]
        if (isinstance(systemFname,str)):
            systemFname=[systemFname]

        # Save the system file names
        self.sysFname = systemFname

        nSys = len(systemFname)
        # Read in the C++ system
        self.readSystemFile(systemFname)

        # Check the data files against the system files, make sure they have
        # the correct number of columns
        nTimes = np.zeros(nSys, dtype=int)
        nPoints = np.zeros(nSys, dtype=int)
        offData = [None]*nSys
        offErr = [None]*nSys
        allReadin = [None]*nSys
        for i in range(nSys):
            # Get the number of windows from the system file
            nTimes[i] = self.sys[i].nwindows()

            # Get the column indices for the items in the class
            base, rLoop, tLoop, offData[i], offErr[i]=self.getColumnIDs(dataFname[i])

            # Concatenate the
            allReadin[i] = base
            if (not rLoop is None):
                allReadin[i] = np.append(allReadin[i],rLoop)
            if (not tLoop is None):
                allReadin[i] = np.append(allReadin[i],tLoop)
            allReadin[i] = np.append(allReadin[i],offData[i])
            if (not offErr[i] is None):
                allReadin[i] = np.append(allReadin[i],offErr[i])

            # Check that the number of data columns matches the times given in the system file
            assert offData[i].size == nTimes[i], 'The number of off time data columns in '+dataFname[i]+' does not match the number of windows '+str(nTimes[i])+' given in the system file '+systemFname[i]

            # Get the number of points
            nPoints[i] = fIO.getNlines(dataFname[i], 1)

        # Check that all the data files have the same number of points
        if (nSys > 1):
            for i in range(1,nSys):
                assert nPoints[i] == nPoints[0], 'Number of data points '+str(nPoints[i])+' in file '+dataFname[i]+' does not match '+dataFname[0]+' '+str(nPoints[0])
        nPoints = nPoints[0]

        # Read in the columns from the first data file
        tmp = fIO.read_columns(dataFname[0], allReadin[0], 1, nPoints)

        # Initialize the EMData Class
        self.__init__(nPoints, nTimes, nSys)
        # Assign columns to variables
        self.line[:] = tmp[:,0]
        self.id[:] = tmp[:,1]
        self.x[:] = tmp[:,2]
        self.y[:] = tmp[:,3]
        self.elevation[:] = tmp[:,5]
        self.z[:] = tmp[:,4]

        i0 = 6
        # Assign the orientations of the acquisistion loops
        if (not rLoop is None):
            for i in range(nPoints):
                self.R[i] = CircularLoop(z=self.z[i], pitch=tmp[i, i0], roll=tmp[i, i0+1], yaw=tmp[i, i0+2], radius=self.sys[0].loopRadius())
            i0 += 3
        else:
            for i in range(nPoints):
                self.R[i] = CircularLoop(z=self.z[i], radius=self.sys[0].loopRadius())

        if (not tLoop is None):
            for i in range(nPoints):
                self.T[i] = CircularLoop(z=self.z[i], pitch=tmp[i, i0], roll=tmp[i, i0+1], yaw=tmp[i, i0+2], radius=self.sys[0].loopRadius())
            i0 += 3
        else:
            for i in range(nPoints):
                self.T[i] = CircularLoop(z=self.z[i], radius=self.sys[0].loopRadius())

        # Create column indexers for the data and errors
        i1 = i0 + nTimes[0]
        iData = np.arange(i0, i1)

        # Get the data values
        self.set[0].D[:, :] = tmp[:, iData]
        # If the data error columns are given, read them in
        if (not offErr[0] is None):
            i2 = i1 + nTimes[0]
            iStd = np.arange(i1, i2)
            self.set[0].Std[:, :] = tmp[:, iStd]

        # Read in the data for the other systems.  Only read in the data and, if available, the errors
        for i in range(1, self.nSystems):
            # Assign the columns to read
            readColumns = offData[i]
            if (not offErr[i] is None): # Append the error columns if they are available
                readColumns = np.append(readColumns,offErr[i])
            # Read the columns
            tmp = fIO.read_columns(dataFname[i], readColumns, 1, nPoints)
            # Assign the data
            self.set[i].D[:, :] = tmp[:, :nTimes[i]]
            if (not offErr[i] is None):
                self.set[i].Std[:, :] = tmp[:, nTimes[i]:]
                                

    def readSystemFile(self, systemFname):
        """ Reads in the C++ system handler using the system file name """

        nSys = len(systemFname)
        self.sys = np.ndarray(nSys, dtype=TdemSystem)

        for i in range(nSys):
            self.sys[i] = TdemSystem()
            self.sys[i].read(systemFname[i])


    def getColumnIDs(self,dataFname):
        """ Get the column indices for the TdemData file """

        # Get the column headers of the data file
        channels = fIO.getHeaderNames(dataFname)
        channels = [channel.lower() for channel in channels]
        # Get the number of errors channel
        nErr = sum(['offerr[' in channel for channel in channels])
        # Get the number of off time channels
        nOff = sum(['off[' in channel for channel in channels])

        assert nOff > 0, self.fileFormatError(dataFname)

        # To grab the EM data, skip the following header names. (More can be added to this)
        # Get the x,y,z,altitude,elevation,line, and point id columns
        base = np.zeros(6,dtype=np.int32)
        i = -1
        for j, channel in enumerate(channels):
            if (channel in ['n', 'x','northing']):
                i += 1
                base[2] = j
            elif (channel in ['e', 'y', 'easting']):
                i += 1
                base[3] = j
            elif (channel in ['alt', 'laser', 'bheight']):
                i += 1
                base[4] = j
            elif(channel in ['z', 'dtm','dem_elev','dem_np','topo']):
                i += 1
                base[5] = j
            elif(channel in ['line']):
                i += 1
                base[0] = j
            elif(channel in ['id', 'fid']):
                i += 1
                base[1] = j
        assert i == 5, self.fileFormatError(dataFname)

        # Get the loop configurations if they are in the file
        rLoop = None
        if ('rpitch' in channels):
            rLoop = np.zeros(3,dtype=np.int32)
            i=-1
            for j, channel in enumerate(channels):
                if (channel == 'rpitch'):
                    i += 1
                    rLoop[0] = j
                if (channel == 'rroll'):
                    i += 1
                    rLoop[1] = j
                if (channel == 'ryaw'):
                    i += 1
                    rLoop[2] = j
            # Make sure all three are given
            assert i == 2, 'The pitch, roll, and yaw must all be given for the receiver loop'

        tLoop = None
        if ('tpitch' in channels):
            tLoop = np.zeros(3,dtype=np.int32)
            i=-1
            for j, channel in enumerate(channels):
                if (channel == 'tpitch'):
                    i += 1
                    tLoop[0] = j
                if (channel == 'troll'):
                    i += 1
                    tLoop[1] = j
                if (channel == 'tyaw'):
                    i += 1
                    tLoop[2] = j
            # Make sure all three are given
            assert i == 2, 'The pitch, roll, and yaw must all be given for the transmitter loop'

        # Get the column indices for the on times
        onData = None
        # Get the column indices for the ontime errors
        onErr = None
        # Get the column indices for the off times
        offData = np.zeros(nOff,dtype=np.int32)
        i=-1
        for j, channel in enumerate(channels):
            if ('off[' in channel):
                i += 1
                offData[i]=j


        # Get the column indices for the offtime errors
        offErr = None
        if (nErr > 0):
            assert nErr == nOff, 'Number of error columns does not equal the number of windows in the system file\n'+self.fileFormatError(dataFname)
            offErr = np.zeros(nErr,dtype=np.int32)
            i=-1
            for j, channel in enumerate(channels):
                if ('offerr[' in channel):
                    i += 1
                    offErr[i]=j

        return base, rLoop, tLoop, offData, offErr

    def getChannel(self, system, channel):
        """ Gets the data in the specified channel """
        assert system >= 0 and system < self.nSystems, ValueError('System must be less than '+str(self.nSystems))
        assert channel >= 0 and channel < self.nTimes[system], 'Requested channel must be less than '+str(self.nTimes[system])

        tmp = StatArray(self.set[system].D[:, channel], 'Data at time '+str(self.sys[system].windows.centre[channel])+' s', self.set[system].D.units)
        if (np.all(np.isnan(tmp))):
            print('Warning: All values in channels are NaN')
        return tmp

    def estimateAdditiveError(self):
        """ Uses the late times after 1ms to estimate the additive errors and error bounds in the data. """
        for i in range(self.nSystems):
            t = self.times(i)
            i1ms = t.searchsorted(1e-3)
            if (i1ms < t.size):
                print(i1ms)
                print(self.set[i].D.shape[1])

                D=self.set[i].D[:,i1ms:self.set[i].D.shape[1]]
                print(self.set[i].D.shape)
                print(D,D.shape)
                s=np.nanstd(D)
                print(
                'System '+str(i) +'\n'
                '  Minimum at times > 1ms: '+ str(np.nanmin(D)) +'\n'
                '  Maximum at time  = 1ms: '+ str(np.nanmax(D[:,0])) +'\n'
                '  Median:  '+ str(np.nanmedian(D)) +'\n'
                '  Mean:    '+ str(np.nanmean(D)) +'\n'
                '  Std:     '+ str(s) +'\n'
                '  4Std:    '+ str(4.0*s) +'\n')
            else:
                print(
                'System '+str(i) + 'Has no times after 1ms')

    def getDataPoint(self, i):
        """ Get the ith data point from the data set """
        assert 0 <= i < self.N, ValueError("Requested data point must have index (0, "+str(self.N) + ']')
        D = [self.set[j].D[i, :] for j in range(self.nSystems)]
        S = [self.set[j].Std[i, :] for j in range(self.nSystems)]
        this = TdemDataPoint(self.x[i], self.y[i], self.z[i], self.elevation[i], D, S, self.sys, self.T[i],self.R[i])
        return this

    def getLine(self, line):
        """ Gets the data in the given line number """
        i = np.where(self.line == line)[0]
        return self[i]

    def times(self,system=0):
        """ Obtain the times from the system file """
        assert system >= 0 and system < self.nSystems, ValueError('system must be in (0, '+str(self.nSystems)+']')
        return StatArray(self.sys[system].windows.centre,'Time','ms')


    def __getitem__(self, i):
        """ Define item getter for TdemData """
        tmp = TdemData(np.size(i), self.nTimes, self.nSystems)
        tmp.x[:] = self.x[i]
        tmp.y[:] = self.y[i]
        tmp.z[:] = self.z[i]
        tmp.line[:] = self.line[i]
        tmp.id[:] = self.id[i]
        tmp.elevation[:] = self.elevation[i]
        tmp.T[:] = self.T[i]
        tmp.R[:] = self.R[i]
        tmp.sys = np.ndarray(self.nSystems, dtype=TDAEMSystem)
        for j in range(self.nSystems):
            tmp.set[j].D[:, :] = self.set[j].D[i, :]
            tmp.set[j].D[:, :] = self.set[j].Std[i, :]
            tmp.sys[j] = self.sys[j]
        return tmp


    def fileFormatError(self,dataFname):
        s =('\nError Reading from file '+dataFname+'\n'
            'The data columns are read in according to the column names in the first line \n'
            'The header line should contain at least the following column names. Extra columns may exist, but will be ignored \n'
            'In this description, the column name or its alternatives (denoted in brackets) are given followed by what the name represents \n'
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
            'z or dtm or dem_elev or dem_np \n'
            '    Elevation of the ground at the data point\n'
            'alt or laser or bheight \n'
            '    Altitude of the transmitter coil\n'
            'Off[0] to Off[nWindows]  - with the number and brackets\n'
            '    The measurements for each time specified in the accompanying system file under Receiver Window Times'
            '\n'
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
            'OffErr[0] to ErrOff[nWindows]\n'
            '    Error estimates for the data')
        return s


    def mapChannel(self, channel, system=0, *args, **kwargs):
        """ Create a map of the specified data channel """

        tmp = self.getChannel(system, channel)
        kwargs['c'] = tmp

        self.mapPlot(*args, **kwargs)

        cP.title(tmp.name)


    def plotWaveform(self,**kwargs):
        for i in range(self.nSystems):
            plt.subplot(2, 1, i + 1)
            plt.plot(self.sys[i].waveform.time, self.sys[i].waveform.current, **kwargs)
            if (i == self.nSystems-1): cP.xlabel('Time (s)')
            cP.ylabel('Normalized Current (A)')
            plt.margins(0.1, 0.1)

    def pcolor(self,system=0,**kwargs):
        """ Plot the data in the given system as a 2D array
        """
        D=self.set[system]
        times=self.times(system)
        y=StatArray(np.arange(D.nPoints),'DataPoint Index')
        ax = D.D.pcolor(x=times,y=y,**kwargs)
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

        if (not "log" in kwargs):
            kwargs["log"] = 10.0
        return Data.scatter2D(self, **kwargs)

    def Bcast(self, world):
        """ Broadcast the TdemData using MPI """
        pc3d = None
        pc3d = PointCloud3D.Bcast(self, world)
        nTimes = myMPI.Bcast(self.nTimes, world)
        nSystems = myMPI.Bcast(self.nSystems, world)
        
        # Instantiate a new Time Domain Data set on each worker
        this = TdemData(pc3d.N, nTimes, nSystems)

        # Assign the PointCloud Variables
        this.x = pc3d.x
        this.y = pc3d.y
        this.z = pc3d.z

        # On each worker, create a small instantiation of the ndarray of data sets in the TdemData class
        # This allows the broadcast to each worker. Without this setup, each
        # worker cannot see tmp[i].Bcast because it doesn't exist.
        if (world.rank == 0):
            tmp = self.set
        else:
            tmp = np.zeros(this.nSystems, dtype=DataSet)
            for i in range(this.nSystems):
                tmp[i] = DataSet()

        # Each DataSet has been instantiated within this. Broadcast the
        # contents of the Masters self.set[0:nSystems]
        for i in range(this.nSystems):
            this.set[i] = tmp[i].Bcast(world)

        # Broadcast the Data point id, line numbers and elevations
        this.id = myMPI.Bcast(self.id, world)
        this.line = self.line.Bcast(world)
        this.elevation = self.elevation.Bcast(world)

        # Since the Time Domain EM Systems are C++ objects on the back end, I can't Broadcast them through C++ (Currently a C++ Noob)
        # So instead, Broadcast the list of system file names saved in the TdemData Class and read the system files in on each worker.
        # This is cumbersome, but only done once at the beginning of the MPI
        # code.
        strTmp = []
        for i in range(this.nSystems):
            if (world.rank == 0):
                strTmp.append(self.sysFname[i])
            else:
                strTmp.append('')

        systemFname = []
        for i in range(this.nSystems):
            systemFname.append(myMPI.Bcast(strTmp[i], world))
        # Read the same system files on each worker
        this.readSystemFile(systemFname)

        # Broadcast the Transmitter Loops.
        if (world.rank == 0):
            lTmp = [None] * self.N
            for i in range(self.N):
                lTmp[i] = str(self.T[i])
        else:
            lTmp = []
        lTmp = myMPI.Bcast_list(lTmp, world)
        if (world.rank == 0):
            this.T = self.T
        else:
            for i in range(this.N):
                this.T[i] = eval(safeEval(lTmp[i]))

        # Broadcast the Reciever Loops.
        if (world.rank == 0):
            lTmp = [None] * self.N
            for i in range(self.N):
                lTmp[i] = str(self.R[i])
        else:
            lTmp = []
        lTmp = myMPI.Bcast_list(lTmp, world)
        if (world.rank == 0):
            this.R = self.R
        else:
            for i in range(this.N):
                this.R[i] = eval(safeEval(lTmp[i]))

        return this

    def Scatterv(self, myStart, myChunk, world):
        """ Scatterv the TdemData using MPI """
#    myMPI.print("Inside TdemData.Scatterv")
        pc3d = None
        pc3d = PointCloud3D.Scatterv(self, myStart, myChunk, world)
        nTimes = myMPI.Bcast(self.nTimes, world)
        nSys = myMPI.Bcast(self.nSystems, world)
        # Instantiate a new reduced size Time Domain Data set on each worker
        this = TdemData(pc3d.N, nTimes, nSys[0])
        # Assign the PointCloud Variables
        this.x = pc3d.x
        this.y = pc3d.y
        this.z = pc3d.z

        # On each worker, create a small instantiation of the ndarray of data sets in the TdemData class
        # This allows the scatter to each worker. Without this setup, each
        # worker cannot see tmp[i].Scatterv because it doesn't exist.
        if (world.rank == 0):
            tmp = self.set
        else:
            tmp = np.zeros(this.nSystems, dtype=DataSet)
            for i in range(this.nSystems):
                tmp[i] = DataSet()

        # Each DataSet has been instantiated within this. Scatterv the contents
        # of the Masters self.set[0:nSystems]
        for i in range(this.nSystems):
            this.set[i] = tmp[i].Scatterv(myStart, myChunk, world)

        # Scatterv the Data point id, line numbers and elevations
        this.id = self.id.Scatterv(myStart, myChunk, world)
        this.line = self.line.Scatterv(myStart, myChunk, world)
        this.elevation = self.elevation.Scatterv(myStart, myChunk, world)

        # Since the Time Domain EM Systems are C++ objects on the back end, I can't Broadcast them through C++ (Currently a C++ Noob)
        # So instead, Broadcast the list of system file names saved in the TdemData Class and read the system files in on each worker.
        # This is cumbersome, but only done once at the beginning of the MPI
        # code.
        strTmp = []
        for i in range(this.nSystems):
            if (world.rank == 0):
                strTmp.append(self.sysFname[i])
            else:
                strTmp.append('')

        systemFname = []
        for i in range(this.nSystems):
            systemFname.append(myMPI.Bcast(strTmp[i], world))
        # Read the same system files on each worker
        this.readSystemFile(systemFname)

        # Scatterv the Transmitter Loops.
        if (world.rank == 0):
            lTmp = [None] * self.N
            for i in range(self.N):
                lTmp[i] = str(self.T[i])
        else:
            lTmp = []
        lTmp = myMPI.Scatterv_list(lTmp, myStart, myChunk, world)
        if (world.rank == 0):
            this.T[:] = self.T[:myChunk[0]]
        else:
            for i in range(this.N):
                this.T[i] = eval(lTmp[i])

        # Scatterv the Reciever Loops.
        if (world.rank == 0):
            lTmp = [None] * self.N
            for i in range(self.N):
                lTmp[i] = str(self.R[i])
        else:
            lTmp = []
        lTmp = myMPI.Scatterv_list(lTmp, myStart, myChunk, world)
        if (world.rank == 0):
            this.R[:] = self.R[:myChunk[0]]
        else:
            for i in range(this.N):
                this.R[i] = eval(lTmp[i])

        return this
