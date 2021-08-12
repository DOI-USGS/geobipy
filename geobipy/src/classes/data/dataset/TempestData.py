"""
.. module:: StatArray
   :platform: Unix, Windows
   :synopsis: Time domain data set

.. moduleauthor:: Leon Foks

"""
from ...pointcloud.PointCloud3D import PointCloud3D
from .TdemData import TdemData
from ..datapoint.Tempest_datapoint import Tempest_datapoint
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
import h5py


class TempestData(TdemData):
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

        # Data Class containing xyz and channel values
        super().__init__(systems, **kwargs)

        self.primary_field = kwargs.get('primary_field', None)
        self.secondary_field = kwargs.get('secondary_field', None)


    @property
    def components_per_channel(self):
        return self.system[0].components

    @TdemData.data.getter
    def data(self):
        for ic in range(self.n_components):
            slic = self._component_indices(ic, 0)
            self._data[:, slic] = self.primary_field[:, ic][:, None] + self.secondary_field[:, slic]
        return self._data

    @property
    def nChannels(self):
        return 2 * self.system[0].nTimes

    @TdemData.predictedData.getter
    def predictedData(self):
        for ic in range(self.n_components):
            slic = self._component_indices(ic, 0)
            self._predictedData[:, slic] = self.primary_field[:, ic][:, None] + self.secondary_field[:, slic]
        return self._predictedData

    def append(self, other):

        super().append(self, other)

        self.primary_field = np.hstack(self.primary_field, other.primary_field)

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

        if (isinstance(systemFilename, str)):
            systemFilename = [systemFilename]

        nSystems = len(systemFilename)

        self.system = systemFilename

        with h5py.File(dataFilename, 'r') as f:
            gdf = f['linedata']

            self.__init__(lineNumber=np.asarray(gdf['Line']),
                        fiducial=np.asarray(gdf['Fiducial']),
                        x=np.asarray(gdf['Easting']),
                        y=np.asarray(gdf['Northing']),
                        z=np.asarray(gdf['Tx_Height']),
                        elevation=np.asarray(gdf['DTM']),
                        systems=self.system)

            # Assign the orientations of the acquisistion loops
            pitch = np.asarray(gdf['Tx_Pitch'])
            roll = np.asarray(gdf['Tx_Roll'])
            yaw = np.asarray(gdf['Tx_Yaw'])
            self.transmitter = None
            for i in range(self.nPoints):
                self.transmitter[i] = CircularLoop(z=self.z[i], pitch=pitch[i], roll=roll[i], yaw=yaw[i], radius=self.system[0].loopRadius())

            pitch = np.asarray(gdf['Rx_Pitch'])
            roll = np.asarray(gdf['Rx_Roll'])
            yaw = np.asarray(gdf['Rx_Yaw'])
            self.receiver = None
            for i in range(self.nPoints):
                self.receiver[i] = CircularLoop(z=self.z[i], pitch=pitch[i], roll=roll[i], yaw=yaw[i], radius=self.system[0].loopRadius())

            self.loopOffset = np.vstack([np.asarray(gdf['HSep_GPS']), np.asarray(gdf['TSep_GPS']), np.asarray(gdf['VSep_GPS'])]).T

            self.primary_field = np.vstack([np.asarray(gdf['X_PrimaryField']), np.asarray(gdf['Z_PrimaryField'])]).T
            self.secondary_field = np.hstack([np.asarray(gdf['EMX_NonHPRG']).T, np.asarray(gdf['EMZ_NonHPRG']).T])

            self.std = 0.1 * self.data

        self.check()

        return self

    def _initialize_sequential_reading(self, data_filename, system_filename):
        """Special function to initialize a file for reading data points one at a time.

        Parameters
        ----------
        dataFileName : str
            Path to the data file
        systemFname : str
            Path to the system file

        """

        # Read in the EM System file
        self.system = system_filename

        self._open_data_files(data_filename)

        self._nPoints = self._file[0]['linedata/Easting'].size

    def _read_record(self, record):
        """Reads a single data point from the data file.

        FdemData.__initLineByLineRead() must have already been run.

        """
        gdf = self._file[0]['linedata']
        z = np.float64(gdf['Tx_Height'][record])
        primary_field = np.asarray([np.float64(gdf['X_PrimaryField'][record]), np.float64(gdf['Z_PrimaryField'][record])])
        secondary_field = np.hstack([gdf['EMX_NonHPRG'][:, record], gdf['EMZ_NonHPRG'][:, record]])
        std = 0.1 * secondary_field

        out = Tempest_datapoint(
                lineNumber = np.float64(gdf['Line'][record]),
                fiducial = np.float64(gdf['Fiducial'][record]),
                x = np.float64(gdf['Easting'][record]),
                y = np.float64(gdf['Northing'][record]),
                z = z,
                elevation = np.float64(gdf['DTM'][record]),
                # Assign the orientations of the acquisistion loops
                transmitter_loop = CircularLoop(z=z, pitch=np.float64(gdf['Tx_Pitch'][record]), roll=np.float64(gdf['Tx_Roll'][record]), yaw=np.float64(gdf['Tx_Yaw'][record]), radius=self.system[0].loopRadius()),
                receiver_loop = CircularLoop(z=z, pitch=np.float64(gdf['Rx_Pitch'][record]), roll=np.float64(gdf['Rx_Roll'][record]), yaw=np.float64(gdf['Rx_Yaw'][record]), radius=self.system[0].loopRadius()),
                loopOffset = np.hstack([np.float64(gdf['HSep_GPS'][record]), np.float64(gdf['TSep_GPS'][record]), np.float64(gdf['VSep_GPS'][record])]),
                primary_field = primary_field,
                secondary_field = secondary_field,
                std = std,
                system = self.system)

        return out

    def _read_line_fiducial(self, data_filename=None, system_filename=None):
        return self._read_variable(['Line', 'Fiducial'])

    def _read_variable(self, variable):
        gdf = self._file[0]['linedata']

        if isinstance(variable, str):
            variable = [variable]

        return [np.asarray(gdf[var]) for var in variable]

    def _open_data_files(self, data_filename):
        if isinstance(data_filename, str):
            data_filename = [data_filename]
        self._file = []
        for f in data_filename:
            self._file.append(h5py.File(f, 'r'))

    def _close_data_files(self):
        for f in self._file:
            if f.__bool__():
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
        return Tempest_datapoint(x = self.x[i],
                             y = self.y[i],
                             z = self.z[i],
                             elevation = self.elevation[i],
                             primary_field = self.primary_field[i, :],
                             secondary_field = self.secondary_field[i, :],
                             std = self.std[i, :],
                             system = self.system,
                             transmitter_loop = self.transmitter[i],
                             receiver_loop = self.receiver[i],
                             loopOffset = self.loopOffset[i, :],
                             lineNumber = self.lineNumber[i],
                             fiducial = self.fiducial[i],
                             )

    def times(self, system=0):
        """ Obtain the times from the system file """
        assert 0 <= system < self.nSystems, ValueError('system must be in (0, {}]'.format(self.nSystems))
        return StatArray.StatArray(self.system[system].windows.centre, 'Time', 'ms')

    def __getitem__(self, i):
        """ Define item getter for TdemData """
        if not isinstance(i, slice):
            i = np.unique(i)
        return TempestData(self.system,
                        x = self.x[i],
                        y = self.y[i],
                        z = self.z[i],
                        elevation = self.elevation[i],
                        lineNumber = self.lineNumber[i],
                        fiducial = self.fiducial[i],
                        transmitter = self.transmitter[i],
                        receiver = self.receiver[i],
                        loopOffset = self.loopOffset[i, :],
                        secondary_field = self.secondary_field[i, :],
                        primary_field = self.primary_field[i, :],
                        std = self.std[i, :],
                        channelNames = self.channelNames
                        )

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


    def write_csv(self, fileNames, std=False, predictedData=False):

        if isinstance(fileNames, str):
            fileNames = [fileNames]

        assert len(fileNames) == self.nSystems, ValueError("fileNames must have length equal to the number of systems {}".format(self.nSystems))

        for i in range(self.nSystems):

            iSys = self._systemIndices(i)
            # Create the header
            header = "Line Fid Easting Northing Elevation Height txrx_dx txrx_dy txrx_dz TxPitch TxRoll TxYaw RxPitch RxRoll RxYaw "

            for c in self.components_per_channel:
                for x in range(self.nTimes[i]):
                    header += "EM{}_NonHPRG[{}] ".format(c.upper(), x)

            d = np.empty(self.nTimes[i])

            if std:
                for x in range(self.nTimes[i]):
                    header += "OffErr[{}] ".format(x)
                s = np.empty(self.nTimes[i])

            with open(fileNames[i], 'w') as f:
                f.write(header+"\n")
                with np.printoptions(formatter={'float': '{: 0.15g}'.format}, suppress=True):
                    for j in range(self.nPoints):

                        x = np.asarray([self.lineNumber[j], self.fiducial[j], self.x[j], self.y[j], self.elevation[j], self.z[j],
                                        self.loopOffset[j, 0], self.loopOffset[j, 1], self.loopOffset[j, 2],
                                        self.transmitter[j].pitch, self.transmitter[j].roll, self.transmitter[j].yaw,
                                        self.receiver[j].pitch, self.receiver[j].roll, self.receiver[j].yaw])

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
