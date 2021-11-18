"""
.. module:: StatArray
   :platform: Unix, Windows
   :synopsis: Time domain data set

.. moduleauthor:: Leon Foks

"""
from pandas import read_csv
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

    # def __init__(self, system=None, **kwargs):
    #     """ Initialize the TDEM data """

    #     # Data Class containing xyz and channel values
    #     super().__init__(system, **kwargs)

    #     self.primary_field = kwargs.get('primary_field')
    #     self.secondary_field = kwargs.get('secondary_field')
    #     self.predicted_primary_field = kwargs.get('predicted_primary_field')
    #     self.predicted_secondary_field = kwargs.get('predicted_secondary_field')

    # @TdemData.channelNames.setter
    # def channelNames(self, values):
    #     if values is None:
    #         self._channelNames = []
    #         for i in range(self.nSystems):
    #             # Set the channel names
    #             for ic in range(self.n_components):
    #                 for iTime in range(self.nTimes[i]):
    #                     self._channelNames.append('S{} time {:.3e}'.format(self.components[ic].upper(), self.system[i].windows.centre[iTime]))
    #     else:
    #         assert all((isinstance(x, str) for x in values))
    #         assert len(values) == self.nChannels, Exception("Length of channelNames must equal total number of channels {}".format(self.nChannels))
    #         self._channelNames = values

    # @TdemData.components.setter
    # def components(self, values):

    #     if values is None:
    #         values = ['x', 'z']
    #     else:

    #         if isinstance(values, str):
    #             values = [values]

    #         assert np.all([isinstance(x, str) for x in values]), TypeError('components_per_channel must be list of str')

    #     self._components = values

    # @TdemData.data.getter
    # def data(self):

    #     if np.size(self._data, 0) == 0:
    #         self._data = StatArray.StatArray((self.nPoints, self.nChannels), "Data", self.units)

    #     for i in range(self.n_components):
    #         ic = self._component_indices(i, 0)
    #         self._data[:, ic] = self.primary_field[:, i][:, None] + self.secondary_field[:, ic]
    #     return self._data

    @property
    def datapoint_type(self):
        return Tempest_datapoint

    # @property
    # def primary_field(self):
    #     """The data. """
    #     if np.size(self._primary_field, 0) == 0:
    #         self._primary_field = StatArray.StatArray((self.nPoints, self.n_components), "Primary field", self.units)
    #     return self._primary_field


    # @primary_field.setter
    # def primary_field(self, values):
    #     shp = (self.nPoints, self.n_components)
    #     if values is None:
    #         self._primary_field = StatArray.StatArray(shp, "Primary field", self.units)
    #     else:
    #         if self.nPoints == 0:
    #             self.nPoints = np.size(values, 0)
    #         # if self.nChannels == 0:
    #         #     self.channels_per_system = np.size(values, 1)
    #         shp = (self.nPoints, self.n_components)
    #         assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("primary_field must have shape {}".format(shp))
    #         self._primary_field = StatArray.StatArray(values)

    # @property
    # def predicted_primary_field(self):
    #     """The data. """
    #     if np.size(self._primary_field, 0) == 0:
    #         self._primary_field = StatArray.StatArray((self.nPoints, self.n_components), "Primary field", self.units)
    #     return self._primary_field


    # @predicted_primary_field.setter
    # def predicted_primary_field(self, values):
    #     shp = (self.nPoints, self.n_components)
    #     if values is None:
    #         self._predicted_primary_field = StatArray.StatArray(shp, "Predicted primary field", self.units)
    #     else:
    #         if self.nPoints == 0:
    #             self.nPoints = np.size(values, 0)
    #         # if self.nChannels == 0:
    #         #     self.channels_per_system = np.size(values, 1)
    #         shp = (self.nPoints, self.n_components)
    #         assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("predicted_primary_field must have shape {}".format(shp))
    #         self._predicted_primary_field = StatArray.StatArray(values)

    # @property
    # def secondary_field(self):
    #     """The data. """
    #     if np.size(self._secondary_field, 0) == 0:
    #         self._secondary_field = StatArray.StatArray((self.nPoints, self.nChannels), "Secondary field", self.units)
    #     return self._secondary_field

    # @secondary_field.setter
    # def secondary_field(self, values):
    #     shp = (self.nPoints, self.nChannels)
    #     if values is None:
    #         self._secondary_field = StatArray.StatArray(shp, "Secondary field", self.units)
    #     else:
    #         if self.nPoints == 0:
    #             self.nPoints = np.size(values, 0)
    #         if self.nChannels == 0:
    #             self.channels_per_system = np.size(values, 1)
    #         shp = (self.nPoints, self.nChannels)
    #         assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("seconday_field must have shape {}".format(shp))
    #         self._secondary_field = StatArray.StatArray(values)

    # @property
    # def predicted_secondary_field(self):
    #     """The data. """
    #     if np.size(self._secondary_field, 0) == 0:
    #         self._predicted_secondary_field = StatArray.StatArray((self.nPoints, self.nChannels), "Predicted secondary field", self.units)
    #     return self._predicted_secondary_field

    # @predicted_secondary_field.setter
    # def predicted_secondary_field(self, values):
    #     shp = (self.nPoints, self.nChannels)
    #     if values is None:
    #         self._predicted_secondary_field = StatArray.StatArray(shp, "Predicted secondary field", self.units)
    #     else:
    #         if self.nPoints == 0:
    #             self.nPoints = np.size(values, 0)
    #         if self.nChannels == 0:
    #             self.channels_per_system = np.size(values, 1)
    #         shp = (self.nPoints, self.nChannels)
    #         assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("predicted_seconday_field must have shape {}".format(shp))
    #         self._predicted_secondary_field = StatArray.StatArray(values)

    # @TdemData.std.getter
    # def std(self):
    #     """The data. """
    #     if np.size(self._std, 0) == 0:
    #         self._std = StatArray.StatArray((self.nPoints, self.nChannels), "Standard deviation", self.units)

    #     relative_error = self.relative_error * self.secondary_field
    #     self._std = np.sqrt((relative_error**2.0) + (self.additive_error**2.0))

    #     return self._std

    def _as_dict(self):
        out, order = super()._as_dict()

        for i, name in enumerate(self.channelNames):
            out[name.replace(' ', '_')] = self.secondary_field[:, i]

        for i, c in enumerate(self.components):
            out['P{}'.format(c.upper())] = self.primary_field[:, i]

        order = [*order[:15], *['P{}'.format(c.upper()) for c in self.components], *order[15:]]

        return out, order


    # def append(self, other):

    #     super().append(self, other)

    #     self.primary_field = np.hstack(self.primary_field, other.primary_field)

    @classmethod
    def read_csv(cls, data_filename, system_filename):
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
        if (isinstance(system_filename, str)):
            system_filename = [system_filename]

        nSystems = len(system_filename)

        self = cls(system=system_filename)

        self._nPoints, iC, iR, iT, iOffset, iSecondary, iStd, iPrimary = TempestData._csv_channels(data_filename)

        assert len(iSecondary) == self.nChannels, Exception("Number of off time columns {} in {} does not match total number of times {} in system files \n {}".format(
            len(iSecondary), data_filename, self.nChannels, self.fileInformation()))

        if len(iStd) > 0:
            assert len(iStd) == len(iSecondary), Exception("Number of Off time standard deviation estimates does not match number of Off time data columns in file {}. \n {}".format(data_filename, self.fileInformation()))

        # Get all readable column indices for the first file.
        channels = iC + iR + iT + iOffset + iSecondary + iPrimary
        if len(iStd) > 0:
            channels += iStd

        # Read in the columns from the first data file
        try:
            df = read_csv(data_filename, usecols=channels, skipinitialspace = True)
        except:
            df = read_csv(data_filename, usecols=channels, delim_whitespace=True, skipinitialspace = True)
        df = df.replace('NaN', np.nan)

        # Assign columns to variables
        self.lineNumber = df[iC[0]].values
        self.fiducial = df[iC[1]].values
        self.x = df[iC[2]].values
        self.y = df[iC[3]].values
        self.z = df[iC[4]].values
        self.elevation = df[iC[5]].values

        self.transmitter = None
        for i in range(self.nPoints):
            self.transmitter[i] = CircularLoop(x=self.x[i], y=self.y[i], z=self.z[i],
                                               pitch=df[iT[0]].values[i], roll=df[iT[1]].values[i], yaw=df[iT[2]].values[i],
                                               radius=self.system[0].loopRadius())

        loopOffset = df[iOffset].values

        # Assign the orientations of the acquisistion loops
        self.receiver = None
        for i in range(self.nPoints):
            self.receiver[i] = CircularLoop(x = self.transmitter[i].x + loopOffset[i, 0],
                                            y = self.transmitter[i].y + loopOffset[i, 1],
                                            z = self.transmitter[i].z + loopOffset[i, 2],
                                            pitch=df[iR[0]].values[i], roll=df[iR[1]].values[i], yaw=df[iR[2]].values[i],
                                            radius=self.system[0].loopRadius())


        self.primary_field[:, :] = df[iPrimary].values
        self.secondary_field[:, :] = df[iSecondary].values
        
        # If the data error columns are given, assign them
        self.std;
        if len(iStd) > 0:
            self._std[:, :] = df[iStd].values    

        self.check()

        return self

    # def csv_channels(self, data_filename):
    
    #     self._nPoints, self._iC, self._iR, self._iT, self._iOffset, self._iD, self._iS = self._csv_channels(data_filename)

    #     self._channels = []
    #     for i in range(self.nSystems):
    #         channels = self._iC[i] + self._iR[i] + self._iT[i] + self._iOffset[i] + self._iD[i]
    #         if not self._iS[i] is None:
    #             channels += self._iS[i]
    #         self._channels.append(channels)

    #     return self._channels

    @staticmethod
    def _csv_channels(data_filename):
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
        channels = fIO.get_column_name(data_filename)

        primary_names = ('px', 'py', 'pz')
        iPrimary = []
        for channel in channels:
            cTmp = channel.lower()
            if cTmp in primary_names:
                iPrimary.append(channel)
        iPrimary.sort()

        return *TdemData._csv_channels(data_filename), iPrimary

    @classmethod
    def read_netcdf(cls, dataFilename, systemFilename, indices=None):
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

        if indices is None:
            indices = np.s_[:]

        with h5py.File(dataFilename, 'r') as f:
            gdf = f['linedata']

            self = cls(lineNumber=np.asarray(gdf['Line'][indices]),
                        fiducial=np.asarray(gdf['Fiducial'][indices]),
                        x=np.asarray(gdf['Easting'][indices]),
                        y=np.asarray(gdf['Northing'][indices]),
                        z=np.asarray(gdf['Tx_Height'][indices]),
                        elevation=np.asarray(gdf['DTM'][indices]),
                        system=systemFilename)

            # Assign the orientations of the acquisistion loops
            pitch = np.asarray(gdf['Tx_Pitch'][indices])
            roll = np.asarray(gdf['Tx_Roll'][indices])
            yaw = np.asarray(gdf['Tx_Yaw'][indices])

            self.transmitter = [CircularLoop(x=self.x[i], y=self.y[i], z=self.z[i],
                                             pitch=pitch[i], roll=roll[i], yaw=yaw[i],
                                             radius=self.system[0].loopRadius()) for i in range(self.nPoints)]

            pitch = np.asarray(gdf['Rx_Pitch'][indices])
            roll = np.asarray(gdf['Rx_Roll'][indices])
            yaw = np.asarray(gdf['Rx_Yaw'][indices])

            loopOffset = np.vstack([np.asarray(gdf['HSep_GPS'][indices]), np.asarray(gdf['TSep_GPS'][indices]), np.asarray(gdf['VSep_GPS'][indices])]).T

            self.receiver = [CircularLoop(x=self.transmitter[i].x + loopOffset[i, 0],
                                          y=self.transmitter[i].y + loopOffset[i, 1],
                                          z=self.transmitter[i].z + loopOffset[i, 2],
                                          pitch=pitch[i], roll=roll[i], yaw=yaw[i],
                                          radius=self.system[0].loopRadius()) for i in range(self.nPoints)]

            self.primary_field = np.vstack([np.asarray(gdf['X_PrimaryField'][indices]), np.asarray(gdf['Z_PrimaryField'][indices])]).T
            self.secondary_field = np.hstack([np.asarray(gdf['EMX_NonHPRG'][:, indices]).T, np.asarray(gdf['EMZ_NonHPRG'][:, indices]).T])

            self.std = 0.1 * self.data

        self.check()

        return self

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

        self = cls(system_filename)
        self._data_filename = data_filename
        self._open_data_files(data_filename)
        return self

        # # Read in the EM System file
        # self.system = system_filename

        # self._open_data_files(data_filename)

        # self._nPoints = self._file[0]['linedata/Easting'].size

    def _read_record(self, record):
        """Reads a single data point from the data file.

        FdemData.__initLineByLineRead() must have already been run.

        """
        gdf = self._file[0]['linedata']
        x = np.float64(gdf['Easting'][record])
        y = np.float64(gdf['Northing'][record])
        z = np.float64(gdf['Tx_Height'][record])
        primary_field = np.asarray([np.float64(gdf['X_PrimaryField'][record]), np.float64(gdf['Z_PrimaryField'][record])])
        secondary_field = np.hstack([gdf['EMX_NonHPRG'][:, record], gdf['EMZ_NonHPRG'][:, record]])
        std = 0.1 * secondary_field

        transmitter_loop = CircularLoop(x=x, y=y, z=z,
                                        pitch=np.float64(gdf['Tx_Pitch'][record]),
                                        roll=np.float64(gdf['Tx_Roll'][record]),
                                        yaw=np.float64(gdf['Tx_Yaw'][record]),
                                        radius=self.system[0].loopRadius())

        loopOffset = np.vstack([np.asarray(gdf['HSep_GPS'][record]), np.asarray(gdf['TSep_GPS'][record]), np.asarray(gdf['VSep_GPS'][record])])

        receiver_loop = CircularLoop(x=transmitter_loop.x + loopOffset[0],
                                     y=transmitter_loop.y + loopOffset[1],
                                     z=transmitter_loop.z + loopOffset[2],
                                     pitch=np.float64(gdf['Rx_Pitch'][record]),
                                     roll=np.float64(gdf['Rx_Roll'][record]),
                                     yaw=np.float64(gdf['Rx_Yaw'][record]),
                                     radius=self.system[0].loopRadius())

        out = Tempest_datapoint(
                lineNumber = np.float64(gdf['Line'][record]),
                fiducial = np.float64(gdf['Fiducial'][record]),
                x = x,
                y = y,
                z = z,
                elevation = np.float64(gdf['DTM'][record]),
                # Assign the orientations of the acquisistion loops
                transmitter_loop = transmitter_loop,

                receiver_loop = receiver_loop,
                # loopOffset = np.hstack([np.float64(gdf['HSep_GPS'][record]), np.float64(gdf['TSep_GPS'][record]), np.float64(gdf['VSep_GPS'][record])]),
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
        self._nPoints = self._file[0]['linedata/Easting'].size

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
                             lineNumber = self.lineNumber[i],
                             fiducial = self.fiducial[i],
                             )

    # def times(self, system=0):
    #     """ Obtain the times from the system file """
    #     assert 0 <= system < self.nSystems, ValueError('system must be in (0, {}]'.format(self.nSystems))
    #     return StatArray.StatArray(self.system[system].windows.centre, 'Time', 'ms')

    # def __getitem__(self, i):
    #     """ Define item getter for TdemData """
    #     if not isinstance(i, slice):
    #         i = np.unique(i)
    #     return TempestData(self.system,
    #                     x = self.x[i],
    #                     y = self.y[i],
    #                     z = self.z[i],
    #                     elevation = self.elevation[i],
    #                     lineNumber = self.lineNumber[i],
    #                     fiducial = self.fiducial[i],
    #                     transmitter = [self.transmitter[j] for j in i],
    #                     receiver = [self.receiver[j] for j in i],
    #                     secondary_field = self.secondary_field[i, :],
    #                     primary_field = self.primary_field[i, :],
    #                     std = self.std[i, :],
    #                     channelNames = self.channelNames
    #                     )

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


    # def Bcast(self, world, root=0, system=None):
    #     """Broadcast the TdemData using MPI

    #     Parameters
    #     ----------

    #     """

    #     out = super().Bcast(world, root, system)

    #     out._primary_field = self.primary_field.Bcast(world, root=root)
    #     out._secondary_field = self.secondary_field.Bcast(world, root=root)
    #     out._predicted_primary_field = self.predicted_primary_field.Bcast(world, root=root)
    #     out._predicted_secondary_field = self.predicted_secondary_field.Bcast(world, root=root)

    #     return out


    # def Scatterv(self, starts, chunks, world, root=0, system=None):
    #     """ Scatterv the TdemData using MPI """

    #     out = super().Scatterv(starts, chunks, world, root)

    #     out.primary_field = self.primary_field.Scatterv(starts, chunks, world, root=root)
    #     out.secondary_field = self.secondary_field.Scatterv(starts, chunks, world, root=root)
    #     out.predicted_primary_field = self.predicted_primary_field.Scatterv(starts, chunks, world, root=root)
    #     out.predicted_secondary_field = self.predicted_secondary_field.Scatterv(starts, chunks, world, root=root)

    #     return out

