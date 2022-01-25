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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._file = None

    @property
    def additive_error(self):
        """The data. """
        if np.size(self._additive_error, 0) == 0:
            self._additive_error = StatArray.StatArray((self.nPoints, self.nChannels), "Additive error", "%")
        return self._additive_error

    @additive_error.setter
    def additive_error(self, values):
        shp = (self.nPoints, self.nChannels)
        if values is None:
            self._additive_error = StatArray.StatArray(np.ones(shp), "Additive error", "%")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values, 0)
                shp = (self.nPoints, self.nChannels)
            assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("additive_error must have shape {}".format(shp))
            self._additive_error = StatArray.StatArray(values)

    @property
    def datapoint_type(self):
        return Tempest_datapoint

    @property
    def file(self):
        return self._file

    @property
    def relative_error(self):
        """The data. """
        if np.size(self._relative_error, 0) == 0:
            self._relative_error = StatArray.StatArray((self.nPoints, self.n_components * self.nSystems), "Relative error", "%")
        return self._relative_error

    @relative_error.setter
    def relative_error(self, values):
        shp = (self.nPoints, self.n_components * self.nSystems)
        if values is None:
            self._relative_error = StatArray.StatArray(np.ones(shp), "Relative error", "%")
        else:
            if self.nPoints == 0:
                self.nPoints = np.size(values, 0)
                shp = (self.nPoints, self.n_components * self.nSystems)
            assert np.allclose(np.shape(values), shp) or np.size(values) == self.nPoints, ValueError("relative_error must have shape {}".format(shp))
            self._relative_error = StatArray.StatArray(values)

    def _as_dict(self):
        out, order = super()._as_dict()

        for i, name in enumerate(self.channelNames):
            out[name.replace(' ', '_')] = self.secondary_field[:, i]

        for i, c in enumerate(self.components):
            out['P{}'.format(c.upper())] = self.primary_field[:, i]

        order = [*order[:15], *['P{}'.format(c.upper()) for c in self.components], *order[15:]]

        return out, order

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
    
    #     self._nPoints, self._iC, self._iR, self._iT, self._iOffset, self._iData, self._iStd, self._iPrimary = TdemData._csv_channels(data_filename)

    #     self._channels = self._iC + self._iR + self._iT + self._iOffset + self._iData
    #     if len(self._iStd) > 0:
    #         self._channels += self._iStd
    #     self._channels += self._iPrimary

    #     return self._channels

    # @staticmethod
    # def _csv_channels(data_filename):
    #     """Reads the column indices for the co-ordinates, loop orientations, and data from the TdemData file.

    #     Parameters
    #     ----------
    #     dataFilename : str or list of str
    #         Path to the data file(s)
    #     system : list of geobipy.TdemSystem
    #         System class for each time domain acquisition system.

    #     Returns
    #     -------
    #     indices : list of ints
    #         Size 6 indices to line, fid, easting, northing, height, and elevation.
    #     rLoopIndices : list of ints
    #         Size 3 indices to pitch, roll, and yaw, for the receiver loop.
    #     tLoopIndices : list of ints
    #         Size 3 indices to pitch, roll, and yaw, for the transmitter loop.
    #     offDataIndices : list of ints
    #         Indices to the off time data columns.  Size == number of time windows.
    #     offErrIndices : list of ints
    #         Indices to the off time uncertainty estimate columns.  Size == number of time windows.

    #     """
    #     channels = fIO.get_column_name(data_filename)

        

    #     return *TdemData._csv_channels(data_filename), iPrimary

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
        # self._data_filename = data_filename
        self._open_data_files(data_filename)
        return self

    def _read_record(self, record):
        """Reads a single data point from the data file.

        FdemData.__initLineByLineRead() must have already been run.

        """
        if self._filename.endswith('.csv'):
            out = super()._read_record(record)
            return out

        gdf = self._file['linedata']
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

        out = self.datapoint_type(
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

    def _read_line_fiducial(self, filename=None):
        if filename.endswith('.csv'):
            return super()._read_line_fiducial(filename)
        
        return self._read_variable(['Line', 'Fiducial'])

    def _read_variable(self, variable):
        gdf = self._file['linedata']

        if isinstance(variable, str):
            variable = [variable]

        return [np.asarray(gdf[var]) for var in variable]

    def _open_data_files(self, filename):

        if filename.endswith('.csv'):
            super()._open_csv_files(filename)
            return

        self._file = h5py.File(filename, 'r')
        self._filename = filename
        self._nPoints = self._file['linedata/Easting'].size

    @classmethod
    def fromHdf(cls, grp, **kwargs):
        """ Reads the object from a HDF group """

        if kwargs.get('index') is not None:
            return Tempest_datapoint.fromHdf(grp, **kwargs)

        return super(TempestData, cls).fromHdf(grp)

