"""
"""
from numpy import allclose, asarray, atleast_1d, float64, full, hstack
from numpy import nan, ones
from numpy import s_, shape, size, vstack
from pandas import read_csv
from matplotlib.figure import Figure

from geobipy.src.base import utilities
from .TdemData import TdemData
from ..datapoint.Tempest_datapoint import Tempest_datapoint
from ....classes.core import StatArray
from ...system.CircularLoop import CircularLoop
from ...system.CircularLoops import CircularLoops
from ....base import plotting as cP

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

    single = Tempest_datapoint

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._additive_error = StatArray.StatArray((self.nPoints, self.nChannels), "Additive error", "%")
        self._relative_error = StatArray.StatArray((self.nPoints, self.n_components * self.nSystems), "Relative error", "%")

        self._file = None

    @property
    def additive_error(self):
        """The data. """
        if size(self._additive_error, 0) == 0:
            self._additive_error = StatArray.StatArray((self.nPoints, self.nChannels), "Additive error", "%")
        return self._additive_error

    @additive_error.setter
    def additive_error(self, values):
        if values is not None:
            self.nPoints, self.nChannels = size(values, 0), size(values, 1)
            shp = (self.nPoints, self.nChannels)
            if not allclose(self._additive_error.shape, shp):
                self._additive_error = StatArray.StatArray(values, "Additive error", self.units)
                return

            self._additive_error[:, :] = values

    @property
    def file(self):
        return self._file


    @property
    def relative_error(self):
        """The data. """
        if size(self._relative_error, 0) == 0:
            self._relative_error = StatArray.StatArray((self.nPoints, self.n_components * self.nSystems), "Relative error", "%")
        return self._relative_error

    @relative_error.setter
    def relative_error(self, values):
        if values is not None:
            self.nPoints = size(values, 0)
            shp = (self.nPoints, self.n_components * self.nSystems)
            if not allclose(self._relative_error.shape, shp):
                self._relative_error = StatArray.StatArray(values, "Relative error", "%")
                return

            self._relative_error[:, :] = values

    def _as_dict(self):
        out, order = super()._as_dict()

        for i, name in enumerate(self.channel_names):
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
        df = df.replace('NaN', nan)

        # Assign columns to variables
        self.lineNumber = df[iC[0]].values
        self.fiducial = df[iC[1]].values
        self.x = df[iC[2]].values
        self.y = df[iC[3]].values
        self.z = df[iC[4]].values
        self.elevation = df[iC[5]].values

        self.transmitter = CircularLoops(x=self.x,
                                         y=self.y,
                                         z=self.z,
                                         pitch=df[iT[0]].values, roll=df[iT[1]].values, yaw=df[iT[2]].values,
                                         radius=full(self.nPoints, fill_value=self.system[0].loopRadius()))

        loopOffset = df[iOffset].values

        # Assign the orientations of the acquisistion loops
        self.receiver = CircularLoops(x = self.transmitter.x + loopOffset[:, 0],
                                      y = self.transmitter.y + loopOffset[:, 1],
                                      z = self.transmitter.z + loopOffset[:, 2],
                                      pitch=df[iR[0]].values, roll=df[iR[1]].values, yaw=df[iR[2]].values,
                                      radius=full(self.nPoints, fill_value=self.system[0].loopRadius()))


        self.primary_field[:, :] = df[iPrimary].values
        self.secondary_field[:, :] = df[iSecondary].values

        # If the data error columns are given, assign them
        self.std;
        if len(iStd) > 0:
            self._std[:, :] = df[iStd].values

        self.check()

        return self

    def plot_data(self, system=0, channels=None, x='index', **kwargs):
        """ Plots the data

        Parameters
        ----------
        system : int
            System to plot
        channels : sequence of ints
            Channels to plot

        """

        legend = kwargs.pop('legend', True)
        kwargs['yscale'] = kwargs.get('yscale', 'linear')

        x = self.axis(x)

        if channels is None:
            i = self._systemIndices(system)
            ax = cP.plot(x, self.data[:, i],
                         label=self.channel_names[i], **kwargs)
        else:
            channels = atleast_1d(channels)
            for j, i in enumerate(channels):
                ax = cP.plot(x, self.data[:, i],
                             label=self.channel_names[i], **kwargs)

        plt.xlabel(utilities.getNameUnits(x))

        # Put a legend to the right of the current axis
        if legend:
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
            leg.set_title(self.data.getNameUnits())

        return ax


    def plot_predicted(self, system=0, channels=None, xAxis='index', **kwargs):
        """ Plots the data

        Parameters
        ----------
        system : int
            System to plot
        channels : sequence of ints
            Channels to plot

        """

        legend = kwargs.pop('legend', True)
        kwargs['yscale'] = kwargs.get('yscale', 'linear')
        kwargs['linestyle'] = kwargs.get('linestyle', '-.')

        x = self.getXAxis(xAxis)

        if channels is None:
            i = self._systemIndices(system)
            ax = cP.plot(x, self.predictedData[:, i],
                         label=self.channel_names[i], **kwargs)
        else:
            channels = atleast_1d(channels)
            for j, i in enumerate(channels):
                ax = cP.plot(x, self.predictedData[:, i],
                             label=self.channel_names[i], **kwargs)

        plt.xlabel(utilities.getNameUnits(x))

        # Put a legend to the right of the current axis
        if legend:
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
            leg.set_title(self.predictedData.getNameUnits())

        return ax

    def _init_posterior_plots(self, gs, sharex=None, sharey=None):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        splt = gs.subgridspec(2, 2, wspace=0.3)
        ax = []
        # Height axis
        ax.append(plt.subplot(splt[0, 0], sharex=sharex, sharey=sharey))

        sharex = ax[0] if sharex is None else sharex
        # Data axis
        ax.append(plt.subplot(splt[0, 1], sharex=sharex))

        splt2 = splt[1, :].subgridspec(self.nSystems * self.n_components, 2, wspace=0.2)
        # Relative error axes
        ax_rel = [plt.subplot(splt2[0, 0], sharex=sharex)]
        ax_rel += [plt.subplot(splt2[i, 0], sharex=ax_rel[0], sharey=ax_rel[0]) for i in range(1, self.n_components * self.nSystems)]
        ax.append(ax_rel)
        # # Additive Error axes
        ax.append(plt.subplot(splt2[0, 1], sharex=sharex))

        return ax

    def plot_posteriors(self, axes=None, height_kwargs={}, data_kwargs={}, rel_error_kwargs={}, transmitter_pitch_kwargs={}, sharex=None, sharey=None, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', plt.gcf())

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes, sharex=sharex, sharey=sharey)

        # assert len(axes) == 4, ValueError("axes must have length 4")
        # assert len(axes) == 4, ValueError("Must have length 3 list of axes for the posteriors. self.init_posterior_plots can generate them")

        self.z.plot_posteriors(ax = axes[0], **height_kwargs)

        self.plot(ax=axes[1], legend=False, **data_kwargs)
        self.plot_predicted(ax=axes[1], legend=False, **data_kwargs)

        self.relative_error.plot_posteriors(ax=axes[2], **rel_error_kwargs)

        self.transmitter.pitch.plot_posteriors(ax=axes[3], **transmitter_pitch_kwargs)

        return axes

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
            indices = s_[:]

        with h5py.File(dataFilename, 'r') as f:
            gdf = f['linedata']

            self = cls(lineNumber=asarray(gdf['Line'][indices]),
                        fiducial=asarray(gdf['Fiducial'][indices]),
                        x=asarray(gdf['Easting_Albers'][indices]),
                        y=asarray(gdf['Northing_Albers'][indices]),
                        z=asarray(gdf['Tx_Height'][indices]),
                        elevation=asarray(gdf['DTM'][indices]),
                        system=systemFilename)

            # Assign the orientations of the acquisistion loops
            pitch = asarray(gdf['Tx_Pitch'][indices])
            roll = asarray(gdf['Tx_Roll'][indices])
            yaw = asarray(gdf['Tx_Yaw'][indices])

            self.transmitter = CircularLoops(x=self.x, y=self.y, z=self.z,
                                             pitch=pitch, roll=roll, yaw=yaw,
                                             radius=full(self.nPoints, fill_value=self.system[0].loopRadius()))

            pitch = asarray(gdf['Rx_Pitch'][indices])
            roll = asarray(gdf['Rx_Roll'][indices])
            yaw = asarray(gdf['Rx_Yaw'][indices])

            loopOffset = vstack([asarray(gdf['HSep_GPS'][indices]), asarray(gdf['TSep_GPS'][indices]), asarray(gdf['VSep_GPS'][indices])]).T

            self.receiver = CircularLoops(x=self.transmitter.x + loopOffset[:, 0],
                                          y=self.transmitter.y + loopOffset[:, 1],
                                          z=self.transmitter.z + loopOffset[:, 2],
                                          pitch=pitch, roll=roll, yaw=yaw,
                                          radius=full(self.nPoints, fill_value=self.system[0].loopRadius()))

            self.primary_field = vstack([asarray(gdf['X_PrimaryField'][indices]), asarray(gdf['Z_PrimaryField'][indices])]).T
            self.secondary_field = hstack([asarray(gdf['EMX_NonHPRG'][:, indices]).T, asarray(gdf['EMZ_NonHPRG'][:, indices]).T])

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

    def _read_record(self, record=None, mpi_enabled=False):
        """Reads a single data point from the data file.

        FdemData.__initLineByLineRead() must have already been run.

        """
        if self._data_filename.endswith('.csv'):
            out = super()._read_record(record, mpi_enabled=mpi_enabled)
            return out

        assert record is not None, ValueError("Need to provide a record index for netcdf files")

        gdf = self._file['survey/tabular/0']
        x = float64(gdf['Easting'][record])
        y = float64(gdf['Northing'][record])
        z = float64(gdf['Tx_Height'][record])
        primary_field = asarray([float64(gdf['X_PrimaryField'][record]), float64(gdf['Z_PrimaryField'][record])])
        secondary_field = hstack([gdf['EMX_NonHPRG'][record, :], gdf['EMZ_NonHPRG'][record, :]])

        transmitter_loop = CircularLoop(x=x, y=y, z=z,
                                        pitch=float64(gdf['Tx_Pitch'][record]),
                                        roll=float64(gdf['Tx_Roll'][record]),
                                        yaw=float64(gdf['Tx_Yaw'][record]),
                                        radius=self.system[0].loopRadius())

        loopOffset = vstack([asarray(gdf['HSep_GPS'][record]), asarray(gdf['TSep_GPS'][record]), asarray(gdf['VSep_GPS'][record])])

        receiver_loop = CircularLoop(x=transmitter_loop.x + loopOffset[0],
                                      y=transmitter_loop.y + loopOffset[1],
                                      z=transmitter_loop.z + loopOffset[2],
                                      pitch=float64(gdf['Rx_Pitch'][record]),
                                      roll=float64(gdf['Rx_Roll'][record]),
                                      yaw=float64(gdf['Rx_Yaw'][record]),
                                      radius=self.system[0].loopRadius())

        out = self.single(
                lineNumber = float64(gdf['Line'][record]),
                fiducial = float64(gdf['Fiducial'][record]),
                x = x,
                y = y,
                z = z,
                elevation = float64(gdf['DTM'][record]),
                transmitter_loop = transmitter_loop,
                receiver_loop = receiver_loop,
                primary_field = primary_field,
                secondary_field = secondary_field,
                system = self.system)

        return out

    def _read_line_fiducial(self, filename=None):
        if filename.endswith('.csv'):
            return super()._read_line_fiducial(filename)

        self.lineNumber, self.fiducial = self._read_variable(['Line', 'Fiducial'])

    def _read_variable(self, variable):
        gdf = self._file['survey/tabular/0']

        if isinstance(variable, str):
            variable = [variable]

        return [asarray(gdf[var]) for var in variable]

    def _open_data_files(self, filename):

        if filename.endswith('.csv'):
            super()._open_csv_files(filename)
            return

        self._file = h5py.File(filename, 'r')
        self._data_filename = filename
        self.lineNumber, self.fiducial = self._read_variable(['Line', 'Fiducial'])

    @classmethod
    def fromHdf(cls, grp, **kwargs):
        """ Reads the object from a HDF group """

        if kwargs.get('index') is not None:
            return cls.single.fromHdf(grp, **kwargs)

        out = super(TempestData, cls).fromHdf(grp, **kwargs)
        out.primary_field = StatArray.StatArray.fromHdf(grp['primary_field'])
        return out

    def create_synthetic_data(self, model, prng):

        ds = TempestData(system=self.system)

        ds.x = model.x
        ds.y = model.y
        ds.z = np.full(model.x.nCells, fill_value = 120.0)
        ds.elevation = np.zeros(model.x.nCells)
        ds.fiducial = np.arange(model.x.nCells)

        ds.loop_pair.transmitter = CircularLoops(
                        x = ds.x, y = ds.y, z = ds.z,
                        pitch = np.zeros(model.x.nCells), #np.random.uniform(low=-1.0, high=1.0, size=model.x.nCells),
                        roll  = np.zeros(model.x.nCells), #np.random.uniform(low=-1.0, high=1.0, size=model.x.nCells),
                        yaw   = np.zeros(model.x.nCells), #np.random.uniform(low=-1.0, high=1.0, size=model.x.nCells),
                        radius = np.full(model.x.nCells, fill_value=ds.system[0].loopRadius()))

        ds.loop_pair.receiver = CircularLoops(
                        x = ds.transmitter.x - 107.0,
                        y = ds.transmitter.y + 0.0,
                        z = ds.transmitter.z - 45.0,
                        pitch = np.zeros(model.x.nCells), #np.random.uniform(low=-0.5, high=0.5, size=model.x.nCells),
                        roll  = np.zeros(model.x.nCells), #np.random.uniform(low=-0.5, high=0.5, size=model.x.nCells),
                        yaw   = np.zeros(model.x.nCells), #np.random.uniform(low=-0.5, high=0.5, size=model.x.nCells),
                        radius = np.full(model.x.nCells, fill_value=ds.system[0].loopRadius()))

        ds.relative_error = np.repeat(np.r_[0.001, 0.001][None, :], model.x.nCells, 0)
        add_error = np.r_[0.011474, 0.012810, 0.008507, 0.005154, 0.004742, 0.004477, 0.004168, 0.003539, 0.003352, 0.003213, 0.003161, 0.003122, 0.002587, 0.002038, 0.002201,
                        0.007383, 0.005693, 0.005178, 0.003659, 0.003426, 0.003046, 0.003095, 0.003247, 0.002775, 0.002627, 0.002460, 0.002178, 0.001754, 0.001405, 0.001283]
        ds.additive_error = np.repeat(add_error[None, :], model.x.nCells, 0)

        dp = ds.datapoint(0)

        for k in range(model.x.nCells):
            mod = model[k]

            dp.forward(mod)
            dp.secondary_field[:] = dp.predicted_secondary_field
            dp.primary_field[:] = dp.predicted_primary_field

            ds.primary_field[k, :] = dp.primary_field
            ds.secondary_field[k, :] = dp.secondary_field

        ds_noisy = deepcopy(ds)

        # Add noise to various solvable parameters

        # ds.z += np.random.uniform(low=-5.0, high=5.0, size=model.x.nCells)
        # ds.receiver.x += np.random.normal(loc=0.0, scale=0.25**2.0, size=model.x.nCells)
        # ds.receiver.z += np.random.normal(loc = 0.0, scale = 0.25**2.0, size=model.x.nCells)
        # ds.receiver.pitch += np.random.normal(loc = 0.0, scale = 0.25**2.0, size=model.x.nCells)
        # ds.receiver.roll += np.random.normal(loc = 0.0, scale = 0.5**2.0, size=model.x.nCells)
        # ds.receiver.yaw += np.random.normal(loc = 0.0, scale = 0.5**2.0, size=model.x.nCells)

        ds_noisy.secondary_field += prng.normal(scale=ds.std, size=(model.x.nCells, ds.nChannels))

        return ds, ds_noisy