from abc import abstractmethod
from cached_property import cached_property
from copy import deepcopy
from ...pointcloud.Point import Point
from ....classes.core import StatArray
from ....base import utilities as cf
from ....base import plotting as cP
from ....base import MPI as myMPI
from ...statistics.Histogram2D import Histogram2D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DataPoint(Point):
    """Class defines a data point.

    Contains an easting, northing, height, elevation, observed and predicted data, and uncertainty estimates for the data.

    DataPoint(x, y, z, elevation, nChannels, data, std, units)

    Parameters
    ----------
    nChannelsPerSystem : int or array_like
        Number of data channels in the data
        * If int, a single acquisition system is assumed.
        * If array_like, each entry is the number of channels for each system.
    x : float
        Easting co-ordinate of the data point
    y : float
        Northing co-ordinate of the data point
    z : float
        Height above ground of the data point
    elevation : float, optional
        Elevation from sea level of the data point
    data : geobipy.StatArray or array_like, optional
        Data values to assign the data of length sum(nChannelsPerSystem).
        * If None, initialized with zeros.
    std : geobipy.StatArray or array_like, optional
        Estimated uncertainty standard deviation of the data of length sum(nChannelsPerSystem).
        * If None, initialized with ones if data is None, else 0.1*data values.
    predictedData : geobipy.StatArray or array_like, optional
        Predicted data values to assign the data of length sum(nChannelsPerSystem).
        * If None, initialized with zeros.
    units : str, optional
        Units of the data.  Default is "ppm".
    channelNames : list of str, optional
        Names of each channel of length sum(nChannelsPerSystem)

    """

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=None,
                       channels_per_system=1, components_per_channel=None,
                       data=None, std=None, predictedData=None,
                       units=None, channelNames=None,
                       lineNumber=0.0, fiducial=0.0):
        """ Initialize the Data class """

        super().__init__(x, y, z)

        # Number of Channels
        self.components_per_channel = components_per_channel
        self._channels_per_system = np.atleast_1d(np.asarray(channels_per_system * self.n_components))

        self.elevation = elevation

        self.units = units

        # StatArray of data
        self.data = data

        self.std = std

        self.predictedData = predictedData

        self.lineNumber = lineNumber

        self.fiducial = fiducial

        self.channelNames = channelNames

        self.relErr = None
        self.addErr = None

        # self.errorPosterior = None

    @property
    def components_per_channel(self):
        return self._components_per_channel

    @components_per_channel.setter
    def components_per_channel(self, values):
        if values is None:
            self._components_per_channel = ['z']
        else:
            assert np.all([isinstance(x, str) for x in values]), TypeError('components_per_channel must be list of str')
            self._components_per_channel = values

    @property
    def n_components(self):
        if self.components_per_channel is None:
            return 1
        return np.size(self.components_per_channel)

    @property
    def n_posteriors(self):
        return self._n_error_posteriors + self.z.hasPosterior

    @property
    def _n_error_posteriors(self):
        # if not self.errorPosterior is None:
        #     return len(self.errorPosterior)
        # else:
        return self.nSystems * np.sum([x.hasPosterior for x in [self.relErr, self.addErr]])

    def __deepcopy__(self, memo={}):

        out = super().__deepcopy__(memo)

        out._components_per_channel = deepcopy(self._components_per_channel, memo)
        out._channels_per_system = deepcopy(self._channels_per_system, memo)

        out._elevation = deepcopy(self.elevation, memo)
        out._units = deepcopy(self.units, memo)
        out._data = deepcopy(self.data, memo)
        out._std = deepcopy(self.std, memo)
        out._predictedData = deepcopy(self.predictedData, memo)
        out._lineNumber = deepcopy(self.lineNumber, memo)
        out._fiducial = deepcopy(self.fiducial, memo)
        out._channelNames = deepcopy(self.channelNames, memo)
        out._relErr = deepcopy(self.relErr, memo)
        out._addErr = deepcopy(self.addErr, memo)
        # out._errorPosterior = deepcopy(self.errorPosterior, memo)

        return out

    @property
    def additive_error(self):
        return self._addErr

    @property
    def addErr(self):
        return self._addErr

    @addErr.setter
    def addErr(self, values):
        if values is None:
            values = self.nSystems
        else:
            if np.size(values) == 1:
                values = np.full(self.nSystems, fill_value=values)
            else:
                values = np.asarray(values)
            assert np.size(values) == self.nSystems, ValueError("additiveError must be a list of size equal to the number of systems {}".format(self.nSystems))
            assert (np.all(values > 0.0)), ValueError("additiveErr must be > 0.0. Make sure the values are in linear space")
            # assert (isinstance(relativeErr[i], float) or isinstance(relativeErr[i], np.ndarray)), TypeError(
            #     "relativeErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))

        self._addErr = StatArray.StatArray(values, '$\epsilon_{Additive}$', self.units)

    @property
    def channelNames(self):
        return self._channelNames

    @channelNames.setter
    def channelNames(self, values):
        if values is None:
            self._channelNames = ['Channel {}'.format(i) for i in range(self.nChannels)]
        else:
            assert len(values) == self.nChannels, Exception("Length of channelNames must equal total number of channels {}".format(self.nChannels))
            self._channelNames = values

    @property
    def systemOffset(self):
        return np.hstack([0, np.cumsum(self.channels_per_system)])

    @property
    def fiducial(self):
        return self._fiducial

    @fiducial.setter
    def fiducial(self, value):
        self._fiducial = StatArray.StatArray(value, 'fiducial')

    @property
    def lineNumber(self):
        return self._lineNumber

    @lineNumber.setter
    def lineNumber(self, value):
        self._lineNumber = StatArray.StatArray(np.float64(value), 'Line number')

    @property
    def channels_per_system(self):
        return self._channels_per_system

    @property
    def nSystems(self):
        return np.size(self.channels_per_system)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, values):

        if values is None:
            values = self.nChannels
        else:
            assert np.size(values) == self.nChannels, ValueError("data must have size {}".format(self.nChannels))

        self._data = StatArray.StatArray(values, "Data", self.units)

    @property
    def deltaD(self):
        """Get the difference between the predicted and observed data,

        .. math::
            \delta \mathbf{d} = \mathbf{d}^{pre} - \mathbf{d}^{obs}.

        Returns
        -------
        out : StatArray
            The residual between the active observed and predicted data
            with size equal to the number of active channels.

        """
        return StatArray.StatArray(self.predictedData - self.data, name="$\\mathbf{Fm} - \\mathbf{d}_{obs}$", units=self.units)

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, value):
        if value is None:
            value = 1
        self._elevation = StatArray.StatArray(value, "Elevation", "m")

    @property
    def nActiveChannels(self):
        return self.active.size

    @property
    def nChannels(self):
        return np.sum(self.channels_per_system)

    @property
    def predictedData(self):
        """The predicted data. """
        return self._predictedData


    @predictedData.setter
    def predictedData(self, values):
        if values is None:
            values = self.nChannels
        else:
            if isinstance(values, list):
                assert len(values) == self.nSystems, ValueError("predictedData as a list must have {} elements".format(self.nSystems))
                values = np.hstack(values)
            assert values.size == self.nChannels, ValueError("Size of predictedData must equal total number of time channels {}".format(self.nChannels))
            # Mask invalid data values less than 0.0 to NaN
        self._predictedData = StatArray.StatArray(values, "Predicted Data", self.units)

    @property
    def relative_error(self):
        return self._relErr

    @property
    def relErr(self):
        return self._relErr

    @relErr.setter
    def relErr(self, values):
        if values is None:
            values = self.nSystems
        else:
            if np.size(values) == 1:
                values = np.full(self.nSystems, fill_value=values)
            else:
                values = np.asarray(values)
            assert np.size(values) == self.nSystems, ValueError("relErr must be a list of size equal to the number of systems {}".format(self.nSystems))
            assert (np.all(values > 0.0)), ValueError("relErr must be > 0.0.")
            # assert (isinstance(additiveErr[i], float) or isinstance(additiveErr[i], np.ndarray)), TypeError(
            #     "additiveErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))

        self._relErr = StatArray.StatArray(values, '$\epsilon_{Relative}x10^{2}$', '%')


    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value):
        if value is None:
            value = np.ones(self.nChannels)
        else:
            if isinstance(value, list):
                assert len(value) == self.nSystems, ValueError("std as a list must have {} elements".format(self.nSystems))
                value = np.hstack(value)
            assert value.size == self.nChannels, ValueError("Size of std must equal total number of time channels {}".format(nChannels))

        self._std = StatArray.StatArray(value, "Standard deviation", self.units)

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if value is None:
            value = ""
        else:
            assert isinstance(value, str), TypeError('units must have type str')
        self._units = value

    def generate_noise(self, additive_error, relative_error):

        std = np.sqrt(additive_error**2.0 + (relative_error * self.predictedData)**2.0)
        return np.random.randn(self.nChannels) * std

    def init_posterior_plots(self, gs):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if isinstance(gs, matplotlib.figure.Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        splt = gs.subgridspec(2, 2, width_ratios=[1, 4], height_ratios=[2, 1], wspace=0.3)
        ax = []
        # Height axis
        ax.append(plt.subplot(splt[0, 0]))
        # Data axis
        ax.append(plt.subplot(splt[0, 1]))

        splt2 = splt[1, :].subgridspec(self.nSystems, 2, wspace=0.2)
        # Relative error axes
        ax.append([plt.subplot(splt2[i, 0]) for i in range(self.nSystems)])
        # Additive Error axes
        ax.append([plt.subplot(splt2[i, 1]) for i in range(self.nSystems)])

        return ax

    def plot_posteriors(self, axes=None, height_kwargs={}, data_kwargs={}, rel_error_kwargs={}, add_error_kwargs={}, **kwargs):

        assert len(axes) == 4, ValueError("Must have length 3 list of axes for the posteriors. self.init_posterior_plots can generate them")

        best = kwargs.pop('best', None)
        if not best is None:
            height_kwargs['line'] = best.z
            rel_error_kwargs['line'] = best.relErr
            add_error_kwargs['line'] = best.addErr

        height_kwargs['rotate'] = height_kwargs.get('rotate', True)
        self.z.plotPosteriors(ax = axes[0], **height_kwargs)

        self.plot(ax=axes[1], **data_kwargs)
        self.plotPredicted(color=cP.wellSeparated[0], ax=axes[1], **data_kwargs)
        if not best is None:
            best.plotPredicted(color=cP.wellSeparated[3], ax=axes[1], **data_kwargs)

        # data_kwargs['noColorbar'] = data_kwargs.get('noColorbar', True)
        # ax.append(self.predictedData.plotPosteriors(ax = axes[1], **data_kwargs))

        self.relErr.plotPosteriors(ax=axes[2], **rel_error_kwargs)
        self.addErr.plotPosteriors(ax=axes[3], **add_error_kwargs)

    def weightingMatrix(self, power=1.0):
        """Return a diagonal data weighting matrix of the reciprocated data standard deviations."""
        return np.diag(1.0 / self.std[self.active])

    def _systemIndices(self, system=0):
        """The slice indices for the requested system.

        Parameters
        ----------
        system : int
            Requested system index.

        Returns
        -------
        out : numpy.slice
            The slice pertaining to the requested system.

        """

        assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))
        return np.s_[self.systemOffset[system]:self.systemOffset[system+1]]


    def likelihood(self, log):
        """Compute the likelihood of the current predicted data given the observed data and assigned errors

        Returns
        -------
        out : np.float64
            Likelihood of the data point

        """
        return self.predictedData.probability(i=self.active, log=log)

    def dataMisfit(self):
        """Compute the :math:`L_{2}` norm squared misfit between the observed and predicted data

        .. math::
            \| \mathbf{W}_{d} (\mathbf{d}^{obs}-\mathbf{d}^{pre})\|_{2}^{2},

        where :math:`\mathbf{W}_{d}` are the reciprocal data errors.

        Parameters
        ----------
        squared : bool
            Return the squared misfit.

        Returns
        -------
        out : np.float64
            The misfit value.

        """
        # The data misfit is the mahalanobis distance of the multivariate distance.
        # assert not any(self.std[self.active] <= 0.0), ValueError('Cannot compute the misfit when the data standard deviations are zero.')
        tmp2 = 1.0 / self.std[self.active]
        return np.sqrt(np.float64(np.sum((cf.Ax(tmp2, self.deltaD[self.active]))**2.0, dtype=np.float64)))


    # def scaleJ(self, Jin, power=1.0):
    #     """ Scales a matrix by the errors in the given data

    #     Useful if a sensitivity matrix is generated using one data point, but must be scaled by the errors in another.

    #     Parameters
    #     ----------
    #     Jin : array_like
    #         2D array representing a matrix
    #     power : float
    #         Power to raise the error level too. Default is -1.0

    #     Returns
    #     -------
    #     Jout : array_like
    #         Matrix scaled by the data errors

    #     Raises
    #     ------
    #     ValueError
    #         If the number of rows in Jin do not match the number of active channels in the datapoint

    #     """
    #     assert Jin.shape[0] == self.nActiveChannels, ValueError("Number of rows of Jin must match the number of active channels in the datapoint {}".format(self.nActiveChannels))

    #     Jout = np.zeros(Jin.shape)
    #     Jout[:, :] = Jin * (np.repeat(self.std[self.active, np.newaxis]**-power, Jout.shape[1], 1))
    #     return Jout

    @property
    def summary(self):
        """ Print a summary of the EMdataPoint """
        msg = ('Data Point: \n'
               'Channel Names {} \n'
               'x: {} \n'
               'y: {} \n'
               'z: {} \n'
               'elevation: {} \n'
               'Number of active channels: {} \n'
               '{} {} {}').format(self._channelNames, self.x, self.y, self.z, self.elevation, self.nActiveChannels, self.data[self.active].summary, self.predictedData[self.active].summary, self.std[self.active].summary)
        return msg



    def createHdf(self, parent, myName, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = super().createHdf(parent, myName, withPosterior, nRepeats, fillvalue)

        grp.create_dataset('channels_per_system', data=self.channels_per_system)

        self.fiducial.createHdf(grp, 'fiducial', nRepeats=nRepeats, fillvalue=fillvalue)
        self.lineNumber.createHdf(grp, 'line_number', nRepeats=nRepeats, fillvalue=fillvalue)
        self.elevation.createHdf(grp, 'e', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.data.createHdf(grp, 'd', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.std.createHdf(grp, 's', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.predictedData.createHdf(grp, 'p', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)

        # if not self.errorPosterior is None:
        #     for i, x in enumerate(self.errorPosterior):
        #         x.createHdf(grp, 'joint_error_posterior_{}'.format(i), nRepeats=nRepeats, fillvalue=fillvalue)
            # self.relErr.setPosterior([self.errorPosterior[i].marginalize(axis=1) for i in range(self.nSystems)])
            # self.addErr.setPosterior([self.errorPosterior[i].marginalize(axis=0) for i in range(self.nSystems)])

        self.relErr.createHdf(grp, 'relErr', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.addErr.createHdf(grp, 'addErr', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)

        return grp


    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        super().writeHdf(parent, name, withPosterior, index)

        grp = parent[name]

        self.fiducial.writeHdf(grp, 'fiducial', index=index)
        self.lineNumber.writeHdf(grp, 'line_number', index=index)
        self.elevation.writeHdf(grp, 'e',  withPosterior=withPosterior, index=index)
        self.data.writeHdf(grp, 'd',  withPosterior=withPosterior, index=index)
        self.std.writeHdf(grp, 's',  withPosterior=withPosterior, index=index)
        self.predictedData.writeHdf(grp, 'p',  withPosterior=withPosterior, index=index)

        # if not self.errorPosterior is None:
        #     for i, x in enumerate(self.errorPosterior):
        #         x.writeHdf(grp, 'joint_error_posterior_{}'.format(i), index=index)
            # self.relative_error.setPosterior([self.errorPosterior[i].marginalize(axis=1) for i in range(self.nSystems)])
            # self.additive_error.setPosterior([self.errorPosterior[i].marginalize(axis=0) for i in range(self.nSystems)])

        self.relErr.writeHdf(grp, 'relErr',  withPosterior=withPosterior, index=index)
        self.addErr.writeHdf(grp, 'addErr',  withPosterior=withPosterior, index=index)

    @classmethod
    def fromHdf(cls, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        out = super(DataPoint, cls).fromHdf(grp, index=index)

        out.errorPosterior = None

        if 'fiducial' in grp:
            out.fiducial = StatArray.StatArray.fromHdf(grp['fiducial'], index=index)

        if 'line_number' in grp:
            out.lineNumber = StatArray.StatArray.fromHdf(grp['line_number'], index=index)

        out.elevation = StatArray.StatArray.fromHdf(grp['e'], index=index)

        if 'channels_per_system' in grp:
            out._nChannelsPerSystem = np.asarray(grp['channels_per_system'])

        out._data = StatArray.StatArray.fromHdf(grp['d'], index=index)
        out._std = StatArray.StatArray.fromHdf(grp['s'], index=index)
        out._predictedData = StatArray.StatArray.fromHdf(grp['p'], index=index)

        # if 'joint_error_posterior_0' in grp:
        #     i = 0
        #     self.errorPosterior = []
        #     while 'joint_error_posterior_{}'.format(i) in grp:
        #         self.errorPosterior.append(Histogram2D.fromHdf(grp['joint_error_posterior_{}'.format(i)], index=index))
        #         i += 1

        out._relErr = StatArray.StatArray.fromHdf(grp['relErr'], index=index)
        out._addErr = StatArray.StatArray.fromHdf(grp['addErr'], index=index)

        return out


    def Isend(self, dest, world):
        myMPI.Isend(self.channels_per_system, dest=dest, world=world)
        tmp = np.hstack([self.x, self.y, self.z, self.elevation])
        myMPI.Isend(tmp, dest=dest, world=world)
        self._data.Isend(dest, world)
        self._std.Isend(dest, world)
        self._predictedData.Isend(dest, world)
        world.isend(self._channelNames, dest=dest)


    @classmethod
    def Irecv(cls, source, world):
        ncps = myMPI.Irecv(source=source, world=world)
        tmp = myMPI.Irecv(source=source, world=world)
        d = StatArray.StatArray.Irecv(source, world)
        s = StatArray.StatArray.Irecv(source, world)
        p = StatArray.StatArray.Irecv(source, world)
        cn = world.irecv(source = 0).wait()

        return cls(ncps,
                   x=tmp[0], y=tmp[1], z=tmp[2], elevation=tmp[3],
                   data=d, std=s, predictedData=p,
                   channelNames=cn)

