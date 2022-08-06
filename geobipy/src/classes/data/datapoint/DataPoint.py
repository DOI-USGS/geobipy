from abc import abstractmethod
from cached_property import cached_property
from copy import copy, deepcopy
from ...pointcloud.Point import Point
from ....classes.core import StatArray
from ....base import utilities as cf
from ....base import plotting as cP
from ....base import MPI as myMPI
from ...statistics.Distribution import Distribution
from ...statistics.Histogram import Histogram
from ...mesh.RectilinearMesh1D import RectilinearMesh1D
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
                       data=None, std=None, predictedData=None,
                       units=None, channelNames=None,
                       lineNumber=0.0, fiducial=0.0, **kwargs):
        """ Initialize the Data class """

        super().__init__(x, y, z, elevation=elevation, **kwargs)

        self.units = units

        # StatArray of data
        self.data = data

        self.std = std

        self.predictedData = predictedData

        self.lineNumber = lineNumber

        self.fiducial = fiducial

        self.channelNames = channelNames

        self.relative_error = None
        self.additive_error = None

        # self.errorPosterior = None

    @cached_property
    def active(self):
        """Gets the indices to the observed data values that are not NaN

        Returns
        -------
        out : array of ints
            Indices into the observed data that are not NaN

        """
        return ~np.isnan(self.data)

    @property
    def additive_error(self):
        return self._additive_error

    @additive_error.setter
    def additive_error(self, values):
        if values is None:
            values = self.nSystems
        else:
            assert np.size(values) == self.nSystems, ValueError("additive_error must have size 1")
            # assert (np.all(np.asarray(values) > 0.0)), ValueError("additiveErr must be > 0.0. Make sure the values are in linear space")
            # assert (isinstance(relativeErr[i], float) or isinstance(relativeErr[i], np.ndarray)), TypeError(
            #     "relativeErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))

        self._additive_error = StatArray.StatArray(values, '$\epsilon_{Additive}$', self.units)

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
    def n_posteriors(self):
        return self._n_error_posteriors + self.z.hasPosterior

    @property
    def _n_error_posteriors(self):
        # if not self.errorPosterior is None:
        #     return len(self.errorPosterior)
        # else:
        return self.nSystems * np.sum([x.hasPosterior for x in [self.relative_error, self.additive_error]])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, values):
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
    def n_active_channels(self):
        return self.active.size

    @property
    def nChannels(self):
        return self.data.size

    @property
    def nSystems(self):
        return 1

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
                assert len(values) == self.nSystems, ValueError("std as a list must have {} elements".format(self.nSystems))
                values = np.hstack(values)
            assert np.size(values) == self.nChannels, ValueError("Size of std must equal total number of time channels {}".format(self.nChannels))

        self._predictedData = StatArray.StatArray(values, "Predicted Data", self.units)

    @property
    def relative_error(self):
        return self._relative_error

    @relative_error.setter
    def relative_error(self, values):

        if values is None:
            values = np.full(self.nSystems, fill_value=0.01)
        else:
            assert np.size(values) == self.nSystems, ValueError("relative_error must be a list of size equal to the number of systems {}".format(self.nSystems))
            # assert (np.all(np.asarray(values) > 0.0)), ValueError("relative_error must be > 0.0.")
            # assert (isinstance(additiveErr[i], float) or isinstance(additiveErr[i], np.ndarray)), TypeError(
            #     "additiveErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))

        self._relative_error = StatArray.StatArray(values, '$\epsilon_{Relative}x10^{2}$', '%')

    @property
    def std(self):
        """ Compute the data errors. """

        assert self.relative_error > 0.0, ValueError("relative_error must be > 0.0")

        # For each system assign error levels using the user inputs
        variance = ((self.relative_error * self.data)**2.0) + (self.additive_error**2.0)
        self._std[:] = np.sqrt(variance)

        # Update the variance of the predicted data prior
        if self.predictedData.hasPrior:
            self.predictedData.prior.variance[np.diag_indices(np.sum(self.active))] = variance[self.active]

        return self._std

    @std.setter
    def std(self, value):

        if value is None:
            value = np.full(self.nChannels, fill_value=0.01)
        else:
            if isinstance(value, list):
                assert len(value) == self.nSystems, ValueError("std as a list must have {} elements".format(self.nSystems))
                value = np.hstack(value)
            assert np.size(value) == self.nChannels, ValueError("Size of std must equal total number of time channels {}".format(self.nChannels))

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

    def __deepcopy__(self, memo={}):

        out = super().__deepcopy__(memo)

        out._components = deepcopy(self._components, memo)
        out._channels_per_system = deepcopy(self.channels_per_system, memo)

        out._units = deepcopy(self.units, memo)
        out._data = deepcopy(self.data, memo)
        out._relative_error = deepcopy(self.relative_error, memo)
        out._additive_error = deepcopy(self.additive_error, memo)
        out._std = deepcopy(self.std, memo)
        out._predictedData = deepcopy(self.predictedData, memo)
        out._lineNumber = deepcopy(self.lineNumber, memo)
        out._fiducial = deepcopy(self.fiducial, memo)
        out._channelNames = deepcopy(self.channelNames, memo)
        
        # out._errorPosterior = deepcopy(self.errorPosterior, memo)

        return out

    # def initialize(self, **kwargs):
    
    #     self.relative_error[:] = self.kwargs['initial_relative_error']
    #     self.additive_error[:] = self.kwargs['initial_additive_error']

    def generate_noise(self, additive_error, relative_error):

        std = np.sqrt(additive_error**2.0 + (relative_error * self.predictedData)**2.0)
        return np.random.randn(self.nChannels) * std

    def prior_derivative(self, order, model=None):

        if order == 1:
            return np.dot(self.J[self.active, :].T, self.predictedData.priorDerivative(order=1, i=self.active))
        elif order == 2:
            J = self.sensitivity(model)[self.active, :]
            WdT_Wd = self.predictedData.priorDerivative(order=2)
            return np.dot(J.T, np.dot(WdT_Wd, J))

    def _init_posterior_plots(self, gs):
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
        ax.append(self.z._init_posterior_plots(splt[0, 0]))
        # Data axis
        ax.append(plt.subplot(splt[0, 1]))

        # Relative error axes
        ax.append(self.relative_error._init_posterior_plots(splt[1, 0]))
        # Additive Error axes
        ax.append(self.additive_error._init_posterior_plots(splt[1, 1]))

        return ax

    def plot_posteriors(self, axes=None, height_kwargs={}, data_kwargs={}, rel_error_kwargs={}, add_error_kwargs={}, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', plt.gcf())
            
        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes)

        assert len(axes) == 4, ValueError("Must have length 3 list of axes for the posteriors. self.init_posterior_plots can generate them")

        best = kwargs.pop('best', None)
        if not best is None:
            height_kwargs['line'] = best.z
            rel_error_kwargs['line'] = best.relative_error
            add_error_kwargs['line'] = best.additive_error

        height_kwargs['transpose'] = height_kwargs.get('transpose', True)
        self.z.plotPosteriors(ax = axes[0], **height_kwargs)

        axes[1].cla()
        self.predictedData.plotPosteriors(ax = axes[1], colorbar=False, **data_kwargs)
        self.plot(ax=axes[1], **data_kwargs)
        
        c = cP.wellSeparated[0] if best is None else cP.wellSeparated[3]
        self.plotPredicted(color=c, ax=axes[1], **data_kwargs)

        self.relative_error.plotPosteriors(ax=axes[2], **rel_error_kwargs)
        self.additive_error.plotPosteriors(ax=axes[3], **add_error_kwargs)

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
        # The data misfit is the mahalanobis distance of the multivariate distribution.
        # assert not any(self.std[self.active] <= 0.0), ValueError('Cannot compute the misfit when the data standard deviations are zero.')
        tmp2 = 1.0 / self.std[self.active]
        misfit = np.float64(np.sum((cf.Ax(tmp2, self.deltaD[self.active]))**2.0, dtype=np.float64))
        return misfit

    def initialize(self, **kwargs):
        self.relative_error = kwargs['initial_relative_error']
        self.additive_error = kwargs['initial_additive_error']


    def set_priors(self, height_prior=None, relative_error_prior=None, additive_error_prior=None, data_prior=None, **kwargs):

        if height_prior is None:
            if kwargs.get('solve_height', False):
                height_prior = Distribution('Uniform', self.z - kwargs['maximum_height_change'], self.z + kwargs['maximum_height_change'], prng=kwargs.get('prng'))

        # Define prior, proposal, posterior for relative error
        if relative_error_prior is None:
            if kwargs.get('solve_relative_error', False):
                relative_error_prior = Distribution('Uniform', kwargs['minimum_relative_error'], kwargs['maximum_relative_error'], prng=kwargs.get('prng'))

        # Define prior, proposal, posterior for additive error
        if additive_error_prior is None:
            if kwargs.get('solve_additive_error', False):
                # log = Trisinstance(self, TdemDataPoint)
                additive_error_prior = Distribution('Uniform', kwargs['minimum_additive_error'], kwargs['maximum_additive_error'], log=False, prng=kwargs.get('prng'))
        
        if data_prior is None:
            data_prior = Distribution('MvLogNormal', self.data[self.active], self.std[self.active]**2.0, linearSpace=False, prng=kwargs.get('prng'))
        
        self.set_height_prior(height_prior)
        self.set_relative_error_prior(relative_error_prior)
        self.set_additive_error_prior(additive_error_prior)
        self.set_data_prior(data_prior)
        

    def set_height_prior(self, prior):
        if not prior is None:
            self.z.prior = prior

    def set_data_prior(self, prior):
        if not prior is None:
            self.predictedData.prior = prior

    def set_relative_error_prior(self, prior):
        if not prior is None:
            assert prior.ndim == self.nSystems, ValueError("relative_error_prior must have {} dimensions".format(self.nSystems))
            self.relative_error.prior = prior

    def set_additive_error_prior(self, prior):
        if not prior is None:
            assert prior.ndim == self.nSystems, ValueError("additive_error_prior must have {} dimensions".format(self.nSystems))
            self.additive_error.prior = prior

    def set_proposals(self, height_proposal=None, relative_error_proposal=None, additive_error_proposal=None, **kwargs):
        """Set the proposals on the datapoint's perturbable parameters

        Parameters
        ----------
        heightProposal : geobipy.baseDistribution, optional
            The proposal to attach to the height. Must be univariate
        relativeErrorProposal : geobipy.baseDistribution, optional
            The proposal to attach to the relative error.
            If the datapoint has only one system, relativeErrorProposal is univariate.
            If there are more than one system, relativeErrorProposal is multivariate.
        additiveErrorProposal : geobipy.baseDistribution, optional
            The proposal to attach to the relative error.
            If the datapoint has only one system, additiveErrorProposal is univariate.
            If there are more than one system, additiveErrorProposal is multivariate.

        """

        self.set_height_proposal(height_proposal, **kwargs)
        self.set_relative_error_proposal(relative_error_proposal, **kwargs)
        self.set_additive_error_proposal(additive_error_proposal, **kwargs)


    def set_height_proposal(self, proposal, **kwargs):

        if proposal is None:
            if kwargs.get('solve_height', False):
                proposal = Distribution('Normal', self.z.value, kwargs['height_proposal_variance'], prng=kwargs['prng'])

        self.z.proposal = proposal

    def set_relative_error_proposal(self, proposal, **kwargs):
        if proposal is None:
            if kwargs.get('solve_relative_error', False):
                proposal = Distribution('MvNormal', self.relative_error, kwargs['relative_error_proposal_variance'], prng=kwargs['prng'])
        self.relative_error.proposal = proposal

    def set_additive_error_proposal(self, proposal, **kwargs):
        if proposal is None:
            if kwargs.get('solve_additive_error', False):
                proposal = Distribution('MvNormal', self.additive_error, kwargs['additive_error_proposal_variance'], linearSpace=False, prng=kwargs['prng'])

        self.additive_error.proposal = proposal

    def set_posteriors(self, log=None):
        """ Set the posteriors based on the attached priors

        Parameters
        ----------
        log :

        """
        # Create a histogram to set the height posterior.
        self.set_height_posterior()
        # # Initialize the histograms for the relative errors
        # # Set the posterior for the data point.
        self.set_relative_error_posterior()
        self.set_additive_error_posterior(log=log)

        # self.set_predicted_data_posterior()

    def set_height_posterior(self):
        """

        """
        if self.z.hasPrior:
            mesh = RectilinearMesh1D(edges = StatArray.StatArray(self.z.prior.bins(), name=self.z.name, units=self.z.units), relativeTo=self.z)
            self.z.posterior = Histogram(mesh=mesh)

    def set_relative_error_posterior(self):
        """

        """
        if self.relative_error.hasPrior:
            bins = StatArray.StatArray(np.atleast_2d(self.relative_error.prior.bins()), name=self.relative_error.name, units=self.relative_error.units)        
            posterior = []
            for i in range(self.nSystems):
                b = bins[i, :]
                mesh = RectilinearMesh1D(edges = b, relativeTo=0.5*(b.max()-b.min()))
                posterior.append(Histogram(mesh=mesh))
            self.relative_error.posterior = posterior

    def set_additive_error_posterior(self, log=None):
        """

        """
        if self.additive_error.hasPrior:
            bins = StatArray.StatArray(np.atleast_2d(self.additive_error.prior.bins()), name=self.additive_error.name, units=self.data.units)

            posterior = []
            for i in range(self.nSystems):
                b = bins[i, :]
                mesh = RectilinearMesh1D(edges = b, log=log, relativeTo=0.5*(b.max()-b.min()))
                posterior.append(Histogram(mesh=mesh))
            self.additive_error.posterior = posterior

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
        msg = super().summary
        names = copy(self.channelNames)
        j = np.arange(5, self.nChannels, 5)
        for i in range(j.size):
            names.insert(j[i]+i, '\n')

        msg += "channel names:\n{}\n".format("|   "+(', '.join(names).replace("\n,", "\n|  ")))
        msg += "data:\n{}".format("|   "+(self.data[self.active].summary.replace("\n", "\n|   "))[:-4])
        msg += "predicted data:\n{}".format("|   "+(self.predictedData[self.active].summary.replace("\n", "\n|   "))[:-4])
        msg += "std:\n{}".format("|   "+(self.std[self.active].summary.replace("\n", "\n|   "))[:-4])
        msg += "line number:\n{}".format("|   "+(self.lineNumber.summary.replace("\n", "\n|   "))[:-4])
        msg += "fiducial:\n{}".format("|   "+(self.fiducial.summary.replace("\n", "\n|   "))[:-4])
        msg += "relative error:\n{}".format("|   "+(self.relative_error.summary.replace("\n", "\n|   "))[:-4])
        msg += "additive error:\n{}".format("|   "+(self.additive_error.summary.replace("\n", "\n|   "))[:-4])

        return msg

    def createHdf(self, parent, myName, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = super().createHdf(parent, myName, withPosterior, add_axis, fillvalue)

        self.fiducial.createHdf(grp, 'fiducial', add_axis=add_axis, fillvalue=fillvalue)
        self.lineNumber.createHdf(grp, 'line_number', add_axis=add_axis, fillvalue=fillvalue)
        self.data.createHdf(grp, 'data', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.std.createHdf(grp, 'std', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.predictedData.createHdf(grp, 'predicted_data', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.relative_error.createHdf(grp, 'relative_error', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.additive_error.createHdf(grp, 'additive_error', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)


        # if not self.errorPosterior is None:
        #     for i, x in enumerate(self.errorPosterior):
        #         x.createHdf(grp, 'joint_error_posterior_{}'.format(i), add_axis=add_axis, fillvalue=fillvalue)
            # self.relative_error.setPosterior([self.errorPosterior[i].marginalize(axis=1) for i in range(self.nSystems)])
            # self.additive_error.setPosterior([self.errorPosterior[i].marginalize(axis=0) for i in range(self.nSystems)])

        if add_axis is not None:
            grp.attrs['repr'] = 'Data'
            
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
        self.data.writeHdf(grp, 'data',  withPosterior=withPosterior, index=index)
        self.std.writeHdf(grp, 'std',  withPosterior=withPosterior, index=index)
        self.predictedData.writeHdf(grp, 'predicted_data',  withPosterior=withPosterior, index=index)

        # if not self.errorPosterior is None:
        #     for i, x in enumerate(self.errorPosterior):
        #         x.writeHdf(grp, 'joint_error_posterior_{}'.format(i), index=index)
            # self.relative_error.setPosterior([self.errorPosterior[i].marginalize(axis=1) for i in range(self.nSystems)])
            # self.additive_error.setPosterior([self.errorPosterior[i].marginalize(axis=0) for i in range(self.nSystems)])

        self.relative_error.writeHdf(grp, 'relative_error',  withPosterior=withPosterior, index=index)
        self.additive_error.writeHdf(grp, 'additive_error',  withPosterior=withPosterior, index=index)

    @classmethod
    def fromHdf(cls, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        out = super(DataPoint, cls).fromHdf(grp, index=index, **kwargs)

        if 'fiducial' in grp:
            out.fiducial = StatArray.StatArray.fromHdf(grp['fiducial'], index=index)
        if 'line_number' in grp:
            out.lineNumber = StatArray.StatArray.fromHdf(grp['line_number'], index=index)

        out.data = StatArray.StatArray.fromHdf(grp['data'], index=index)
        out.std = StatArray.StatArray.fromHdf(grp['std'], index=index)
        out.predictedData = StatArray.StatArray.fromHdf(grp['predicted_data'], index=index)

        # if 'joint_error_posterior_0' in grp:
        #     i = 0
        #     self.errorPosterior = []
        #     while 'joint_error_posterior_{}'.format(i) in grp:
        #         self.errorPosterior.append(Histogram2D.fromHdf(grp['joint_error_posterior_{}'.format(i)], index=index))
        #         i += 1

        out.relative_error = StatArray.StatArray.fromHdf(grp['relative_error'], index=index)
        out.additive_error = StatArray.StatArray.fromHdf(grp['additive_error'], index=index)

        return out


    def Isend(self, dest, world):

        super().Isend(dest, world)

        self.lineNumber.Isend(dest, world)
        self.fiducial.Isend(dest, world)

        self.relative_error.Isend(dest, world)
        self.additive_error.Isend(dest, world)
        # self.std.Isend(dest, world)


    @classmethod
    def Irecv(cls, source, world, **kwargs):

        out = super(DataPoint, cls).Irecv(source, world, **kwargs)

        out._lineNumber = StatArray.StatArray.Irecv(source, world)
        out._fiducial = StatArray.StatArray.Irecv(source, world)

        out._relative_error = StatArray.StatArray.Irecv(source, world)
        out._additive_error = StatArray.StatArray.Irecv(source, world)

        # out._std = StatArray.StatArray.Irecv(source, world)

        return out
