from abc import abstractmethod
from cached_property import cached_property
from copy import copy, deepcopy

from numpy import arange, argwhere, asarray, atleast_2d, diag_indices
from numpy import dot, float64, full, hstack, isnan
from numpy import s_, size, squeeze, sqrt, sum
from numpy import all as npall

from numpy.random import randn

from ...pointcloud.Point import Point
from ....classes.core import StatArray
from ....base import utilities as cf
from ....base import plotting as cP
from ....base import MPI as myMPI
from ...statistics.Distribution import Distribution
from ...statistics.Histogram import Histogram
from ...mesh.RectilinearMesh1D import RectilinearMesh1D

from matplotlib.figure import Figure
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
    channel_names : list of str, optional
        Names of each channel of length sum(nChannelsPerSystem)

    """
    __slots__ = ('_units', '_data', '_std', '_predictedData', '_lineNumber', '_fiducial', '_channel_names',
                 '_relative_error', '_additive_error', '_sensitivity_matrix', '_components')

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=None,
                       data=None, std=None, predictedData=None,
                       units=None, channel_names=None,
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

        self.channel_names = channel_names

        self.relative_error = None
        self.additive_error = None

        self._sensitivity_matrix = None

    @cached_property
    def active(self):
        """Gets the indices to the observed data values that are not NaN

        Returns
        -------
        out : array of ints
            Indices into the observed data that are not NaN

        """
        return ~isnan(self.data)

    @cached_property
    def active_system_indices(self):
        out =  squeeze(argwhere([any(self.active[i]) for i in self.system_indices]))
        return out

    @property
    def line_number(self):
        return self._lineNumber

    @property
    def additive_error(self):
        return self._additive_error

    @additive_error.setter
    def additive_error(self, values):
        if values is None:
            values = self.nSystems
        else:
            assert size(values) == self.nSystems, ValueError("additive_error must have size 1")
            # assert (npall(asarray(values) > 0.0)), ValueError("additiveErr must be > 0.0. Make sure the values are in linear space")
            # assert (isinstance(relativeErr[i], float) or isinstance(relativeErr[i], ndarray)), TypeError(
            #     "relativeErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))

        self._additive_error = StatArray.StatArray(values, '$\epsilon_{Additive}$', self.units)

    @property
    def addressof(self):
        """ Print a summary of the EMdataPoint """
        msg = super().addressof
        msg += "data:\n{}".format("|   "+(self.data.addressof.replace("\n", "\n|   "))[:-4])
        msg += "predicted data:\n{}".format("|   "+(self.predictedData.addressof.replace("\n", "\n|   "))[:-4])
        msg += "std:\n{}".format("|   "+(self.std.addressof.replace("\n", "\n|   "))[:-4])
        msg += "line number:\n{}".format("|   "+(self.lineNumber.addressof.replace("\n", "\n|   "))[:-4])
        msg += "fiducial:\n{}".format("|   "+(self.fiducial.addressof.replace("\n", "\n|   "))[:-4])
        msg += "relative error:\n{}".format("|   "+(self.relative_error.addressof.replace("\n", "\n|   "))[:-4])
        msg += "additive error:\n{}".format("|   "+(self.additive_error.addressof.replace("\n", "\n|   "))[:-4])
        msg += "sensitivitiy matrix:\n{}".format("|   "+(self.sensitivity_matrix.addressof.replace("\n", "\n|   "))[:-4])

        return msg

    @property
    def channel_names(self):
        return self._channel_names

    @channel_names.setter
    def channel_names(self, values):
        if values is None:
            self._channel_names = ['Channel {}'.format(i) for i in range(self.nChannels)]
        else:
            assert len(values) == self.nChannels, Exception("Length of channel_names must equal total number of channels {}".format(self.nChannels))
            self._channel_names = values

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
        self._lineNumber = StatArray.StatArray(float64(value), 'Line number')

    @property
    def n_posteriors(self):
        return super().n_posteriors + self._n_error_posteriors

    @property
    def _n_error_posteriors(self):
        # if not self.errorPosterior is None:
        #     return len(self.errorPosterior)
        # else:
        return self.nSystems * sum([x.hasPosterior for x in [self.relative_error, self.additive_error]])

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

    # @property
    # def elevation(self):
    #     return self._elevation

    # @elevation.setter
    # def elevation(self, value):
    #     if value is None:
    #         value = 1
    #     self._elevation = StatArray.StatArray(value, "Elevation", "m")

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
                values = hstack(values)
            assert size(values) == self.nChannels, ValueError("Size of std must equal total number of time channels {}".format(self.nChannels))

        self._predictedData = StatArray.StatArray(values, "Predicted Data", self.units)

    @property
    def relative_error(self):
        return self._relative_error

    @relative_error.setter
    def relative_error(self, values):

        if values is None:
            values = full(self.nSystems, fill_value=0.01)
        else:
            assert size(values) == self.nSystems, ValueError("relative_error must be a list of size equal to the number of systems {}".format(self.nSystems))
            # assert (npall(asarray(values) > 0.0)), ValueError("relative_error must be > 0.0.")
            # assert (isinstance(additiveErr[i], float) or isinstance(additiveErr[i], ndarray)), TypeError(
            #     "additiveErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))

        assert npall(values > 0.0), ValueError("Relative error {} must be > 0.0".format(values))

        self._relative_error = StatArray.StatArray(values, '$\epsilon_{Relative}x10^{2}$', '%')

    @property
    def sensitivity_matrix(self):
        return self._sensitivity_matrix

    @property
    def std(self):
        """ Compute the data errors. """

        assert self.relative_error > 0.0, ValueError("relative_error must be > 0.0")

        # For each system assign error levels using the user inputs
        variance = ((self.relative_error * self.data)**2.0) + (self.additive_error**2.0)
        self._std[:] = sqrt(variance)

        # Update the variance of the predicted data prior
        if self.predictedData.hasPrior:
            self.predictedData.prior.variance[diag_indices(sum(self.active))] = variance[self.active]

        return self._std

    @std.setter
    def std(self, value):

        if value is None:
            value = full(self.nChannels, fill_value=0.01)
        else:
            if isinstance(value, list):
                assert len(value) == self.nSystems, ValueError("std as a list must have {} elements".format(self.nSystems))
                value = hstack(value)
            assert size(value) == self.nChannels, ValueError("Size of std {} must equal total number of time channels {}".format(size(value), self.nChannels))

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
        out._channel_names = deepcopy(self.channel_names, memo)

        out._sensitivity_matrix = deepcopy(self._sensitivity_matrix)

        return out

    # def initialize(self, **kwargs):

    #     self.relative_error[:] = self.kwargs['initial_relative_error']
    #     self.additive_error[:] = self.kwargs['initial_additive_error']

    def generate_noise(self, additive_error, relative_error):

        std = sqrt(additive_error**2.0 + (relative_error * self.predictedData)**2.0)
        return randn(self.nChannels) * std

    def prior_derivative(self, order):

        J = self.sensitivity_matrix[self.active, :]

        if order == 1:
            return dot(J.T, self.predictedData.priorDerivative(order=1, i=self.active))

        elif order == 2:
            WdT_Wd = self.predictedData.priorDerivative(order=2)
            return dot(J.T, dot(WdT_Wd, J))

    @property
    def probability(self):
        """Evaluate the probability for the EM data point given the specified attached priors

        Parameters
        ----------
        rEerr : bool
            Include the relative error when evaluating the prior
        aEerr : bool
            Include the additive error when evaluating the prior
        height : bool
            Include the elevation when evaluating the prior
        calibration : bool
            Include the calibration parameters when evaluating the prior
        verbose : bool
            Return the components of the probability, i.e. the individually evaluated priors

        Returns
        -------
        out : float64
            The evaluation of the probability using all assigned priors

        Notes
        -----
        For each boolean, the associated prior must have been set.

        Raises
        ------
        TypeError
            If a prior has not been set on a requested parameter

        """
        probability = super().probability

        if self.relative_error.hasPrior:  # Relative Errors
            probability += self.relative_error.probability(log=True, active=self.active_system_indices)

        if self.additive_error.hasPrior:  # Additive Errors
            probability += self.additive_error.probability(log=True, active=self.active_system_indices)

        # P_calibration = float64(0.0)
        # if calibration:  # Calibration parameters
        #     P_calibration = self.calibration.probability(log=True)
        #     probability += P_calibration
        return probability

    def _init_posterior_plots(self, gs=None):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if gs is None:
            gs = plt.figure()

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        # Split the top and bottom
        splt = gs.subgridspec(2, 2, width_ratios=[1, 4], height_ratios=[2, 1], wspace=0.3)
        ax = []
        # Point posteriors
        ax.append(super()._init_posterior_plots(splt[0, 0]))
        # Data axis
        ax.append(plt.subplot(splt[0, 1]))

        # Relative error axes
        ax.append(self.relative_error._init_posterior_plots(splt[1, 0]))
        # Additive Error axes
        ax.append(self.additive_error._init_posterior_plots(splt[1, 1]))

        return ax

    def plot_posteriors(self, axes=None, data_kwargs={}, rel_error_kwargs={}, add_error_kwargs={}, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', plt.gcf())

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes)


        assert len(axes) == 4, ValueError("Must have length 3 list of axes for the posteriors. self.init_posterior_plots can generate them")

        overlay = kwargs.pop('overlay', None)
        if not overlay is None:
            rel_error_kwargs['overlay'] = overlay.relative_error
            add_error_kwargs['overlay'] = overlay.additive_error

        super().plot_posteriors(axes=axes[0], **kwargs)

        axes[1].cla()
        self.predictedData.plot_posteriors(ax = axes[1], colorbar=False, **data_kwargs)
        self.plot(ax=axes[1], **data_kwargs)

        if overlay is None:
            c = cP.wellSeparated[0]
            self.plot_predicted(color=c, ax=axes[1], **data_kwargs)
        else:
            c = cP.wellSeparated[3]
            overlay.plot_predicted(color=c, ax=axes[1], **data_kwargs)

        self.relative_error.plot_posteriors(ax=axes[2], **rel_error_kwargs)
        self.additive_error.plot_posteriors(ax=axes[3], **add_error_kwargs)

    @property
    def system_indices(self):
        return tuple([s_[self.systemOffset[system]:self.systemOffset[system+1]] for system in arange(self.nSystems)])

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
        return self.system_indices[system]


    def likelihood(self, log):
        """Compute the likelihood of the current predicted data given the observed data and assigned errors

        Returns
        -------
        out : float64
            Likelihood of the data point

        """
        return self.predictedData.probability(i=self.active, log=log)

    def data_misfit(self):
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
        out : float64
            The misfit value.

        """
        # The data misfit is the mahalanobis distance of the multivariate distribution.
        # assert not any(self.std[self.active] <= 0.0), ValueError('Cannot compute the misfit when the data standard deviations are zero.')
        tmp2 = 1.0 / self.std[self.active]
        misfit = float64(sum((cf.Ax(tmp2, self.deltaD[self.active]))**2.0, dtype=float64))
        return misfit

    def initialize(self, **kwargs):
        self.relative_error = kwargs['initial_relative_error']
        self.additive_error = kwargs['initial_additive_error']

    def perturb(self):
        """Propose a new EM data point given the specified attached propsal distributions

        Parameters
        ----------
        newHeight : bool
            Propose a new observation height.
        newRelativeError : bool
            Propose a new relative error.
        newAdditiveError : bool
            Propose a new additive error.

        newCalibration : bool
            Propose new calibration parameters.

        Returns
        -------
        out : subclass of EmDataPoint
            The proposed data point

        Notes
        -----
        For each boolean, the associated proposal must have been set.

        Raises
        ------
        TypeError
            If a proposal has not been set on a requested parameter

        """
        super().perturb()

        if self.relative_error.hasProposal:
            # Generate a new error
            self.relative_error.perturb(imposePrior=True, log=True, i=self.active_system_indices)
            # Update the mean of the proposed errors
            self.relative_error.proposal.mean = self.relative_error

        if self.additive_error.hasProposal:
            # Generate a new error
            self.additive_error.perturb(imposePrior=True, log=True, i=self.active_system_indices)
            # Update the mean of the proposed errors
            self.additive_error.proposal.mean = self.additive_error

    def set_priors(self, relative_error_prior=None, additive_error_prior=None, data_prior=None, **kwargs):

        super().set_priors(**kwargs)

        # Define prior, proposal, posterior for relative error
        if relative_error_prior is None:
            if kwargs.get('solve_relative_error', False):
                relative_error_prior = Distribution('Uniform', kwargs['minimum_relative_error'], kwargs['maximum_relative_error'], log=True, prng=kwargs.get('prng'))

        # Define prior, proposal, posterior for additive error
        if additive_error_prior is None:
            if kwargs.get('solve_additive_error', False):
                # log = Trisinstance(self, TdemDataPoint)
                additive_error_prior = Distribution('Uniform', kwargs['minimum_additive_error'], kwargs['maximum_additive_error'], log=True, prng=kwargs.get('prng'))

        if data_prior is None:
            data_prior = Distribution('MvNormal', self.data[self.active], self.std[self.active]**2.0, prng=kwargs.get('prng'))

        self.set_relative_error_prior(relative_error_prior)
        self.set_additive_error_prior(additive_error_prior)
        self.set_data_prior(data_prior)

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

    def set_proposals(self, relative_error_proposal=None, additive_error_proposal=None, **kwargs):
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
        super().set_proposals(**kwargs)
        self.set_relative_error_proposal(relative_error_proposal, **kwargs)
        self.set_additive_error_proposal(additive_error_proposal, **kwargs)

    def set_relative_error_proposal(self, proposal, **kwargs):
        if proposal is None:
            if kwargs.get('solve_relative_error', False):
                proposal = Distribution('MvLogNormal', self.relative_error, kwargs['relative_error_proposal_variance'], linearSpace=True, prng=kwargs.get('prng'))
        self.relative_error.proposal = proposal

    def set_additive_error_proposal(self, proposal, **kwargs):
        if proposal is None:
            if kwargs.get('solve_additive_error', False):
                proposal = Distribution('MvLogNormal', self.additive_error, kwargs['additive_error_proposal_variance'], linearSpace=True, prng=kwargs.get('prng'))

        self.additive_error.proposal = proposal

    def reset_posteriors(self):
        self.z.reset_posteriors()
        self.relative_error.reset_posteriors()
        self.additive_error.reset_posteriors()

    def set_posteriors(self, log=10):
        """ Set the posteriors based on the attached priors

        Parameters
        ----------
        log :

        """
        # Create a histogram to set the height posterior.
        super().set_posteriors()
        # # Initialize the histograms for the relative errors
        # # Set the posterior for the data point.
        self.set_relative_error_posterior()
        self.set_additive_error_posterior(log=log)

        # self.set_predicted_data_posterior()

    def set_relative_error_posterior(self):
        """

        """
        if self.relative_error.hasPrior:
            bins = StatArray.StatArray(atleast_2d(self.relative_error.prior.bins()), name=self.relative_error.name, units=self.relative_error.units)
            posterior = []
            for i in range(self.nSystems):
                b = bins[i, :]
                mesh = RectilinearMesh1D(edges = b, relativeTo=0.5*(b.max()-b.min()), log=10)
                posterior.append(Histogram(mesh=mesh))
            self.relative_error.posterior = posterior

    def set_additive_error_posterior(self, log=10):
        """

        """
        if self.additive_error.hasPrior:
            bins = StatArray.StatArray(atleast_2d(self.additive_error.prior.bins()), name=self.additive_error.name, units=self.data.units)

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

    #     Jout = zeros(Jin.shape)
    #     Jout[:, :] = Jin * (repeat(self.std[self.active, newaxis]**-power, Jout.shape[1], 1))
    #     return Jout

    @property
    def summary(self):
        """ Print a summary of the EMdataPoint """
        msg = super().summary
        names = copy(self.channel_names)
        j = arange(5, self.nChannels, 5)
        for i in range(j.size):
            names.insert(j[i]+i, '\n')

        msg += "channel names:\n{}\n".format("|   "+(', '.join(names).replace("\n,", "\n|  ")))
        msg += "data:\n{}\n".format("|   "+(self.data[self.active].summary.replace("\n", "\n|   "))[:-4])
        msg += "predicted data:\n{}\n".format("|   "+(self.predictedData[self.active].summary.replace("\n", "\n|   "))[:-4])
        msg += "std:\n{}\n".format("|   "+(self.std[self.active].summary.replace("\n", "\n|   "))[:-4])
        msg += "line number:\n{}\n".format("|   "+(self.lineNumber.summary.replace("\n", "\n|   "))[:-4])
        msg += "fiducial:\n{}\n".format("|   "+(self.fiducial.summary.replace("\n", "\n|   "))[:-4])
        msg += "relative error:\n{}\n".format("|   "+(self.relative_error.summary.replace("\n", "\n|   "))[:-4])
        msg += "additive error:\n{}\n".format("|   "+(self.additive_error.summary.replace("\n", "\n|   "))[:-4])

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

        if 'components' in grp:
            out._components = asarray(grp['components'])

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

    @classmethod
    def Irecv(cls, source, world, **kwargs):

        out = super(DataPoint, cls).Irecv(source, world, **kwargs)

        out._lineNumber = StatArray.StatArray.Irecv(source, world)
        out._fiducial = StatArray.StatArray.Irecv(source, world)

        out._relative_error = StatArray.StatArray.Irecv(source, world)
        out._additive_error = StatArray.StatArray.Irecv(source, world)

        return out
