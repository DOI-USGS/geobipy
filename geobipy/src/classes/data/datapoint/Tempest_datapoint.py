from copy import deepcopy

from ....classes.core import StatArray
from .TdemDataPoint import TdemDataPoint
from ...forwardmodelling.Electromagnetic.TD.tdem1d import (tdem1dfwd, tdem1dsen)
from ...statistics.Histogram import Histogram
from ...model.Model import Model
from ...mesh.RectilinearMesh1D import RectilinearMesh1D
from ...mesh.RectilinearMesh2D import RectilinearMesh2D
from ...statistics.Distribution import Distribution
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ....base import utilities as cf
from ....base import plotting as cP


class Tempest_datapoint(TdemDataPoint):
    """ Initialize a Tempest Time domain data point


    TdemDataPoint(x, y, z, elevation, data, std, system, transmitter_loop, receiver_loop, lineNumber, fiducial)

    Parameters
    ----------
    x : np.float64
        The easting co-ordinate of the data point
    y : np.float64
        The northing co-ordinate of the data point
    z : np.float64
        The height of the data point above ground
    elevation : np.float64, optional
        The elevation of the data point, default is 0.0
    data : list of arrays, optional
        A list of 1D arrays, where each array contains the data in each system.
        The arrays are vertically concatenated inside the TdemDataPoint object
    std : list of arrays, optional
        A list of 1D arrays, where each array contains the errors in each system.
        The arrays are vertically concatenated inside the TdemDataPoint object
    system : TdemSystem, optional
        Time domain system class
    transmitter_loop : EmLoop, optional
        Transmitter loop class
    receiver_loop : EmLoop, optional
        Receiver loop class
    lineNumber : float, optional
        The line number associated with the datapoint
    fiducial : float, optional
        The fiducial associated with the datapoint

    Returns
    -------
    out : TdemDataPoint
        A time domain EM sounding

    Notes
    -----
    The data argument is a set of lists with length equal to the number of systems.
    These data are unpacked and vertically concatenated in this class.
    The parameter self._data will have length equal to the sum of the number of time gates in each system.
    The same is true for the errors, and the predicted data vector.

    """

    @TdemDataPoint.additive_error.setter
    def additive_error(self, values):
        if values is None:
            values = self.nChannels
        else:
            assert np.size(values) == self.nChannels, ValueError(("Tempest data must a have additive error values for all time gates and all components. \n"
                                                              "additive_error must have size {}").format(self.nChannels))

        self._additive_error = StatArray.StatArray(values, '$\epsilon_{additive}$', self.units)

    @TdemDataPoint.data.getter
    def data(self):
        for j in range(self.nSystems):
            for i in range(self.n_components):
                ic = self._component_indices(i, j)
                self._data[ic] = self.primary_field[i] + self.secondary_field[ic]
        return self._data

    @TdemDataPoint.predictedData.getter
    def predictedData(self):
        for j in range(self.nSystems):
            for i in range(self.n_components):
                ic = self._component_indices(i, j)
                self._predictedData[ic] = self.predicted_primary_field[i] + self.predicted_secondary_field[ic]
        return self._predictedData

    @TdemDataPoint.relative_error.setter
    def relative_error(self, values):
        if values is None:
            values = np.full(self.n_components * self.nSystems, fill_value=0.01)
        else:
            assert np.size(values) == self.n_components * self.nSystems, ValueError(("Tempest data must a have relative error for the primary and secondary fields, for each system. \n"
                            "relative_error must have size {}").format(self.n_components * self.nSystems))

        assert np.all(values > 0.0), ValueError("Relative error {} must be > 0.0".format(values))

        self._relative_error = StatArray.StatArray(values, '$\epsilon_{Relative}$', '%')

    @TdemDataPoint.std.getter
    def std(self):
        """ Updates the data errors

        Assumes a t^-0.5 behaviour e.g. logarithmic gate averaging
        V0 is assumed to be ln(Error @ 1ms)

        Parameters
        ----------
        relativeErr : list of scalars or list of array_like
            A fraction percentage that is multiplied by the observed data. The list should have length equal to the number of systems. The entries in each item can be scalar or array_like.
        additiveErr : list of scalars or list of array_like
            An absolute value of additive error. The list should have length equal to the number of systems. The entries in each item can be scalar or array_like.

        Raises
        ------
        TypeError
            If relativeErr or additiveErr is not a list
        TypeError
            If the length of relativeErr or additiveErr is not equal to the number of systems
        TypeError
            If any item in the relativeErr or additiveErr lists is not a scalar or array_like of length equal to the number of time channels
        ValueError
            If any relative or additive errors are <= 0.0
        """

        assert np.all(self.relative_error > 0.0), ValueError('relative_error must be > 0.0')

        # For each system assign error levels using the user inputs
        for i in range(self.nSystems):
            for j in range(self.n_components):
                ic = self._component_indices(j, i)
                relative_error = self.relative_error[(i*self.n_components)+j] * self.secondary_field[ic]
                variance = relative_error**2.0 + self.additive_error[i]**2.0
                self._std[ic] = np.sqrt(variance)


        # Update the variance of the predicted data prior
        if self.predictedData.hasPrior:
            self.predictedData.prior.variance[np.diag_indices(np.sum(self.active))] = self._std[self.active]**2.0

        return self._std

    @TdemDataPoint.units.setter
    def units(self, value):
        if value is None:
            value = r"fT"
        else:
            assert isinstance(value, str), TypeError(
                'units must have type str')
        self._units = value

    def halfspace_misfit(self, conductivity_range, n_samples=100, pitch_range=None):
        assert conductivity_range[1] > conductivity_range[0], ValueError("Maximum conductivity must be greater than the minimum")
        conductivity = RectilinearMesh1D(centres = np.logspace(*(np.log10(conductivity_range)), n_samples+1), log=10)

        if pitch_range is None:
            misfit = Model(mesh=conductivity)
            model = self.new_model()

            for i in range(conductivity.nCells):
                model.values[0] = conductivity.centres_absolute[i]
                self.forward(model)
                misfit.values[i] = self.dataMisfit()

        else:
            pitch = RectilinearMesh1D(centres = np.linspace(*pitch_range, n_samples))
            misfit = Model(mesh = RectilinearMesh2D(x=conductivity, y=pitch))

            model = self.new_model()
            for i in range(conductivity.nCells):
                model.values[0] = conductivity.centres_absolute[i]
                for j in range(pitch.nCells):
                    self.transmitter.pitch = pitch.centres[j]

                    self.forward(model)
                    misfit.values[i, j] = self.dataMisfit()

        return misfit

    def find_best_halfspace(self, conductivity=1.0, pitch=0.0):
        """Computes the best value of a half space that fits the data.

        Carries out a brute force search of the halfspace conductivity that best fits the data.
        The profile of data misfit vs halfspace conductivity is not quadratic, so a bisection will not work.

        Parameters
        ----------
        minConductivity : float, optional
            The minimum conductivity to search over
        maxConductivity : float, optional
            The maximum conductivity to search over
        nSamples : int, optional
            The number of values between the min and max

        Returns
        -------
        out : np.float64
            The best fitting log10 conductivity for the half space

        """
        from scipy.optimize import minimize

        dp = deepcopy(self)

        def minimize_me(x):
            model = dp.new_model()
            model.values[0] = x[0]
            dp.transmitter.pitch = x[1]
            dp.forward(model)
            return dp.dataMisfit()

        out = minimize(minimize_me, [conductivity, pitch], method='Nelder-Mead', bounds=((0.0, np.inf),(-90.0, 90.0)))

        model = self.new_model()
        model.values[0] = out.x[0]
        self.transmitter.pitch = out.x[1]

        return model

    def initialize(self, **kwargs):
        super().initialize(**kwargs)

        if 'initial_transmitter_pitch' in kwargs:
            self.transmitter.pitch = kwargs['initial_transmitter_pitch']

    def _init_posterior_plots(self, gs):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if isinstance(gs, matplotlib.figure.Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        n = np.sum([self.relative_error.hasPosterior, self.additive_error.hasPosterior, self.transmitter.pitch.hasPosterior])

        splt = gs.subgridspec(2, n, height_ratios=[2, 1], wspace=0.3)
        ax = []
        # Height axis
        ax.append(self.z._init_posterior_plots(splt[0, :1]))
        # Data axis
        ax.append(plt.subplot(splt[0, 1:]))

        i = 0
        # Relative error axes
        tmp = self.relative_error._init_posterior_plots(splt[1, i])
        if tmp is not None:
            i += 1
        ax.append(tmp)

        # Additive Error axes
        tmp = self.additive_error._init_posterior_plots(splt[1, i])
        if tmp is not None:
            i += 1
            for j in range(self.nSystems):
                others = np.s_[(j * self.n_components):(j * self.n_components)+self.n_components]
                tmp[1].get_shared_y_axes().join(tmp[1], *tmp[others])
        ax.append(tmp)

        # Pitch axes
        tmp = self.transmitter.pitch._init_posterior_plots(splt[1, i])
        ax.append(tmp)

        return ax

    def perturb(self):
        """Propose a new EM data point given the specified attached propsal distributions

        Parameters
        ----------
        height : bool
            Propose a new observation height.
        relative_error : bool
            Propose a new relative error.
        additive_error : bool
            Propose a new additive error.
        pitch : bool
            Propose new pitch.

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
        self.perturb_pitch()

    def perturb_pitch(self):
        if self.transmitter.pitch.hasProposal:
            # Generate a new error
            self.transmitter.pitch.perturb(imposePrior=True)
            # Update the mean of the proposed errors
            self.transmitter.pitch.proposal.mean = self.transmitter.pitch

    def plotWaveform(self,**kwargs):
        for i in range(self.nSystems):
            if (self.nSystems > 1):
                plt.subplot(2, 1, i + 1)
            plt.plot(self.system[i].waveform.time, self.system[i].waveform.current, **kwargs)
            cP.xlabel('Time (s)')
            cP.ylabel('Normalized Current (A)')
            plt.margins(0.1, 0.1)

    def plot(self, **kwargs):
        kwargs['xscale'] = kwargs.get('xscale', 'log')
        kwargs['yscale'] = kwargs.get('yscale', 'linear')
        return super().plot(**kwargs)

    def plot_posteriors(self, axes=None, height_kwargs={}, data_kwargs={}, rel_error_kwargs={}, add_error_kwargs={}, pitch_kwargs={}, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', plt.gcf())

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes)

        assert len(axes) == 5, ValueError("Must have length 5 list of axes for the posteriors. self.init_posterior_plots can generate them")

        best = kwargs.pop('best', None)
        if not best is None:
            height_kwargs['line'] = best.z
            rel_error_kwargs['line'] = best.relative_error
            # add_error_kwargs['line'] = best.additive_error
            pitch_kwargs['line'] = best.transmitter.pitch

        height_kwargs['transpose'] = height_kwargs.get('transpose', True)
        self.z.plotPosteriors(ax = axes[0], **height_kwargs)

        axes[1].clear()
        self.predictedData.plotPosteriors(ax = axes[1], colorbar=False, **data_kwargs)
        self.plot(ax=axes[1], **data_kwargs)

        c = cP.wellSeparated[0] if best is None else cP.wellSeparated[3]
        self.plotPredicted(color=c, ax=axes[1], **data_kwargs)

        self.relative_error.plotPosteriors(ax=axes[2], **rel_error_kwargs)

        add_error_kwargs['colorbar'] = False
        self.additive_error.plotPosteriors(ax=axes[3], **add_error_kwargs)

        self.transmitter.pitch.plotPosteriors(ax = axes[4], **pitch_kwargs)

    def plotPredicted(self, **kwargs):
        kwargs['xscale'] = kwargs.get('xscale', 'log')
        kwargs['yscale'] = kwargs.get('yscale', 'linear')
        return super().plotPredicted(**kwargs)

    def plot_secondary_field(self, title='Secondary field', **kwargs):

        ax = kwargs.pop('ax', None)
        ax = plt.gca() if ax is None else plt.sca(ax)
        plt.cla()

        kwargs['marker'] = kwargs.pop('marker', 'v')
        kwargs['markersize'] = kwargs.pop('markersize', 7)
        c = kwargs.pop('color', [cP.wellSeparated[i+1]
                       for i in range(self.nSystems)])
        mfc = kwargs.pop('markerfacecolor', [
                         cP.wellSeparated[i+1] for i in range(self.nSystems)])
        assert len(c) == self.nSystems, ValueError(
            "color must be a list of length {}".format(self.nSystems))
        assert len(mfc) == self.nSystems, ValueError(
            "markerfacecolor must be a list of length {}".format(self.nSystems))
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor', 'k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth', 1.0)
        kwargs['alpha'] = kwargs.pop('alpha', 0.8)
        kwargs['linestyle'] = kwargs.pop('linestyle', 'none')
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)

        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'linear')

        logx = kwargs.pop('logX', None)
        logy = kwargs.pop('logY', None)

        for i in range(self.nSystems):
            system_times, _ = cf._log(self.off_time(i), logx)
            for j in range(self.n_components):
                ic = self._component_indices(j, i)
                self.secondary_field[ic].plot(x=system_times, **kwargs)

    def plot_predicted_secondary_field(self, title='Secondary field', **kwargs):
        ax = kwargs.pop('ax', None)
        ax = plt.gca() if ax is None else plt.sca(ax)

        noLabels = kwargs.pop('nolabels', False)

        if (not noLabels):
            cP.xlabel('Time (s)')
            cP.ylabel(cf.getNameUnits(self.predictedData))
            cP.title(title)

        kwargs['color'] = kwargs.pop('color', cP.wellSeparated[3])
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)
        kwargs['alpha'] = kwargs.pop('alpha', 0.7)
        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'linear')

        logx = kwargs.pop('logX', None)
        logy = kwargs.pop('logY', None)

        for i in range(self.nSystems):
            system_times, _ = cf._log(self.off_time(i), logx)
            for j in range(self.n_components):
                ic = self._component_indices(j, i)
                self.predicted_secondary_field[ic].plot(x=system_times, **kwargs)

    def set_priors(self, height_prior=None, relative_error_prior=None, additive_error_prior=None, transmitter_pitch_prior=None, data_prior=None, **kwargs):

        super().set_priors(height_prior, relative_error_prior, additive_error_prior, data_prior, **kwargs)

        if transmitter_pitch_prior is None:
            if kwargs.get('solve_transmitter_pitch', False):
                transmitter_pitch_prior = Distribution('Uniform',
                                                        self.transmitter.pitch - kwargs['maximum_transmitter_pitch_change'],
                                                        self.transmitter.pitch + kwargs['maximum_transmitter_pitch_change'],
                                                        prng=kwargs['prng'])

        self.transmitter.set_priors(pitch_prior=transmitter_pitch_prior)

    def set_relative_error_prior(self, prior):
        if not prior is None:
            assert prior.ndim == self.n_components * self.nSystems, ValueError("relative_error_prior must have {} dimensions".format(self.n_components * self.nSystems))
            self.relative_error.prior = prior

    def set_additive_error_prior(self, prior):
        if not prior is None:
            assert prior.ndim == self.nChannels, ValueError("additive_error_prior must have {} dimensions".format(self.nChannels))
            self.additive_error.prior = prior

    def set_proposals(self, height_proposal=None, relative_error_proposal=None, additive_error_proposal=None, transmitter_pitch_proposal=None, **kwargs):

        super().set_proposals(height_proposal, relative_error_proposal, additive_error_proposal, **kwargs)

        if transmitter_pitch_proposal is None:
            if kwargs.get('solve_transmitter_pitch', False):
                transmitter_pitch_proposal = Distribution('Normal', self.transmitter.pitch.value, kwargs['transmitter_pitch_proposal_variance'], prng=kwargs['prng'])

        self.transmitter.set_proposals(pitch_proposal=transmitter_pitch_proposal)

    def set_posteriors(self, log=None):

        super().set_posteriors(log=log)

        self.transmitter.set_posteriors()

    def set_relative_error_posterior(self):

        if self.relative_error.hasPrior:
            bins = StatArray.StatArray(np.atleast_2d(self.relative_error.prior.bins()), name=self.relative_error.name, units=self.relative_error.units)
            posterior = []
            for i in range(self.nSystems*self.n_components):
                b = bins[i, :]
                mesh = RectilinearMesh1D(edges = b, relativeTo=0.5*(b.max()-b.min()))
                posterior.append(Histogram(mesh=mesh))
            self.relative_error.posterior = posterior

    def set_additive_error_posterior(self, log=None):
        if self.additive_error.hasPrior:
            bins = RectilinearMesh1D(edges=StatArray.StatArray(self.additive_error.prior.bins()[0, :], name=self.additive_error.name, units=self.data.units), log=10)
            posterior = []
            for j in range(self.nSystems):
                system_times = RectilinearMesh1D(centres=self.off_time(j), log=10)
                for k in range(self.n_components):
                    # icomp = self._component_indices(k, j)
                    mesh = RectilinearMesh2D(x=system_times, y=bins)
                    posterior.append(Histogram(mesh=mesh))
            self.additive_error.posterior = posterior

    def update_posteriors(self):

        super().update_posteriors()

        if self.transmitter.pitch.hasPosterior:
            self.transmitter.pitch.updatePosterior()

    def update_additive_error_posterior(self):
        if self.additive_error.hasPosterior:
            i = 0
            for j in range(self.nSystems):
                system_times = self.off_time(j)
                for k in range(self.n_components):
                    icomp = self._component_indices(k, j)
                    self.additive_error.posterior[i].update(x=system_times, y=self.additive_error[icomp])
                    i += 1

    def _empymodForward(self, mod):

        print('stuff')

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):

        grp = super().createHdf(parent, name, withPosterior, add_axis, fillvalue)

        if add_axis is not None:
            grp.attrs['repr'] = 'TempestData'

        return grp
