from copy import deepcopy

from numpy import any, atleast_2d, diag, diag_indices, dot
from numpy import full, linspace, log10, logspace
from numpy import ones, s_, size, sqrt, sum, squeeze, zeros
from numpy import all as npall
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplot, gcf, gca, sca, cla, plot, margins

from ...core.DataArray import DataArray
from ...statistics.StatArray import StatArray
from .TdemDataPoint import TdemDataPoint
from ...forwardmodelling.Electromagnetic.TD.tdem1d import (
    tdem1dfwd, tdem1dsen, ga_fm_dlogc)
from ...statistics.Distribution import Distribution
from ...statistics.Histogram import Histogram
from ...model.Model import Model
from ...mesh.RectilinearMesh1D import RectilinearMesh1D
from ...mesh.RectilinearMesh2D import RectilinearMesh2D
from ...system.Loop_pair import Loop_pair

from ....base import utilities as cf
from ....base import plotting as cp


class Tempest_datapoint(TdemDataPoint):
    """ Initialize a Tempest Time domain data point

    TdemDataPoint(x, y, z, elevation, data, std, system, transmitter_loop, receiver_loop, line_number, fiducial)

    Parameters
    ----------
    x : float64
        The easting co-ordinate of the data point
    y : float64
        The northing co-ordinate of the data point
    z : float64
        The height of the data point above ground
    elevation : float64, optional
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
    line_number : float, optional
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
    __slots__ = ('_additive_error_multiplier', '_reference_additive_error')
    _total_field = None

    def __init__(self, *args, additive_error_multiplier=None, total_field=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.reference_additive_error = None
        self.additive_error_multiplier = additive_error_multiplier

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out._reference_additive_error = deepcopy(self._reference_additive_error)
        out._additive_error_multiplier = deepcopy(self.additive_error_multiplier, memo=memo)
        return out

    @property
    def n_data_channels(self):
        return sum(self.nTimes)

    @property
    def reference_additive_error(self):
        return self._reference_additive_error

    @reference_additive_error.setter
    def reference_additive_error(self, values):
        if values is None:
            values = self.n_data_channels
        else:
            assert size(values) == self.n_data_channels, ValueError(("additive_error must have size {}").format(self.n_data_channels))

        self._reference_additive_error = StatArray(values, r'$epsilon_{additive}$', self.units)

    @property
    def additive_error(self):
        self._additive_error[:] = self.additive_error_multiplier * self.reference_additive_error
        return self._additive_error

    @additive_error.setter
    def additive_error(self, values):
        if values is None:
            values = self.n_data_channels
        else:
            assert size(values) == self.n_data_channels, ValueError(("additive_error must have size {}").format(self.n_data_channels))

        self._additive_error = StatArray(values, r'$epsilon_{additive}$', self.units)

    @property
    def additive_error_multiplier(self):
        return self._additive_error_multiplier

    @additive_error_multiplier.setter
    def additive_error_multiplier(self, values):
        if values is None:
            self._additive_error_multiplier = StatArray(ones(self.nSystems), 'Multiplier')
        else:
            assert size(values) == self.nSystems, ValueError(f'additive_error_multiplier must have size {self.nSystems} but has size {size(values)}')
            self._additive_error_multiplier = StatArray(values, 'Multiplier')

    @TdemDataPoint.data.getter
    def data(self):
        self._data[:] = 0.0
        for i in range(self.n_components):
            ic = self._component_indices(i, 0)
            # Compute Sum(Pc + Sc) for c in x, y, z
            self._data[:] += (self.primary_field[i] + self.secondary_field[ic])**2.0
        self._data[:] = sqrt(self._data)
        return self._data

    @TdemDataPoint.predicted_data.setter
    def predicted_data(self, values):
        if values is None:
            values = self.n_data_channels
        else:
            assert size(values) == self.n_data_channels, ValueError(f"data must have size {self.n_data_channels} not {size(values)}")

        self._predicted_data = DataArray(values, "Predicted total field", self.units)

    @TdemDataPoint.predicted_data.getter
    def predicted_data(self):
        self._predicted_data[:] = 0.0
        for i in range(self.n_components):
            ic = self._component_indices(i, 0)
            dic = s_[:]
            self._predicted_data[dic] += (self.predicted_primary_field[i] + self.predicted_secondary_field[ic])**2.0
        self._predicted_data[:] = sqrt(self._predicted_data)
        return self._predicted_data

    @TdemDataPoint.relative_error.setter
    def relative_error(self, values):
        if values is None:
            values = full(self.nSystems, fill_value=0.01)
        else:
            assert size(values) == self.nSystems, ValueError((f"relative_error must have size {self.nSystems}"))

        assert npall(values > 0.0), ValueError(f"Relative error {values} must be > 0.0")

        self._relative_error = StatArray(values, r'$\epsilon_{Relative}$', '%')

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

        assert npall(self.relative_error > 0.0), ValueError('relative_error must be > 0.0')
        # For each system assign error levels using the user inputs
        relative_error = self.relative_error * self.data
        variance = relative_error**2.0 + self.additive_error**2.0
        self._std[:] = sqrt(variance)

        # Update the variance of the predicted data prior
        if self.predicted_data.hasPrior:
            self.predicted_data.prior.variance = self._std[self.active]**2.0

        return self._std

    @property
    def total_field(self):
        return self._total_field

    @total_field.setter
    def total_field(self, value):
        assert isinstance(value, bool), ValueError("total_field must have type bool")
        self._total_field = value

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
        conductivity = RectilinearMesh1D(centres = logspace(*(log10(conductivity_range)), n_samples+1), log=10)

        if pitch_range is None:
            misfit = Model(mesh=conductivity)
            model = self.new_model

            for i in range(conductivity.nCells):
                model.values[0] = conductivity.centres_absolute[i]
                self.forward(model)
                misfit.values[i] = self.data_misfit()

        else:
            pitch = RectilinearMesh1D(centres = linspace(*pitch_range, n_samples))
            misfit = Model(mesh = RectilinearMesh2D(x=conductivity, y=pitch))

            model = self.new_model
            for i in range(conductivity.nCells):
                model.values[0] = conductivity.centres_absolute[i]
                for j in range(pitch.nCells):
                    self.receiver.pitch = pitch.centres[j]

                    self.forward(model)
                    misfit.values[i, j] = self.data_misfit()

        return misfit

    # def find_best_halfspace(self):
    #     """Computes the best value of a half space that fits the data.

    #     Carries out a brute force search of the halfspace conductivity that best fits the data.
    #     The profile of data misfit vs halfspace conductivity is not quadratic, so a bisection will not work.

    #     Parameters
    #     ----------
    #     minConductivity : float, optional
    #         The minimum conductivity to search over
    #     maxConductivity : float, optional
    #         The maximum conductivity to search over
    #     nSamples : int, optional
    #         The number of values between the min and max

    #     Returns
    #     -------
    #     out : float64
    #         The best fitting log10 conductivity for the half space

    #     """
    #     from scipy.optimize import minimize

    #     dp = deepcopy(self)
    #     # dp.relative_error[:] = 0.01
    #     # dp.additive_error[:] = 0.0

    #     model = dp.new_model

    #     def minimize_me(x):
    #         model.values[0] = x[0]
    #         dp.receiver.pitch = x[1]
    #         dp.forward(model)
    #         return dp.data_misfit()

    #     conductivities = logspace(-8, 5, 12)
    #     pitch = linspace(-20.0, 20.0, 12)

    #     out = empty((conductivities.size, 3))

    #     for i in range(conductivities.size):
    #         tmp = minimize(minimize_me, [conductivities[i], pitch[i]], method='Nelder-Mead', bounds=((0.0, inf),(-90.0, 90.0)), options={'maxiter':10000, 'fatol':1e-6})
    #         out[i, :] = tmp.x[0], tmp.x[1], tmp.fun

    #     j = argmin(out[:, 2])

    #     # tmp = minimize(minimize_me, [out[j, 0], out[j, 1]], method='Nelder-Mead', bounds=((0.0, inf),(-90.0, 90.0)), options={'maxiter':10000, 'fatol':1e-12, 'xatol':1e-6})
    #     # print(tmp)

    #     model.values[0] = out[j, 0]
    #     self.receiver.pitch = out[j, 1]

    #     return model

    def initialize(self, **kwargs):
        super().initialize(**kwargs)

        if 'initial_receiver_pitch' in kwargs:
            self.receiver.pitch = kwargs['initial_receiver_pitch']

        self.reference_additive_error = kwargs['initial_additive_error']

    def _init_posterior_plots(self, gs=None):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if gs is None:
            gs = figure()

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        n_rows = 1

        if (self.relative_error.hasPosterior & self.additive_error_multiplier.hasPosterior) or self.loop_pair.hasPosterior:
            n_rows = 2

        splt = gs.subgridspec(n_rows, 1, wspace=0.3)

        n_cols = 1
        width_ratios = None
        if self.relative_error.hasPosterior or self.additive_error_multiplier.hasPosterior:
            n_cols = 2
            width_ratios = (1, 2)

        ## Top row of plot
        splt_top = splt[0].subgridspec(1, n_cols, width_ratios=width_ratios)

        ax = {}
        # Data axis
        ax['data'] = subplot(splt_top[-1])

        # Place relative error posteriors next to the data.
        if self.relative_error.hasPosterior:
            # Relative error axes
            ax['relative_error'] = self.relative_error._init_posterior_plots(splt_top[0])
        else: # If no relative error, move additive errors up.
            if self.additive_error_multiplier.hasPosterior:
                ax['additive_error_multiplier'] = self.additive_error_multiplier._init_posterior_plots(splt_top[0])

        ## Bottom row of plot
        n_cols = sum([self.relative_error.hasPosterior & self.additive_error_multiplier.hasPosterior, self.transmitter.hasPosterior, super(Loop_pair, self.loop_pair).hasPosterior, self.receiver.hasPosterior])

        if n_cols > 0:
            splt_bottom = splt[1].subgridspec(1, n_cols)

            i = 0
            # Additive Error axes
            if self.relative_error.hasPosterior & self.additive_error_multiplier.hasPosterior:
                ax['additive_error_multiplier'] = self.additive_error_multiplier._init_posterior_plots(splt_bottom[i])
                i += 1

            # Loop pair
            if self.loop_pair.hasPosterior:
                ax['loop_pair'] = self.loop_pair._init_posterior_plots(splt_bottom[i:])

        return ax

    def perturb(self):
        super().perturb()
        if self.additive_error_multiplier.hasProposal:
            self.additive_error_multiplier.perturb()

    def plotWaveform(self,**kwargs):
        for i in range(self.nSystems):
            if (self.nSystems > 1):
                subplot(2, 1, i + 1)
            plot(self.system[i].waveform.time, self.system[i].waveform.current, **kwargs)
            cp.xlabel('Time (s)')
            cp.ylabel('Normalized Current (A)')
            margins(0.1, 0.1)

    def plot(self, title='Tempest data', with_error_bars=True, **kwargs):
        """ Plot the Inphase and Quadrature Data for an EM measurement
        """
        ax = kwargs.pop('ax', None)
        ax = plt.gca() if ax is None else ax

        markers = tuple(kwargs.pop('marker', ('o', 'x', 'v')))
        kwargs['markersize'] = kwargs.pop('markersize', 3)
        c = kwargs.pop('color', [cp.wellSeparated[i+1] for i in range(self.nSystems)])
        mfc = kwargs.pop('markerfacecolor', [cp.wellSeparated[i+1] for i in range(self.nSystems)])
        assert len(c) == self.nSystems, ValueError("color must be a list of length {}".format(self.nSystems))
        assert len(mfc) == self.nSystems, ValueError("markerfacecolor must be a list of length {}".format(self.nSystems))
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor', 'k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth', 1.0)
        kwargs['alpha'] = kwargs.pop('alpha', 0.8)
        kwargs['linestyle'] = kwargs.pop('linestyle', 'none')
        kwargs['linewidth'] = kwargs.pop('linewidth', 1)

        kwargs.pop('logX', None)
        kwargs.pop('logY', None)
        xscale = kwargs.get('xscale', 'log')
        yscale = kwargs.get('yscale', 'linear')

        marker = cycle(markers)

        for j in range(self.nSystems):
            system_times = self.off_time(j)

            # kwargs['marker'] = markers[self._components[k]]
            kwargs['marker'] = next(marker)

            if (with_error_bars):
                ax.errorbar(system_times, self.data, yerr=self.std,
                                color=c[j],
                                markerfacecolor=mfc[j],
                                **kwargs)
            else:
                ax.plot(system_times, self.data,
                            markerfacecolor=mfc[j],
                            **kwargs)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(cf.getNameUnits(self.data))
        ax.set_title(title)

        if self.nSystems > 1:
            ax.legend()

        return ax

    def plot_posteriors(self, axes=None, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', gcf())

        if not isinstance(axes, dict):
            axes = self._init_posterior_plots(axes)

        required_axes = 1 + sum([x.hasPosterior for x in (self.relative_error, self.additive_error_multiplier, self.loop_pair)])
        assert len(axes) == required_axes, ValueError("Must have length {} list of axes for the posteriors. self.init_posterior_plots can generate them".format(required_axes))

        data_kwargs = kwargs.pop('data_kwargs', {})
        rel_error_kwargs = kwargs.pop('rel_error_kwargs', {})
        add_error_kwargs = kwargs.pop('add_error_kwargs', {})
        loop_pair_kwargs = kwargs.pop('loop_pair_kwargs', {})

        overlay = kwargs.get('overlay', None)
        if not overlay is None:
                rel_error_kwargs['overlay'] = overlay.relative_error
                add_error_kwargs['overlay'] = overlay.additive_error_multiplier
                loop_pair_kwargs['overlay'] = overlay.loop_pair

        ax = axes['data']; ax.cla()
        self.predicted_data.plot_posteriors(ax = ax, colorbar=False, **data_kwargs)
        self.plot(ax=ax, **data_kwargs)

        c = cp.wellSeparated[0] if overlay is None else cp.wellSeparated[3]
        self.plot_predicted(color=c, ax=ax, **data_kwargs)

        if self.relative_error.hasPosterior:
            self.relative_error.plot_posteriors(ax=axes['relative_error'], **rel_error_kwargs)

        if self.additive_error_multiplier.hasPosterior:
            add_error_kwargs['colorbar'] = False
            self.additive_error_multiplier.plot_posteriors(ax=axes['additive_error_multiplier'], **add_error_kwargs)

        if self.loop_pair.hasPosterior:
            self.loop_pair.plot_posteriors(axes = axes['loop_pair'], **loop_pair_kwargs)

    def plot_predicted(self, title='Tempest data', **kwargs):
        kwargs['xscale'] = kwargs.get('xscale', 'log')
        kwargs['yscale'] = kwargs.get('yscale', 'linear')

        ax = kwargs.get('ax', None)
        ax = plt.gca() if ax is None else ax

        labels = kwargs.pop('labels', True)

        if (labels):
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(cf.getNameUnits(self.predicted_data))
            ax.set_title(title)

        kwargs['color'] = kwargs.pop('color', cp.wellSeparated[3])
        kwargs['linewidth'] = kwargs.pop('linewidth', 1)
        kwargs['alpha'] = kwargs.pop('alpha', 0.7)
        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'log')

        kwargs.pop('logX', None)
        kwargs.pop('logY', None)

        for j in range(self.nSystems):
            system_times = self.off_time(j)

            if npall(self.data <= 0.0):
                active = (self.predicted_data > 0.0)
            else:
                active = self.active

            p = self.predicted_data[active]
            p.plot(x=system_times[active], **kwargs)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

    def plot_secondary_field(self, title='Secondary field', **kwargs):

        # ax = kwargs.get('ax', None)
        # ax = gca() if ax is None else sca(ax)
        # cla()

        kwargs['marker'] = kwargs.pop('marker', 'v')
        kwargs['markersize'] = kwargs.pop('markersize', 7)
        c = kwargs.pop('color', [cp.wellSeparated[i+1] for i in range(self.nSystems)])
        mfc = kwargs.pop('markerfacecolor', [cp.wellSeparated[i+1] for i in range(self.nSystems)])
        assert len(c) == self.nSystems, ValueError("color must be a list of length {}".format(self.nSystems))
        assert len(mfc) == self.nSystems, ValueError("markerfacecolor must be a list of length {}".format(self.nSystems))
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
        # ax = kwargs.pop('ax', None)
        # ax = gca() if ax is None else sca(ax)

        noLabels = kwargs.pop('nolabels', False)

        # if (not noLabels):
        #     kwargs['xlabel'] = 'Time (s)'
        #     kwargs['ylabel'] = cf.getNameUnits(self.predicted_data)
        #     kwargs['title'] = title

        kwargs['color'] = kwargs.pop('color', cp.wellSeparated[3])
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)
        kwargs['alpha'] = kwargs.pop('alpha', 0.7)
        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'linear')

        logx = kwargs.pop('logX', None)
        logy = kwargs.pop('logY', None)

        # for i in range(self.nSystems):
        system_times, _ = cf._log(self.off_time(0), logx)
        for j in range(self.n_components):
            ic = self._component_indices(j, 0)
            self.predicted_secondary_field[ic].plot(x=system_times, **kwargs)

    def prior_derivative(self, order):

        J = self.sensitivity_matrix[self.active, :]

        if order == 1:
            return dot(J.T, self.predicted_data.priorDerivative(order=1, i=self.active))

        elif order == 2:
            WdT_Wd = self.predicted_data.priorDerivative(order=2)
            return dot(J.T, dot(WdT_Wd, J))


    # def set_priors(self, relative_error_prior=None, additive_error_prior=None, data_prior=None, **kwargs):

        # if additive_error_prior is None:
        #     if kwargs.get('solve_additive_error', False):
        #         additive_error_prior = Distribution('Uniform', kwargs['minimum_additive_error'], kwargs['maximum_additive_error'], log=10, prng=kwargs.get('prng'))

        # self.additive_error.prior = additive_error_prior

        # kwargs['solve_additive_error'] = False
        # super().set_priors(relative_error_prior, None, data_prior, **kwargs)

    # def set_relative_error_prior(self, prior):
    #     if not prior is None:
    #         assert prior.ndim == self.nSystems, ValueError("relative_error_prior must have {} dimensions".format(self.nSystems))
    #         self.relative_error.prior = prior

    # def set_additive_error_prior(self, prior):
    #     if not prior is None:
    #         assert prior.ndim == self.nChannels, ValueError("additive_error_prior must have {} dimensions".format(self.nChannels))
    #         self.additive_error.prior = prior

    # def set_proposals(self, relative_error_proposal=None, additive_error_proposal=None, **kwargs):
    #     super().set_proposals(relative_error_proposal, additive_error_proposal, **kwargs)

    def set_additive_error_proposal(self, proposal=None, **kwargs):
        if proposal is None:
            if kwargs.get('solve_additive_error', False):
                proposal = Distribution('MvLogNormal', self.additive_error_multiplier, kwargs['additive_error_multiplier_proposal_variance'], linearSpace=True, prng=kwargs.get('prng'))

        self.additive_error_multiplier.proposal = proposal

    def set_additive_error_posterior(self, log=None):
        """"""
        if self.additive_error.hasPrior:
            mesh = RectilinearMesh1D(edges = DataArray(logspace(-0.1, 0.1, 150), name=self.additive_error_multiplier.name), log=log)
            self.additive_error_multiplier.posterior = Histogram(mesh=mesh)

    # def update_additive_error_posterior(self):
    #     if self.additive_error.hasPosterior:
    #         i = 0
    #         # for j in range(self.nSystems):
    #         system_times = self.off_time(0)
    #         for k in range(self.n_components):
    #             icomp = self._component_indices(k, 0)
    #             self.additive_error.posterior[i].update(x=system_times, y=self.additive_error[icomp])
    #             i += 1

    def update_additive_error_posterior(self):
        self.additive_error_multiplier.update_posterior(active=self.active_system_indices)

    def _empymodForward(self, mod):
        print('stuff')

    def sensitivity(self, model, ix=None, model_changed=False):
        """ Compute the sensitivty matrix for the given model """

        assert isinstance(model, Model), TypeError("Invalid model class for sensitivity matrix [1D]")

        J = DataArray(tdem1dsen(self, model, ix, model_changed), 'Sensitivity', r'$\frac{V}{SAm^{3}}$')

        self._sensitivity_matrix = DataArray(zeros((self.n_data_channels, model.nCells.item())))
        dp = 1.0 / self.predicted_data
        for i in range(self.n_components):
            ic = self._component_indices(i, 0)
            # Compute Sum(Pc + Sc) for c in x, y, z
            # self._sensitivity_matrix += J[ic, :]

            dc = self.predicted_primary_field[i] + self.predicted_secondary_field[ic]
            self._sensitivity_matrix += dot(diag(dc * dp), J[ic, :])

        return self.sensitivity_matrix

    def fm_dlogc(self, model):

        assert isinstance(model, Model), TypeError("Invalid model class for sensitivity matrix [1D]")

        self._sensitivity_matrix = DataArray(zeros((self.n_data_channels, model.nCells.item())), 'Sensitivity', r'$\frac{V}{SAm^{3}}$')
        values, J = ga_fm_dlogc(self, model)
        dp = 1.0 / self.predicted_data
        for i in range(self.n_components):
            ic = self._component_indices(i, 0)
            # Compute Sum(Pc + Sc) for c in x, y, z
            # self._sensitivity_matrix += J[ic, :]

            dc = self.predicted_primary_field[i] + self.predicted_secondary_field[ic]
            self._sensitivity_matrix += dot(diag(dc * dp), J[ic, :])
            # self._sensitivity_matrix += (dp * dc * J[ic, :].T).T

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):

        grp = super().createHdf(parent, name, withPosterior, add_axis, fillvalue)

        self.additive_error_multiplier.createHdf(grp, 'additive_error_multiplier', add_axis=add_axis, fillvalue=fillvalue)

        if add_axis is not None:
            grp.attrs['repr'] = 'TempestData'

        return grp

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        super().writeHdf(parent, name, withPosterior, index)
        grp = parent[name]
        self.additive_error_multiplier.writeHdf(grp, 'additive_error_multiplier', index=index)

    @classmethod
    def fromHdf(cls, grp, **kwargs):
        """ Reads the object from a HDF group """
        self = super(Tempest_datapoint, cls).fromHdf(grp, **kwargs)
        self._additive_error_multiplier = StatArray.fromHdf(grp['additive_error_multiplier'], **kwargs)
        return self