from copy import deepcopy

from ....classes.core import StatArray
from ...model.Model1D import Model1D
from .TdemDataPoint import TdemDataPoint
from ...forwardmodelling.Electromagnetic.TD.tdem1d import (tdem1dfwd, tdem1dsen)
from ...system.EmLoop import EmLoop
from ...system.SquareLoop import SquareLoop
from ...system.CircularLoop import CircularLoop
from ....base.logging import myLogger
from ...system.TdemSystem import TdemSystem
from ...system.filters.butterworth import butterworth
from ...system.Waveform import Waveform
from ...statistics.Histogram1D import Histogram1D
from ...statistics.Histogram2D import Histogram2D
from ...statistics.Distribution import Distribution
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#from ....base import Error as Err
from ....base import fileIO as fIO
from ....base import utilities as cf
from ....base import plotting as cP
from ....base import MPI as myMPI
from os.path import split as psplt
from os.path import join


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

    @TdemDataPoint.addErr.setter
    def addErr(self, values):
        if values is None:
            values = self.nChannels
        else:
            assert np.size(values) == self.nChannels, ValueError(("Tempest data must a have additive error values for all time gates and all components. \n"
                                                              "addErr must have size {}").format(self.nChannels))
            assert (np.all(np.asarray(values) > 0.0)), ValueError("addErr must be > 0.0.")

        self._addErr = StatArray.StatArray(values, '$\epsilon_{additive}x10^{2}$', self.units)

    # @property
    # def channels(self):
    #     return np.squeeze(np.asarray([np.tile(self.off_time(i), 2) for i in range(self.nSystems)]))

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

    @TdemDataPoint.relErr.setter
    def relErr(self, values):
        if values is None:
            values = self.n_components * self.nSystems
        else:
            assert np.size(values) == self.n_components * self.nSystems, ValueError(("Tempest data must a have relative error for the primary and secondary fields, for each system. \n"
                            "relErr must have size {}").format(self.n_components * self.nSystems))
            assert (np.all(np.asarray(values) > 0.0)), ValueError("relErr must be > 0.0.")

        self._relErr = StatArray.StatArray(values, '$\epsilon_{Relative}x10^{2}$', '%')

    @TdemDataPoint.std.getter
    def std(self):
        """ Compute the data errors. """

        # For each system assign error levels using the user inputs
        for i in range(self.nSystems):
            for j in range(self.n_components):
                ic = self._component_indices(j, i)
                relative_error = self.relErr[(i*2)+j] * self.data[ic]

                self._std[ic] = np.sqrt((relative_error**2.0) + (self.addErr[ic]**2.0))

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

        splt2 = splt[1, :].subgridspec(self.nSystems*self.n_components, 2, wspace=0.2)
        # Relative error axes
        ax.append([plt.subplot(splt2[i, 0]) for i in range(self.nSystems*self.n_components)])

        # # Additive Error axes
        # ax.append([None for i in range(self.nSystems*self.n_components)])

        # Pitch axes
        ax.append([plt.subplot(splt2[i, 1]) for i in range(self.nSystems)])

        return ax


    def set_priors(self, height_prior=None, data_prior=None, relative_error_prior=None, additive_error_prior=None, transmitter_pitch_prior=None, **kwargs):

        super().set_priors(height_prior, data_prior, relative_error_prior, additive_error_prior, **kwargs)

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
            self.relErr.prior = prior

    def set_proposals(self, height_proposal=None, relative_error_proposal=None, additive_error_proposal=None, transmitter_pitch_proposal=None, **kwargs):

        super().set_proposals(height_proposal, relative_error_proposal, additive_error_proposal, **kwargs)

        if transmitter_pitch_proposal is None:
            if kwargs.get('solve_transmitter_pitch', False):
                transmitter_pitch_proposal = Distribution('Normal', self.transmitter.pitch.value, kwargs['transmitter_pitch_proposal_variance'], prng=kwargs['prng'])

        self.transmitter.set_proposals(pitch_proposal=transmitter_pitch_proposal)

    def set_posteriors(self, log=None):

        super().set_posteriors(log=None)

        self.transmitter.set_posteriors()

    # def set_additive_error_prior(self, prior):
    #     if not prior is None:
    #         assert additive_error_prior.ndim == self.nChannels, ValueError("additive_error_prior must have {} dimensions".format(self.nChannels))
    #         self.addErr.set_prior(prior)

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

    def plot_posteriors(self, axes=None, height_kwargs={}, data_kwargs={}, rel_error_kwargs={}, pitch_kwargs={}, **kwargs):

        assert len(axes) == 4, ValueError("Must have length 3 list of axes for the posteriors. self.init_posterior_plots can generate them")

        best = kwargs.pop('best', None)
        if not best is None:
            height_kwargs['line'] = best.z
            rel_error_kwargs['line'] = best.relErr
            # add_error_kwargs['line'] = best.addErr
            pitch_kwargs['line'] = best.transmitter.pitch

        height_kwargs['rotate'] = height_kwargs.get('rotate', True)
        self.z.plotPosteriors(ax = axes[0], **height_kwargs)

        self.predictedData.plotPosteriors(ax = axes[1], noColorbar=True, **data_kwargs)
        self.plot(ax=axes[1], **data_kwargs)
        
        c = cP.wellSeparated[0] if best is None else cP.wellSeparated[3]
        self.plotPredicted(color=c, ax=axes[1], **data_kwargs)

        self.relErr.plotPosteriors(ax=axes[2], **rel_error_kwargs)
        # self.addErr.plotPosteriors(ax=axes[3], **add_error_kwargs)

        self.transmitter.pitch.plotPosteriors(ax = axes[3], **pitch_kwargs)

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

    def setPosteriors(self, log=None):
        return super().setPosteriors(log=None)

    def set_relative_error_posterior(self):
        if self.relErr.hasPrior:
            rb = StatArray.StatArray(np.atleast_2d(self.relErr.prior.bins()), name=self.relErr.name, units=self.relErr.units)
            self.relErr.posterior = [Histogram1D(edges = rb[i, :]) for i in range(self.nSystems*self.n_components)]

    def set_additive_error_posterior(self, log=None):
        if self.addErr.hasPrior:
            ab = StatArray.StatArray(np.atleast_2d(self.addErr.prior.bins()), name=self.addErr.name, units=self.data.units)
            self.addErr.posterior = [Histogram1D(edges = ab[i, :], log=log) for i in range(self.nSystems)]

    # def set_priors(self, height_prior=None, data_prior=None, relative_error_prior=None, additive_error_prior=None):

    #     super().set_priors(height_prior, None, relative_error_prior, additive_error_prior)

    #     if not data_prior is None:
    #         self.predictedData.set_prior(data_prior)

    # def set_predicted_data_posterior(self):
    #     if self.predictedData.hasPrior:
    #         times = np.log10(self.times(0))

    #         xbuf = 0.05*(times[-1] - times[0])
    #         xbins = np.logspace(times[0]-xbuf, times[-1]+xbuf, 200)
    #         ybins = np.linspace(0.8*np.nanmin(self.data), 1.2*np.nanmax(self.data), 200)
    #         # rto = 0.5 * (ybins[0] + ybins[-1])
    #         # ybins -= rto

    #         H = Histogram2D(xEdges=xbins, xlog=10, yEdges=ybins)

    #         self.predictedData.setPosterior(H)

    #         # H = Histogram2D(xEdges = StatArray.StatArray(self.z.prior.bins(), name=self.z.name, units=self.z.units), relativeTo=self.z)
    #         # self.z.setPosterior(H)



    def updatePosteriors(self):

        super().updatePosteriors()

        # if self.predictedData.hasPosterior:
        #     for i in range(self.n_components):
        #         j = self._component_indices(i, 0)
        #         self.predictedData.posterior.update_line(x=self.channels[j], y=self.predictedData[j])

        if self.transmitter.pitch.hasPosterior:
            self.transmitter.pitch.updatePosterior()


    # def forward(self, mod):
    #     """ Forward model the data from the given model """

    #     assert isinstance(mod, Model1D), TypeError("Invalid model class for forward modeling [1D]")
    #     fm = tdem1dfwd(self, mod)

    #     self.predicted_primary_field[:] = np.r_[fm[0].PX, -fm[0].PZ]


    #     # for i in range(self.nSystems):
    #         # iSys = self._systemIndices(i)

    #     self.predicted_secondary_field[:] = np.hstack([fm[0].SX, -fm[0].SZ])  # Store the necessary component


    # def sensitivity(self, model, ix=None, modelChanged=True):
    #     """ Compute the sensitivty matrix for the given model """

    #     assert isinstance(model, Model1D), TypeError("Invalid model class for sensitivity matrix [1D]")
    #     return StatArray.StatArray(tdem1dsen(self, model, ix, modelChanged), 'Sensitivity', '$\\frac{V}{SAm^{3}}$')


    def _empymodForward(self, mod):

        print('stuff')

    # def _simPEGForward(self, mod):

    #     from SimPEG import Maps
    #     from simpegEM1D import (EM1DSurveyTD, EM1D, set_mesh_1d)

    #     mesh1D = set_mesh_1d(mod.depth)
    #     expmap = Maps.ExpMap(mesh1D)
    #     prob = EM1D(mesh1D, sigmaMap = expmap, chi = mod.chim)

    #     if (self.dualMoment()):

    #         print(self.system[0].loopRadius(), self.system[0].peakCurrent())

    #         simPEG_survey = EM1DSurveyTD(
    #             rx_location=np.array([0., 0., 0.]),
    #             src_location=np.array([0., 0., 0.]),
    #             topo=np.r_[0., 0., 0.],
    #             depth=-mod.depth,
    #             rx_type='dBzdt',
    #             wave_type='general',
    #             src_type='CircularLoop',
    #             a=self.system[0].loopRadius(),
    #             I=self.system[0].peakCurrent(),
    #             time=self.system[0].windows.centre,
    #             time_input_currents=self.system[0].waveform.transmitterTime,
    #             input_currents=self.system[0].waveform.transmitterCurrent,
    #             n_pulse=2,
    #             base_frequency=self.system[0].baseFrequency(),
    #             use_lowpass_filter=True,
    #             high_cut_frequency=450000,
    #             moment_type='dual',
    #             time_dual_moment=self.system[1].windows.centre,
    #             time_input_currents_dual_moment=self.system[1].waveform.transmitterTime,
    #             input_currents_dual_moment=self.system[1].waveform.transmitterCurrent,
    #             base_frequency_dual_moment=self.system[1].baseFrequency(),
    #         )
    #     else:

    #         simPEG_survey = EM1DSurveyTD(
    #             rx_location=np.array([0., 0., 0.]),
    #             src_location=np.array([0., 0., 0.]),
    #             topo=np.r_[0., 0., 0.],
    #             depth=-mod.depth,
    #             rx_type='dBzdt',
    #             wave_type='general',
    #             src_type='CircularLoop',
    #             a=self.system[0].loopRadius(),
    #             I=self.system[0].peakCurrent(),
    #             time=self.system[0].windows.centre,
    #             time_input_currents=self.system[0].waveform.transmitterTime,
    #             input_currents=self.system[0].waveform.transmitterCurrent,
    #             n_pulse=1,
    #             base_frequency=self.system[0].baseFrequency(),
    #             use_lowpass_filter=True,
    #             high_cut_frequency=7e4,
    #             moment_type='single',
    #         )

    #     prob.pair(simPEG_survey)

    #     self._predictedData[:] = -simPEG_survey.dpred(mod.par)

    # def Isend(self, dest, world, **kwargs):

    #     super().Isend(dest, world, **kwargs)

    #     self.primary_field.Isend(dest, world)
    #     self.secondary_field.Isend(dest, world)
    #     self.predicted_primary_field.Isend(dest, world)
    #     self.predicted_secondary_field.Isend(dest, world)


    # @classmethod
    # def Irecv(cls, source, world, **kwargs):

    #     out = super(Tempest_datapoint, cls).Irecv(source, world, **kwargs)

    #     out._primary_field = StatArray.StatArray.Irecv(source, world)
    #     out._secondary_field = StatArray.StatArray.Irecv(source, world)
    #     out._predicted_primary_field = StatArray.StatArray.Irecv(source, world)
    #     out._predicted_secondary_field = StatArray.StatArray.Irecv(source, world)

    #     return out