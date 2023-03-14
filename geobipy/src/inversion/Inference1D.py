""" @Inference1D
Class to store inversion results. Contains plotting and writing to file procedures
"""
from copy import deepcopy
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure

from ..base import plotting as cP
from ..base import utilities as cF
import numpy as np
import h5py
from ..classes.core import StatArray
from ..classes.statistics.Distribution import Distribution
from ..classes.statistics.Histogram import Histogram
from ..classes.core.myObject import myObject
# from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
# from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.mesh.RectilinearMesh1D import RectilinearMesh1D
# from ..classes.model.Model import Model
from ..classes.core.Stopwatch import Stopwatch
from ..base.HDF import hdfRead
from cached_property import cached_property

class Inference1D(myObject):
    """Define the results for the Bayesian MCMC Inversion.

    Contains histograms and inversion related variables that can be updated as the Bayesian inversion progresses.

    Inference1D(saveMe, plotMe, savePNG, dataPoint, model, ID, \*\*kwargs)

    Parameters
    ----------
    saveMe : bool, optional
        Whether to save the results to HDF5 files.
    plotMe : bool, optional
        Whether to plot the results on the fly. Only use this in serial mode.
    savePNG : bool, optional
        Whether to save a png of each single data point results. Don't do this in parallel please.
    dataPoint : geobipy.dataPoint
        Datapoint to use in the inversion.
        The relative error prior must have been set with dataPoint.relative_error.set_prior()
        The additive error prior must have been set with dataPoint.additive_error.set_prior()
        The height prior must have been set with dataPoint.z.set_prior()
    model : geobipy.model
        Model representing the subsurface.
    ID : int, optional

    OtherParameters
    ---------------
    nMarkovChains : int, optional
        Number of markov chains that will be tested.
    plotEvery : int, optional
        When plotMe = True, update the plot when plotEvery iterations have progressed.
    parameterDisplayLimits : sequence of ints, optional
        Limits of the parameter axis in the hitmap plot.
    reciprocateParameters : bool, optional
        Take the reciprocal of the parameters when plotting the hitmap.
    reciprocateName : str, optional
        Name of the parameters if they are reciprocated.
    reciprocateUnits : str, optional
        Units of the parameters if they are reciprocated.

    """

    def __init__(self, datapoint, prng=None, world=None, **kwargs):
        """ Initialize the results of the inversion """

        self.fig = None
        self.kwargs = kwargs

        self.prng = prng
        self.rank = 1 if world is None else world.rank

        if not 'n_markov_chains' in kwargs:
            return

        self.n_markov_chains = self.kwargs['n_markov_chains']
        # Get the initial best fitting halfspace and set up
        # priors and posteriors using user parameters

        # ------------------------------------------------
        # Intialize the datapoint with the user parameters
        # ------------------------------------------------
        self.initialize_datapoint(datapoint)

        # # Initialize the calibration parameters
        # if (kwargs.solveCalibration):
        #     datapoint.calibration.set_prior('Normal',
        #                            np.reshape(kwargs.calMean, np.size(kwargs.calMean), order='F'),
        #                            np.reshape(kwargs.calVar, np.size(kwargs.calVar), order='F'), prng=prng)
        #     datapoint.calibration[:] = datapoint.calibration.prior.mean
        #     # Initialize the calibration proposal
        #     datapoint.calibration.setProposal('Normal', datapoint.calibration, np.reshape(kwargs.propCal, np.size(kwargs.propCal), order='F'), prng=prng)

        # ---------------------------------
        # Set the earth model properties
        # ---------------------------------
        self.initialize_model()

        # Compute the data misfit
        self.data_misfit = datapoint.dataMisfit()

        # # Calibrate the response if it is being solved for
        # if (self.kwargs.solveCalibration):
        #     self.datapoint.calibrate()

        # Evaluate the prior for the current model
        self.prior = self.model.probability(self.kwargs['solve_parameter'],
                                            self.kwargs['solve_gradient'])

        self.prior += self.datapoint.probability

        # Initialize the burned in state
        self.burned_in_iteration = self._n_markov_chains
        self.burned_in = True

        # Add the likelihood function to the prior
        self.likelihood = 1.0
        if not self.kwargs['ignore_likelihood']:
            self.likelihood = self.datapoint.likelihood(log=True)
            self.burned_in = False
            self.burned_in_iteration = np.int64(0)

        self.posterior = self.likelihood + self.prior

        # Initialize the current iteration number
        # Current iteration number
        self.iteration = np.int64(0)

        # Initialize the vectors to save results
        # StatArray of the data misfit

        self.data_misfit_v = StatArray.StatArray(2 * self._n_markov_chains, name='Data Misfit')
        self.data_misfit_v[0] = self.data_misfit

        target = np.sum(self.datapoint.active)

        self.data_misfit_v.prior = Distribution('chi2', df=target)

        self.relative_chi_squared_fit = 100.0

        edges = StatArray.StatArray(np.linspace(1, 2*target))
        self.data_misfit_v.posterior = Histogram(mesh = RectilinearMesh1D(edges=edges))

        # Initialize a stopwatch to keep track of time
        self.clk = Stopwatch()
        self.invTime = np.float64(0.0)

        # Logicals of whether to plot or save
        self.save_hdf5 = self.kwargs['save_hdf5']  # pop('save', True)
        self.interactive_plot = self.kwargs.get('interactive_plot', False)
        self.save_png = self.kwargs['save_png']  # .pop('savePNG', False)

        # Return none if important parameters are not used (used for hdf 5)
        if datapoint is None:
            return

        assert self.interactive_plot or self.save_hdf5, Exception(
            'You have chosen to neither view or save the inversion results!')

        self._update_plot_every = self.kwargs['update_plot_every']
        self.limits = self.kwargs['parameter_limits']
        self.reciprocateParameter = self.kwargs['reciprocate_parameters']

        # Set the ID for the data point the results pertain to

        # Set the increment at which to plot results
        # Increment at which to update the results

        # Set the display limits of the parameter in the HitMap
        # Display limits for parameters
        # Should we plot resistivity or Conductivity?
        # Logical whether to take the reciprocal of the parameters

        # Multiplier for discrepancy principle
        self.multiplier = np.float64(1.0)

        # Initialize the acceptance level
        # Model acceptance rate
        self.accepted = 0

        n = 2 * np.int32(self.n_markov_chains / self._update_plot_every)
        self.acceptance_x = StatArray.StatArray(np.arange(1, n + 1) * self._update_plot_every, name='Iteration #')
        self.acceptance_rate = StatArray.StatArray(np.full(n, fill_value=np.nan), name='% Acceptance')


        self.iRange = StatArray.StatArray(np.arange(2 * self.n_markov_chains), name="Iteration #", dtype=np.int64)

        # Initialize the index for the best model
        # self.iBestV = StatArray.StatArray(2*self.n_markov_chains, name='Iteration of best model')

        # Initialize the doi
        # self.doi = model.par.posterior.yBinCentres[0]

        # self.meanInterp = StatArray.StatArray(model.par.posterior.y.nCells.value)
        # self.bestInterp = StatArray.StatArray(model.par.posterior.y.nCells.value)
        # self.opacityInterp = StatArray.StatArray(model.par.posterior.y.nCells.value)

        # Initialize time in seconds
        self.inference_time = np.float64(0.0)

        # Initialize the best data, current data and best model
        self.best_model = deepcopy(self.model)
        self.best_datapoint = deepcopy(self.datapoint)
        self.best_posterior = self.posterior
        self.best_iteration = np.int64(0)

    # @cached_property
    # def iteration(self):
    #     return StatArray.StatArray(np.arange(2 * self._n_markov_chains), name="Iteration #", dtype=np.int64)

    @cached_property
    def iz(self):
        return np.arange(self.model.values.posterior.y.nCells.item())

    @property
    def n_markov_chains(self):
        return self._n_markov_chains

    @n_markov_chains.setter
    def n_markov_chains(self, value):
        self._n_markov_chains = np.int64(value)

    @property
    def prng(self):
        return self._prng

    @prng.setter
    def prng(self, value):
        if value is None:
            self._prng = np.random.RandomState()
        else:
            self._prng = value

        self.seed = self.prng.get_state()
        self.kwargs['prng'] = self.prng

    @property
    def update_plot_every(self):
        return self._update_plot_every

    @property
    def user_options(self):
        return self._user_options

    def initialize_datapoint(self, datapoint):

        self.datapoint = datapoint

        # _ = self.datapoint.find_best_halfspace()

        # ---------------------------------------
        # Set the statistical properties of the datapoint
        # ---------------------------------------
        # Set the prior on the data
        self.datapoint.initialize(**self.kwargs)
        # Set the priors, proposals, and posteriors.
        self.datapoint.set_priors(**self.kwargs)
        self.datapoint.set_proposals(**self.kwargs)
        self.datapoint.set_posteriors()

    def initialize_model(self):
        # Find the conductivity of a half space model that best fits the data
        halfspace = self.datapoint.find_best_halfspace()
        self.halfspace = StatArray.StatArray(halfspace.values, 'halfspace')

        # Create an initial model for the first iteration
        # Initialize a 1D model with the half space conductivity
        # Assign the depth to the interface as half the bounds
        # self.model = halfspace.insert_edge(0.5 * (self.kwargs['maximum_depth'] + self.kwargs['minimum_depth']))
        self.model = deepcopy(halfspace)

        # Setup the model for perturbation
        self.model.set_priors(
            value_mean=halfspace.values.item(),
            min_edge=self.kwargs['minimum_depth'],
            max_edge=self.kwargs['maximum_depth'],
            max_cells=self.kwargs['maximum_number_of_layers'],
            solve_value=True, #self.kwargs['solve_parameter'],
            solve_gradient=self.kwargs['solve_gradient'],
            parameterLimits=self.kwargs.get('parameter_limits', None),
            min_width=self.kwargs.get('minimum_thickness', None),
            factor=self.kwargs.get('factor', 10.0), prng=self.prng
        )

        # Assign a Hitmap as a prior if one is given
        # if (not self.kwargs.referenceHitmap is None):
        #     Mod.setReferenceHitmap(self.kwargs.referenceHitmap)

        # Compute the predicted data
        self.datapoint.forward(self.model)

        observation = self.datapoint
        if self.kwargs['ignore_likelihood']:
            observation = None
        else:
            observation.sensitivity(self.model)

        local_variance = self.model.local_variance(observation)

        # Instantiate the proposal for the parameters.
        parameterProposal = Distribution('MvLogNormal', mean=self.model.values, variance=local_variance, linearSpace=True, prng=self.prng)

        probabilities = [self.kwargs['probability_of_birth'],
                         self.kwargs['probability_of_death'],
                         self.kwargs['probability_of_perturb'],
                         self.kwargs['probability_of_no_change']]
        self.model.set_proposals(probabilities=probabilities, proposal=parameterProposal, prng=self.prng)

        self.model.set_posteriors()

    def accept_reject(self):
        """ Propose a new random model and accept or reject it """
        perturbed_datapoint = deepcopy(self.datapoint)

        # Perturb the current model
        observation = perturbed_datapoint
        if self.kwargs.get('ignore_likelihood', False):
            observation = None

        remapped_model, perturbed_model = self.model.perturb(observation)

        # Propose a new data point, using assigned proposal distributions
        perturbed_datapoint.perturb()

        # Forward model the data from the candidate model
        perturbed_datapoint.forward(perturbed_model)

        # Compute the data misfit
        data_misfit1 = perturbed_datapoint.dataMisfit()

        # Evaluate the prior for the current data
        prior1 = perturbed_datapoint.probability
        # Test for early rejection
        if (prior1 == -np.inf):
            return

        # Evaluate the prior for the current model
        prior1 += perturbed_model.probability(self.kwargs['solve_parameter'], self.kwargs['solve_gradient'])

        # Test for early rejection
        if (prior1 == -np.inf):
            return

        # Compute the components of each acceptance ratio
        likelihood1 = 1.0
        observation = None
        if not  self.kwargs.get('ignore_likelihood', False):
            likelihood1 = perturbed_datapoint.likelihood(log=True)
            observation = deepcopy(perturbed_datapoint)

        proposal, proposal1 = perturbed_model.proposal_probabilities(remapped_model, observation)

        posterior1 = prior1 + likelihood1

        prior_ratio = prior1 - self.prior

        likelihood_ratio = likelihood1 - self.likelihood

        proposal_ratio = proposal - proposal1

        try:
            log_acceptance_ratio = np.float128(prior_ratio + likelihood_ratio + proposal_ratio)
            acceptance_probability = cF.expReal(log_acceptance_ratio)
        except:
            log_acceptance_ratio = -np.inf
            acceptance_probability = -1.0

        # If we accept the model
        accepted = acceptance_probability > self.prng.uniform()

        if (accepted):
            self.accepted += 1
            self.data_misfit = data_misfit1
            self.prior = prior1
            self.likelihood = likelihood1
            self.posterior = posterior1
            self.model = perturbed_model
            self.datapoint = perturbed_datapoint
            # Reset the sensitivity locally to the newly accepted model
            self.datapoint.sensitivity(self.model, modelChanged=False)

    def infer(self, hdf_file_handle):
        """ Markov Chain Monte Carlo approach for inversion of geophysical data
        userParameters: User input parameters object
        DataPoint: Datapoint to invert
        ID: Datapoint label for saving results
        pHDFfile: Optional HDF5 file opened using h5py.File('name.h5','w',driver='mpio', comm=world) before calling Inv_MCMC
        """

        if self.interactive_plot:
            self._init_posterior_plots()
            plt.show(block=False)

        self.clk.start()

        Go = True
        failed = False
        while (Go):
            # Accept or reject the new model
            self.accept_reject()

            self.update()

            if self.interactive_plot:
                self.plot_posteriors(axes=self.ax,
                                     fig=self.fig,
                                     title="Fiducial {}".format(self.datapoint.fiducial),
                                     increment=self.kwargs['update_plot_every'])

            Go = self.iteration <= self.n_markov_chains + self.burned_in_iteration

            if not self.burned_in:
                Go = self.iteration < self.n_markov_chains
                if not Go:
                    failed = True



        self.clk.stop()
        # self.invTime = np.float64(self.clk.timeinSeconds())
        # Does the user want to save the HDF5 results?
        if (self.kwargs['save_hdf5']):
            # No parallel write is being used, so write a single file for the data point
            self.write_inference1d(hdf_file_handle)

        # Does the user want to save the plot as a png?
        if (self.kwargs['save_png']):# and not failed):
            # To save any thing the Results must be plot
            self.plot_posteriors(axes = self.ax, fig=self.fig)
            self.toPNG('.', self.datapoint.fiducial)

        return failed

    def __deepcopy__(self, memo={}):
        return None

    @property
    def hitmap(self):
        return self.model.values.posterior

    def update(self):
        """Update the posteriors of the McMC algorithm. """

        self.iteration += 1

        self.data_misfit_v[self.iteration - 1] = self.data_misfit

        # Determine if we are burning in
        if (not self.burned_in):
            target_misfit = np.sum(self.datapoint.active)

            # if self.data_misfit < target_misfit:
            # if (self.iteration > 1000) and (np.isclose(self.data_misfit, self.multiplier*target_misfit, rtol=1e-1, atol=1e-2)):
            if (self.iteration > 1000) and (self.relative_chi_squared_fit < 1.0):
                self.burned_in = True  # Let the results know they are burned in
                self.burned_in_iteration = self.iteration       # Save the burn in iteration to the results
                self.best_iteration = self.iteration
                self.best_model = deepcopy(self.model)
                self.best_datapoint = deepcopy(self.datapoint)
                self.best_posterior = self.posterior

                self.data_misfit_v.reset_posteriors()
                self.model.reset_posteriors()
                self.datapoint.reset_posteriors()

        if (self.posterior > self.best_posterior):
            self.best_iteration = self.iteration
            self.best_model = deepcopy(self.model)
            self.best_datapoint = deepcopy(self.datapoint)
            self.best_posterior = self.posterior

        if (np.mod(self.iteration, self.update_plot_every) == 0):
            time_per_model = self.clk.lap() / self.update_plot_every
            tmp = "i=%i, k=%i, %4.3f s/Model, %0.3f s Elapsed\n" % (self.iteration, np.float64(self.model.nCells[0]), time_per_model, self.clk.timeinSeconds())
            if (self.rank == 1):
                print(tmp, flush=True)

            if (not self.burned_in and not self.datapoint.relative_error.hasPrior):
                self.multiplier *= self.kwargs['multiplier']

        # Added the layer depths to a list, we histogram this list every
        # iPlot iterations
        self.model.update_posteriors(0.5)#self.user_options.clip_ratio)

        # Update the height posterior
        self.datapoint.update_posteriors()

        if ((self.iteration > 0) and (np.mod(self.iteration, self.update_plot_every) == 0)):
            acceptance_percent = 100.0 * np.float64(self.accepted) / np.float64(self.update_plot_every)
            self.acceptance_rate[np.int32(self.iteration / self.update_plot_every)-1] = acceptance_percent
            self.accepted = 0

    def _init_posterior_plots(self, gs=None, **kwargs):
        """ Initialize the plotting region """
        # Setup the figure region. The figure window is split into a 4x3
        # region. Columns are able to span multiple rows

        fig  = kwargs.get('fig', plt.gcf())
        if gs is None:
            fig = kwargs.pop('fig', plt.figure(facecolor='white', figsize=(10, 7)))
            gs = fig

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        gs = gs.subgridspec(2, 2, height_ratios=(1, 6))

        ax = [None] * 4

        ax[0] = cP.pretty(plt.subplot(gs[0, 0]))  # Acceptance Rate 0

        splt = gs[0, 1].subgridspec(1, 2, width_ratios=[4, 1])
        tmp = [plt.subplot(splt[0, 0])]
        tmp.append(plt.subplot(splt[0, 1]))#, sharey=ax[0]))
        ax[1] = tmp  # Data misfit vs iteration 1 and posterior

        ax[2] = self.model._init_posterior_plots(gs[1, 0])
        ax[3] = self.datapoint._init_posterior_plots(gs[1, 1])

        if self.interactive_plot:
            plt.show(block=False)
            plt.interactive(True)

        self.fig, self.ax = fig, ax

        return fig, ax

    def plot_posteriors(self, axes=None, title="", increment=None, **kwargs):
        """ Updates the figures for MCMC Inversion """
        # Plots that change with every iteration
        if self.iteration == 0:
            return

        if axes is None:
            fig = kwargs.pop('fig', None)
            axes = fig
            if fig is None:
                fig, axes = self._init_posterior_plots()

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes)

        plot = True
        if increment is not None:
            if (np.mod(self.iteration, increment) != 0):
                plot = False

        if plot:
            self._plotAcceptanceVsIteration()

            # Update the data misfit vs iteration
            self._plotMisfitVsIteration()

            overlay = self.best_model if self.burned_in else self.model

            self.model.plot_posteriors(
                axes=self.ax[2],
                # ncells_kwargs={
                #     'normalize': True},
                edges_kwargs={
                    'transpose': True,
                    'trim': False},
                values_kwargs={
                    'colorbar': False,
                    'flipY': True,
                    'xscale': 'log',
                    'credible_interval_kwargs': {
                        # 'axis': 1
                    }
                },
                overlay=overlay)

            overlay = self.best_datapoint if self.burned_in else self.datapoint

            self.datapoint.plot_posteriors(
                axes=self.ax[3],
                # height_kwargs={
                #     'normalize': True},
                data_kwargs={},
                # rel_error_kwargs={
                #     'normalize': True},
                # add_error_kwargs={
                #     'normalize': True},
                overlay=overlay)

            cP.suptitle(title)

            # self.fig.tight_layout()
            if self.fig is not None:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

            cP.pause(1e-9)

    def _plotAcceptanceVsIteration(self, **kwargs):
        """ Plots the acceptance percentage against iteration. """

        i = np.s_[:np.int64(self.iteration / self.update_plot_every)]

        acceptance_rate = self.acceptance_rate[i]
        i_positive = np.argwhere(acceptance_rate > 0.0)
        i_zero = np.argwhere(acceptance_rate == 0.0)

        kwargs['ax'] = kwargs.get('ax', self.ax[0])
        kwargs['marker'] = kwargs.get('marker', 'o')
        kwargs['alpha'] = kwargs.get('alpha', 0.7)
        kwargs['linestyle'] = kwargs.get('linestyle', 'none')
        kwargs['markeredgecolor'] = kwargs.get('markeredgecolor', 'k')

        self.acceptance_rate[i_positive].plot(x=self.acceptance_x[i_positive], color='k', **kwargs)
        self.acceptance_rate[i_zero].plot(x=self.acceptance_x[i_zero], color='r', **kwargs)

        self.ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    def _plotMisfitVsIteration(self, **kwargs):
        """ Plot the data misfit against iteration. """

        ax = kwargs.get('ax', self.ax[1])
        m = kwargs.pop('marker', '.')
        # ms = kwargs.pop('markersize', 1)
        a = kwargs.pop('alpha', 0.7)
        ls = kwargs.pop('linestyle', 'none')
        c = kwargs.pop('color', 'k')
        # lw = kwargs.pop('linewidth', 1)

        kwargs['ax'] = ax[0]

        tmp_ax = self.data_misfit_v.plot(self.iRange, i=np.s_[:self.iteration], marker=m, alpha=a, linestyle=ls, color=c, **kwargs)
        plt.ylabel('Data Misfit')

        dum = self.multiplier * self.data_misfit_v.prior.df
        plt.axhline(dum, color='#C92641', linestyle='dashed')
        if (self.burned_in):
            plt.axvline(self.burned_in_iteration, color='#C92641',
                        linestyle='dashed')
            # plt.axvline(self.best_iteration, color=cP.wellSeparated[3])
        plt.yscale('log')
        tmp_ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        plt.xlim([0, self.iRange[self.iteration]])

        if not self.burned_in:
            self.data_misfit_v.reset_posteriors()

        self.data_misfit_v.posterior.update(self.data_misfit_v[np.maximum(0, self.iteration-self.update_plot_every):self.iteration], trim=True)

        kwargs = {'ax' : ax[1],
                  'normalize' : True}
        kwargs['ax'].cla()
        tmp_ax = self.data_misfit_v.posterior.plot(transpose=True, **kwargs)
        ylim = tmp_ax.get_ylim()
        tmp_ax = self.data_misfit_v.prior.plot_pdf(ax=kwargs['ax'], transpose=True, c='#C92641', linestyle='dashed')

        centres = self.data_misfit_v.posterior.mesh.centres
        h_pdf = self.data_misfit_v.posterior.pdf.values
        pdf = self.data_misfit_v.prior.probability(self.data_misfit_v.posterior.mesh.centres, log=False)

        self.relative_chi_squared_fit = np.linalg.norm(h_pdf - pdf)/np.linalg.norm(pdf)

        plt.hlines(np.sum(self.datapoint.active), xmin=0.0, xmax=0.5*tmp_ax.get_xlim()[1], color='#C92641', linestyle='dashed')
        tmp_ax.set_ylim(ylim)


    # def _plotObservedPredictedData(self, **kwargs):
    #     """ Plot the observed and predicted data """
    #     if self.burnedIn:
    #         # self.datapoint.predictedData.plot_posteriors(colorbar=False)
    #         self.datapoint.plot(**kwargs)
    #         self.bestDataPoint.plot_predicted(color=cP.wellSeparated[3], **kwargs)
    #     else:

    #         self.datapoint.plot(**kwargs)
    #         self.datapoint.plot_predicted(color='g', **kwargs)

    def saveToLines(self, h5obj):
        """ Save the results to a HDF5 object for a line """
        self.clk.restart()
        self.toHdf(h5obj, str(self.datapoint.fiducial))

    def save(self, outdir, fiducial):
        """ Save the results to their own HDF5 file """
        with h5py.File(join(outdir, str(fiducial)+'.h5'), 'w') as f:
            self.toHdf(f, str(fiducial))

    def toPNG(self, directory, fiducial, dpi=300):
       """ save a png of the results """
       self.fig.set_size_inches(19, 11)
       figName = join(directory, '{}.png'.format(fiducial))
       self.fig.savefig(figName, dpi=dpi)

    def read(self, fileName, system_file_path, fiducial=None, index=None):
        """ Reads a data point's results from HDF5 file """

        with h5py.File(fileName, 'r')as f:
            R = self.fromHdf(f, system_file_path, index=index, fiducial=fiducial)

        self.plotMe = True
        return self


    def createHdf(self, parent, fiducials):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """

        assert not np.any(np.isnan(fiducials)), ValueError("Cannot have fiducials == NaN")

        fiducials = StatArray.StatArray(np.sort(fiducials))
        nPoints = fiducials.size

        grp = self.datapoint.createHdf(parent, 'data', add_axis=nPoints, fillvalue=np.nan)
        fiducials.writeHdf(grp, 'fiducial')

        # Initialize and write the attributes that won't change
        # parent.create_dataset('ids',data=self.fiducials)
        # self.fiducials.toHdf(parent, 'fiducials')
        # self.fiducials.writeHdf(parent, 'fiducials')
        parent.create_dataset('iplot', data=self.update_plot_every)
        parent.create_dataset('plotme', data=self.interactive_plot)
        parent.create_dataset('reciprocateParameter', data=self.reciprocateParameter)

        if not self.limits is None:
            parent.create_dataset('limits', data=self.limits)
        parent.create_dataset('nmc', data=self.n_markov_chains)
        parent.create_dataset('nsystems', data=self.datapoint.nSystems)
        self.acceptance_x.toHdf(parent,'ratex')
#        parent.create_dataset('ratex', [self.ratex.size], dtype=self.ratex.dtype)
#        parent['ratex'][:] = self.ratex


        # Initialize the attributes that will be written later
        parent.create_dataset('i', shape=(nPoints), dtype=self.iteration.dtype, fillvalue=np.nan)
        parent.create_dataset('iburn', shape=(nPoints), dtype=self.burned_in_iteration.dtype, fillvalue=np.nan)
        parent.create_dataset('ibest', shape=(nPoints), dtype=self.best_iteration.dtype, fillvalue=np.nan)
        parent.create_dataset('burnedin', shape=(nPoints), dtype=type(self.burned_in))
        parent.create_dataset('multiplier',  shape=(nPoints), dtype=self.multiplier.dtype, fillvalue=np.nan)
        parent.create_dataset('invtime',  shape=(nPoints), dtype=float, fillvalue=np.nan)
        parent.create_dataset('savetime',  shape=(nPoints), dtype=float, fillvalue=np.nan)


        # self.meanInterp.createHdf(parent,'meaninterp', add_axis=nPoints, fillvalue=np.nan)
        # self.bestInterp.createHdf(parent,'bestinterp', add_axis=nPoints, fillvalue=np.nan)
        # self.opacityInterp.createHdf(parent,'opacityinterp',add_axis=nPoints, fillvalue=np.nan)
#        parent.create_dataset('opacityinterp', [nPoints,nz], dtype=np.float64)

        self.acceptance_rate.createHdf(parent,'rate', add_axis=nPoints, fillvalue=np.nan)
#        parent.create_dataset('rate', [nPoints,self.rate.size], dtype=self.rate.dtype)
        self.data_misfit_v.createHdf(parent, 'phids', add_axis=nPoints, fillvalue=np.nan)
        #parent.create_dataset('phids', [nPoints,self.PhiDs.size], dtype=self.PhiDs.dtype)
        self.halfspace.createHdf(parent, 'halfspace', add_axis=nPoints, fillvalue=np.nan)

        # Since the 1D models change size adaptively during the inversion, we need to pad the HDF creation to the maximum allowable number of layers.
        tmp = self.model.pad(self.model.mesh.max_cells)
        tmp.createHdf(parent, 'model', add_axis=nPoints, fillvalue=np.nan)

    def write_inference1d(self, parent, index=None):
        """ Given a HDF file initialized as line results, write the contents of results to the appropriate arrays """

        # assert self.datapoint.fiducial in self.fiducials, Exception("The HDF file does not have ID number {}. Available ids are between {} and {}".format(inference1d.fiducial, np.min(self.fiducials), np.max(self.fiducials)))

        hdfFile = parent

        # Get the point index
        if index is None:
            fiducials = StatArray.StatArray.fromHdf(parent['data/fiducial'])
            index = fiducials.searchsorted(self.datapoint.fiducial)

        i = index
        # Add the iteration number
        hdfFile['i'][i] = self.iteration

        # Add the burn in iteration
        hdfFile['iburn'][i] = self.burned_in_iteration

        # Add the burn in iteration
        hdfFile['ibest'][i] = self.best_iteration

        # Add the burned in logical
        hdfFile['burnedin'][i] = self.burned_in

        # Add the depth of investigation
        # hdfFile['doi'][i] = self.doi()

        # Add the multiplier
        hdfFile['multiplier'][i] = self.multiplier

        # Add the inversion time
        # hdfFile['invtime'][i] = self.invTime

        # Add the savetime
#        hdfFile['savetime'][i] = self.saveTime

        # Interpolate the mean and best model to the discretized hitmap
        # hm = self.model.par.posterior
        # self.meanInterp = StatArray.StatArray(hm.mean())
        # self.bestInterp = StatArray.StatArray(self.best_model.piecewise_constant_interpolate(self.best_model.par, hm, axis=0))
        # self.opacityInterp[:] = self.Hitmap.credibleRange(percent=95.0, log='e')

        # # Add the interpolated mean model
        # self.meanInterp.writeHdf(hdfFile, 'meaninterp',  index=i)
        # # Add the interpolated best
        # self.bestInterp.writeHdf(hdfFile, 'bestinterp',  index=i)
        # # Add the interpolated opacity

        # Add the acceptance rate
        self.acceptance_rate.writeHdf(hdfFile, 'rate', index=i)

        # Add the data misfit
        self.data_misfit_v.writeHdf(hdfFile, 'phids', index=i)

        # Write the data posteriors
        self.datapoint.writeHdf(hdfFile,'data',  index=i)
        # Write the highest posterior data
        self.best_datapoint.writeHdf(hdfFile,'data', withPosterior=False, index=i)

        self.halfspace.writeHdf(hdfFile, 'halfspace', index=i)

        # Write the model posteriors
        self.model.writeHdf(hdfFile,'model', index=i)
        # Write the highest posterior data
        self.best_model.writeHdf(hdfFile,'model', withPosterior=False, index=i)


    def read_fromH5Obj(self, h5obj, fName, grpName, system_file_path = ''):
        """ Reads a data points results from HDF5 file """
        grp = h5obj.get(grpName)
        assert not grp is None, "ID "+str(grpName) + " does not exist in file " + fName
        self.fromHdf(grp, system_file_path)


    @classmethod
    def fromHdf(cls, hdfFile, index=None, fiducial=None):

        iNone = index is None
        fNone = fiducial is None

        assert not (iNone and fNone) ^ (not iNone and not fNone), Exception("Must specify either an index OR a fiducial.")

        if not fNone:
            fiducials = StatArray.StatArray.fromHdf(hdfFile['data/fiducial'])
            index = fiducials.searchsorted(fiducial)

        self = cls(None, None)

        s = np.s_[index, :]

        self._n_markov_chains = np.array(hdfFile.get('nmc'))
        self._update_plot_every = np.array(hdfFile.get('iplot'))
        self.interactive_plot = np.array(hdfFile.get('plotme'))

        tmp = hdfFile.get('limits')
        self.limits = None if tmp is None else np.array(tmp)
        self.reciprocateParameter = np.array(hdfFile.get('reciprocateParameter'))

        self.nSystems = np.array(hdfFile.get('nsystems'))
        self.acceptance_x = hdfRead.readKeyFromFile(hdfFile, '', '/', 'ratex')

        self.iteration = hdfRead.readKeyFromFile(hdfFile, '', '/', 'i', index=index)
        self.burned_in_iteration = hdfRead.readKeyFromFile(hdfFile, '', '/', 'iburn', index=index)
        self.burned_in = hdfRead.readKeyFromFile(hdfFile, '', '/', 'burnedin', index=index)
        # self.doi = hdfRead.readKeyFromFile(hdfFile,'','/','doi', index=index)
        self.multiplier = hdfRead.readKeyFromFile(hdfFile, '', '/', 'multiplier', index=index)
        self.acceptance_rate = hdfRead.readKeyFromFile(hdfFile, '', '/', 'rate', index=s)
        # self.best_datapoint = hdfRead.readKeyFromFile(
        #     hdfFile, '', '/', 'bestd', index=index, system_file_path=system_file_path)

        self.datapoint = hdfRead.readKeyFromFile(hdfFile, '', '/', 'data', index=index)
        self.best_datapoint = self.datapoint

        self.data_misfit_v = hdfRead.readKeyFromFile(hdfFile, '', '/', 'phids', index=s)
        self.data_misfit_v.prior = Distribution('chi2', df=np.sum(self.datapoint.active))

        self.model = hdfRead.readKeyFromFile(hdfFile, '', '/', 'model', index=index)
        self.best_model = self.model

        self.halfspace = hdfRead.readKeyFromFile(hdfFile, '', '/', 'halfspace', index=index)

        # self.model.values.posterior.x.relativeTo = self.halfspace

        self.Hitmap = self.model.values.posterior
        # self.currentModel._max_edge = np.log(self.Hitmap.y.centres[-1])
        # except:
        #     self.Hitmap = hdfRead.readKeyFromFile(hdfFile,'','/','hitmap', index=index)

        # self.best_model = hdfRead.readKeyFromFile(
        #     hdfFile, '', '/', 'bestmodel', index=index)
        # self.bestModel._max_edge = np.log(self.Hitmap.y.centres[-1])

        self.invTime = np.array(hdfFile.get('invtime')[index])
        self.saveTime = np.array(hdfFile.get('savetime')[index])

        # Initialize a list of iteration number
        self.iRange = StatArray.StatArray(
            np.arange(2 * self.n_markov_chains), name="Iteration #", dtype=np.int64)

        self.verbose = False

        self.plotMe = True

        # self.fiducial = np.float64(fiducials[index])

        return self
