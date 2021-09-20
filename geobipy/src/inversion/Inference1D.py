""" @Inference1D
Class to store inversion results. Contains plotting and writing to file procedures
"""
from copy import deepcopy
from os.path import join
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import pause
from matplotlib.ticker import MaxNLocator
from ..base import plotting as cP
from ..base import utilities as cF
import numpy as np
from ..base import fileIO as fIO
import h5py
from ..base.HDF.hdfWrite import write_nd
from ..classes.core import StatArray
from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.statistics.Histogram1D import Histogram1D
from ..classes.statistics.Distribution import Distribution
from ..classes.core.myObject import myObject
from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.model.Model1D import Model1D
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
        The relative error prior must have been set with dataPoint.relErr.set_prior()
        The additive error prior must have been set with dataPoint.addErr.set_prior()
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

    def __init__(self, datapoint, prng, kwargs, world=None):
        """ Initialize the results of the inversion """

        self.fig = None
        self._user_options = kwargs
        self.prng = prng
        self.rank = 1 if world is None else world.rank

        if kwargs is None:
            return


        self._n_markov_chains = np.int64(kwargs.nMarkovChains) #np.int64(kwargs.pop('nMarkovChains', 100000))
        # Get the initial best fitting halfspace and set up
        # priors and posteriors using user parameters

        # ------------------------------------------------
        # Intialize the datapoint with the user parameters
        # ------------------------------------------------
        self.initialize_datapoint(datapoint, kwargs)

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
        self.initialize_model(kwargs)

        # Compute the data misfit
        self.data_misfit = datapoint.dataMisfit(squared=True)

        # # Calibrate the response if it is being solved for
        # if (kwargs.solveCalibration):
        #     self.datapoint.calibrate()

        # Evaluate the prior for the current model
        self.prior = self.model.priorProbability(
            kwargs.solveParameter,
            kwargs.solveGradient) + \
            self.datapoint.priorProbability(
            kwargs.solveRelativeError,
            kwargs.solveAdditiveError,
            kwargs.solveHeight,
            kwargs.solveCalibration)

        # Initialize the burned in state
        self.burned_in_iteration = self._n_markov_chains
        self.burned_in = True

        # Add the likelihood function to the prior
        self.likelihood = 1.0
        if not kwargs.ignoreLikelihood:
            self.likelihood = self.datapoint.likelihood(log=True)
            self.burned_in = False
            self.burned_in_iteration = np.int64(0)

        self.posterior = self.likelihood + self.prior

        # Initialize the current iteration number
        # Current iteration number
        self.iteration = np.int64(0)

        # Initialize the vectors to save results
        # StatArray of the data misfit

        self.data_misfit_v = StatArray.StatArray(
            2 * self._n_markov_chains, name='Data Misfit')
        self.data_misfit_v[0] = self.data_misfit

        # Initialize a stopwatch to keep track of time
        self.clk = Stopwatch()
        self.invTime = np.float64(0.0)

        # Logicals of whether to plot or save
        self.save_hdf5 = kwargs.save  # pop('save', True)
        self.interactive_plot = kwargs.plot  # .pop('plot', False)
        self.save_png = kwargs.savePNG  # .pop('savePNG', False)

        # Return none if important parameters are not used (used for hdf 5)
        if datapoint is None:
            return

        assert self.interactive_plot or self.save_hdf5, Exception(
            'You have chosen to neither view or save the inversion results!')

        # np.int64(kwargs.pop('plotEvery', nMarkovChains / 20))
        self._update_plot_every = np.int64(kwargs.plotEvery)
        # np.asarray(kwargs.pop('parameterDisplayLimits', [0.0, 1.0]))
        self.limits = kwargs.parameterLimits
        # kwargs.pop('reciprocateParameters', False)
        self.reciprocateParameter = kwargs.reciprocateParameters

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

        n = 2 * np.int(self.n_markov_chains / 1000)
        self.acceptance_x = StatArray.StatArray(np.arange(1, n + 1) * 1000, name='Iteration #')
        self.acceptance_rate = StatArray.StatArray(n, name='% Acceptance')


        self.iRange = StatArray.StatArray(
            np.arange(2 * self.n_markov_chains), name="Iteration #", dtype=np.int64)

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

    @property
    def n_markov_chains(self):
        return self._n_markov_chains

    @cached_property
    def iteration(self):
        return StatArray.StatArray(np.arange(2 * self._n_markov_chains), name="Iteration #", dtype=np.int64)

    @cached_property
    def iz(self):
        return np.arange(model.par.posterior.y.nCells.value)

    @property
    def update_plot_every(self):
        return self._update_plot_every

    @property
    def user_options(self):
        return self._user_options

    def initialize_datapoint(self, datapoint, kwargs):

        self.datapoint = datapoint
        # ---------------------------------------
        # Set the statistical properties of the datapoint
        # ---------------------------------------
        # Set the prior on the data
        self.datapoint.relErr = kwargs.initialRelativeError
        self.datapoint.addErr = kwargs.initialAdditiveError

        # Define prior, proposal, posterior for height
        heightPrior = None
        heightProposal = None
        if kwargs.solveHeight:
            z = np.float64(self.datapoint.z)
            dz = kwargs.maximumElevationChange
            heightPrior = Distribution('Uniform', z - dz, z + dz, prng=self.prng)
            heightProposal = Distribution('Normal', self.datapoint.z, kwargs.elevationProposalVariance, prng=self.prng)

        data_prior = Distribution('MvLogNormal', self.datapoint.data[self.datapoint.active], self.datapoint.std[self.datapoint.active]**2.0, linearSpace=False, prng=self.prng)

        # Define prior, proposal, posterior for relative error
        relativePrior = None
        relativeProposal = None
        if kwargs.solveRelativeError:
            relativePrior = Distribution('Uniform', kwargs.minimumRelativeError, kwargs.maximumRelativeError, prng=self.prng)
            relativeProposal = Distribution('MvNormal', self.datapoint.relErr, kwargs.relativeErrorProposalVariance, prng=self.prng)

        # Define prior, proposal, posterior for additive error
        additivePrior = None
        additiveProposal = None
        if kwargs.solveAdditiveError:
            log = isinstance(self.datapoint, TdemDataPoint)
            additivePrior = Distribution('Uniform', kwargs.minimumAdditiveError, kwargs.maximumAdditiveError, log=log, prng=self.prng)
            additiveProposal = Distribution('MvLogNormal', self.datapoint.addErr, kwargs.additiveErrorProposalVariance, linearSpace=log, prng=self.prng)

        # Set the priors, proposals, and posteriors.
        self.datapoint.set_priors(height_prior=heightPrior, data_prior=data_prior, relative_error_prior=relativePrior, additive_error_prior=additivePrior)
        self.datapoint.setProposals(heightProposal=heightProposal, relativeErrorProposal=relativeProposal, additiveErrorProposal=additiveProposal)
        self.datapoint.setPosteriors()

        # Update the data errors based on user given parameters
        # if kwargs.solveRelativeError or kwargs.solveAdditiveError:
        self.datapoint.updateErrors(kwargs.initialRelativeError, kwargs.initialAdditiveError)

    def initialize_model(self, kwargs):
        # Find the conductivity of a half space model that best fits the data
        halfspace = self.datapoint.find_best_halfspace()

        # Create an initial model for the first iteration
        # Initialize a 1D model with the half space conductivity
        # Assign the depth to the interface as half the bounds
        self.model = halfspace.insert_edge(0.5 * (kwargs.maximumDepth + kwargs.minimumDepth))

        # Setup the model for perturbation
        self.model.set_priors(
            halfspace.par[0],
            kwargs.minimumDepth,
            kwargs.maximumDepth,
            kwargs.maximumNumberofLayers,
            kwargs.solveParameter,
            kwargs.solveGradient,
            parameterLimits=kwargs.parameterLimits,
            min_width=kwargs.minimumThickness,
            factor=kwargs.factor, prng=self.prng
        )

        # Assign a Hitmap as a prior if one is given
        # if (not kwargs.referenceHitmap is None):
        #     Mod.setReferenceHitmap(kwargs.referenceHitmap)

        # Compute the predicted data
        self.datapoint.forward(self.model)

        if kwargs.ignoreLikelihood:
            inverseHessian = self.model.localParameterVariance()
        else:
            inverseHessian = self.model.localParameterVariance(self.datapoint)

        # Instantiate the proposal for the parameters.
        parameterProposal = Distribution('MvLogNormal', self.model.par, inverseHessian, linearSpace=True, prng=self.prng)

        probabilities = [kwargs.pBirth, kwargs.pDeath, kwargs.pPerturb, kwargs.pNochange]
        self.model.setProposals(probabilities, parameterProposal=parameterProposal, prng=self.prng)

        self.model.setPosteriors()

    def accept_reject(self):
        """ Propose a new random model and accept or reject it """
        perturbed_datapoint = deepcopy(self.datapoint)

        # Perturb the current model
        if self.user_options.ignoreLikelihood:
            remapped_model, perturbed_model = self.model.perturb()
        else:
            remapped_model, perturbed_model = self.model.perturb(perturbed_datapoint)

        # Propose a new data point, using assigned proposal distributions
        perturbed_datapoint.perturb(self.user_options.solveHeight, self.user_options.solveRelativeError, self.user_options.solveAdditiveError, self.user_options.solveCalibration)

        # Forward model the data from the candidate model
        perturbed_datapoint.forward(perturbed_model)

        # Compute the data misfit
        data_misfit1 = perturbed_datapoint.dataMisfit(squared=True)

        # Evaluate the prior for the current model
        prior1 = perturbed_model.priorProbability(self.user_options.solveParameter, self.user_options.solveGradient)
        # Evaluate the prior for the current data
        prior1 += perturbed_datapoint.priorProbability(self.user_options.solveRelativeError, self.user_options.solveAdditiveError, self.user_options.solveHeight, self.user_options.solveCalibration)

        # Test for early rejection
        if (prior1 == -np.inf):
            return

        # Compute the components of each acceptance ratio
        likelihood1 = 1.0
        if not self.user_options.ignoreLikelihood:
            likelihood1 = perturbed_datapoint.likelihood(log=True)
            proposal, proposal1 = perturbed_model.proposalProbabilities(remapped_model, perturbed_datapoint)
        else:
            proposal, proposal1 = perturbed_model.proposalProbabilities(remapped_model)


        posterior1 = prior1 + likelihood1

        prior_ratio = prior1 - self.prior

        likelihood_ratio = likelihood1 - self.likelihood

        proposal_ratio = proposal - proposal1

        # try:
        log_acceptance_ratio = np.float128(
            prior_ratio + likelihood_ratio + proposal_ratio)

        acceptance_probability = cF.expReal(log_acceptance_ratio)
        # except:
        #     log_acceptance_ratio = -np.inf
        #     acceptance_probability = -1.0

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

    def infer(self, hdf_file_handle):
        """ Markov Chain Monte Carlo approach for inversion of geophysical data
        userParameters: User input parameters object
        DataPoint: Datapoint to invert
        ID: Datapoint label for saving results
        pHDFfile: Optional HDF5 file opened using h5py.File('name.h5','w',driver='mpio', comm=world) before calling Inv_MCMC
        """

        if self.interactive_plot:
            self.initFigure()
            plt.show(block=False)

        self.clk.start()

        Go = True
        failed = False
        while (Go):

            # Accept or reject the new model
            self.accept_reject()

            self.update()
            #i, Mod, DataPoint, iBest, bestData, bestModel, multiplier, PhiD, posterior, posteriorComponents, ratioComponents, accepted, dimensionChange, userParameters.clipRatio)

            if self.interactive_plot:
                self.plot("Fiducial {}".format(self.datapoint.fiducial), increment=self.user_options.plotEvery)

            Go = self.iteration <= self.n_markov_chains + self.burned_in_iteration

            if not self.burned_in:
                Go = self.iteration < self.n_markov_chains
                if not Go:
                    failed = True

        self.clk.stop()
        # self.invTime = np.float64(self.clk.timeinSeconds())
        # Does the user want to save the HDF5 results?
        if (self.user_options.save):
            # No parallel write is being used, so write a single file for the data point
            self.write_inference1d(hdf_file_handle)

        # Does the user want to save the plot as a png?
        if (self.user_options.savePNG):# and not failed):
            # To save any thing the Results must be plot
            self.plot()
            self.toPNG('.', self.datapoint.fiducial)

        return failed

    def __deepcopy__(self, memo={}):
        return None

    @property
    def hitmap(self):
        return self.model.par.posterior

    def update(self):
        """Update the posteriors of the McMC algorithm. """

        self.data_misfit_v[self.iteration] = self.data_misfit
        # Determine if we are burning in
        if (not self.burned_in):
            target_misfit = np.sum(self.datapoint.active)
            if (self.data_misfit <= self.multiplier * target_misfit):  # datapoint.target_misfit
                self.burned_in = True  # Let the results know they are burned in
                self.burned_in_iteration = self.iteration       # Save the burn in iteration to the results
                self.best_iteration = self.iteration
                self.best_model = deepcopy(self.model)
                self.best_data = deepcopy(self.datapoint)
                self.best_posterior = self.posterior

        if (self.posterior > self.best_posterior):
            self.best_iteration = self.iteration
            self.best_model = deepcopy(self.model)
            self.best_data = deepcopy(self.datapoint)
            self.best_posterior = self.posterior

        if (np.mod(self.iteration, self.update_plot_every) == 0):
            time_per_model = self.clk.lap() / self.update_plot_every
            tmp = "i=%i, k=%i, %4.3f s/Model, %0.3f s Elapsed\n" % (self.iteration, np.float(self.model.nCells[0]), time_per_model, self.clk.timeinSeconds())
            if (self.rank == 1):
                print(tmp, flush=True)

            if (not self.burned_in and not self.user_options.solveRelativeError):
                self.multiplier *= self.user_options.multiplier

        if (self.burned_in):  # We need to update some plotting options
            # Added the layer depths to a list, we histogram this list every
            # iPlot iterations
            self.model.updatePosteriors(self.user_options.clipRatio)

            # Update the height posterior
            self.datapoint.updatePosteriors()

        if (np.mod(self.iteration, 1000) == 0):
            ratePercent = 0.1 * np.float64(self.accepted)

            self.acceptance_rate[np.int32(self.iteration / 1000)] = ratePercent
            self.accepted = 0

        self.iteration += 1

    def initFigure(self, fig=None):
        """ Initialize the plotting region """
        # Setup the figure region. The figure window is split into a 4x3
        # region. Columns are able to span multiple rows

        # plt.ion()

        if fig is None:
            self.fig = plt.figure(facecolor='white', figsize=(10, 7))
        else:
            self.fig = plt.figure(fig.number)

        mngr = plt.get_current_fig_manager()
        try:
            mng.frame.Maximize(True)
        except:
            try:
                mngr.window.showMaximized()
            except:
                try:
                    mngr.window.state('zoomed')
                except:
                    pass

        gs = self.fig.add_gridspec(2, 2, height_ratios=(1, 6))
        self.ax = [None] * 4

        self.ax[0] = plt.subplot(gs[0, 0])  # Acceptance Rate 0
        self.ax[1] = plt.subplot(gs[0, 1])  # Data misfit vs iteration 1
        for ax in self.ax[:2]:
            cP.pretty(ax)

        self.ax[2] = self.model.init_posterior_plots(gs[1, 0])
        self.ax[3] = self.datapoint.init_posterior_plots(gs[1, 1])

        if self.interactive_plot:
            plt.show(block=False)
        # plt.draw()

    def plot(self, title="", increment=None):
        """ Updates the figures for MCMC Inversion """
        # Plots that change with every iteration
        if self.iteration == 0:
            return

        if (self.fig is None):
            self.initFigure()

        plt.figure(self.fig.number)

        plot = True
        if not increment is None:
            if (np.mod(self.iteration, increment) != 0):
                plot = False

        if plot:

            self._plotAcceptanceVsIteration()

            # Update the data misfit vs iteration
            self._plotMisfitVsIteration()

            self.model.plot_posteriors(
                axes=self.ax[2],
                ncells_kwargs={
                    'normalize': True},
                edges_kwargs={
                    'normalize': True,
                    'rotate': True,
                    'flipY': True,
                    'trim': False},
                parameter_kwargs={
                    # 'reciprocateX':self.reciprocateParameter,
                    'noColorbar': True,
                    'flipY': True,
                    'xscale': 'log',
                    'credible_interval_kwargs': {
                        # 'log':10,
                        # 'reciprocate':True
                    }
                },
                best=self.best_model)

            self.datapoint.plot_posteriors(
                axes=self.ax[3],
                height_kwargs={
                    'normalize': True},
                data_kwargs={},
                rel_error_kwargs={
                    'normalize': True},
                add_error_kwargs={
                    'normalize': True},
                best=self.best_datapoint)

            cP.suptitle(title)

            # self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            cP.pause(1e-9)

    def _plotAcceptanceVsIteration(self, **kwargs):
        """ Plots the acceptance percentage against iteration. """

        i = np.s_[:np.int64(self.iteration / 1000)]
        self.acceptance_rate.plot(self.acceptance_x, i=i,
                       ax=self.ax[0],
                       marker='o',
                       alpha=0.7,
                       linestyle='none',
                       markeredgecolor='k',
                       color='k'
                       )
        # cP.xlabel('Iteration #')
        # cP.ylabel('% Acceptance')
        self.ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    def _plotMisfitVsIteration(self, **kwargs):
        """ Plot the data misfit against iteration. """

        kwargs['ax'] = self.ax[1]
        m = kwargs.pop('marker', '.')
        ms = kwargs.pop('markersize', 2)
        a = kwargs.pop('alpha', 0.7)
        ls = kwargs.pop('linestyle', 'none')
        c = kwargs.pop('color', 'k')
        lw = kwargs.pop('linewidth', 3)

        ax = self.data_misfit_v.plot(self.iRange, i=np.s_[:self.iteration], marker=m, alpha=a, markersize=ms, linestyle=ls, color=c, **kwargs)
        plt.ylabel('Data Misfit')

        dum = self.multiplier * np.sum(self.datapoint.active)
        plt.axhline(dum, color='#C92641', linestyle='dashed', linewidth=lw)
        if (self.burned_in):
            plt.axvline(self.burned_in_iteration, color='#C92641',
                        linestyle='dashed', linewidth=lw)
            # plt.axvline(self.iBest, color=cP.wellSeparated[3])
        # plt.yscale('log')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        plt.xlim([0, self.iRange[self.iteration]])

    # def _plotObservedPredictedData(self, **kwargs):
    #     """ Plot the observed and predicted data """
    #     if self.burnedIn:
    #         # self.datapoint.predictedData.plotPosteriors(noColorbar=True)
    #         self.datapoint.plot(**kwargs)
    #         self.bestDataPoint.plotPredicted(color=cP.wellSeparated[3], **kwargs)
    #     else:

    #         self.datapoint.plot(**kwargs)
    #         self.datapoint.plotPredicted(color='g', **kwargs)

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

        grp = self.datapoint.createHdf(parent,'currentdatapoint', nRepeats=nPoints, fillvalue=np.nan)
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


        # self.meanInterp.createHdf(parent,'meaninterp', nRepeats=nPoints, fillvalue=np.nan)
        # self.bestInterp.createHdf(parent,'bestinterp', nRepeats=nPoints, fillvalue=np.nan)
        # self.opacityInterp.createHdf(parent,'opacityinterp',nRepeats=nPoints, fillvalue=np.nan)
#        parent.create_dataset('opacityinterp', [nPoints,nz], dtype=np.float64)

        self.acceptance_rate.createHdf(parent,'rate',nRepeats=nPoints, fillvalue=np.nan)
#        parent.create_dataset('rate', [nPoints,self.rate.size], dtype=self.rate.dtype)
        self.data_misfit_v.createHdf(
            parent, 'phids', nRepeats=nPoints, fillvalue=np.nan)
        #parent.create_dataset('phids', [nPoints,self.PhiDs.size], dtype=self.PhiDs.dtype)

        self.best_datapoint.createHdf(parent,'bestd', withPosterior=False, nRepeats=nPoints, fillvalue=np.nan)

        # Since the 1D models change size adaptively during the inversion, we need to pad the HDF creation to the maximum allowable number of layers.
        tmp = self.model.pad(self.model.max_cells)

        tmp.createHdf(parent, 'currentmodel', nRepeats=nPoints, fillvalue=np.nan)

        tmp = self.best_model.pad(self.best_model.max_cells)
        tmp.createHdf(parent, 'bestmodel', withPosterior=False, nRepeats=nPoints, fillvalue=np.nan)

    def write_inference1d(self, parent, index=None):
        """ Given a HDF file initialized as line results, write the contents of results to the appropriate arrays """

        # assert self.datapoint.fiducial in self.fiducials, Exception("The HDF file does not have ID number {}. Available ids are between {} and {}".format(inference1d.fiducial, np.min(self.fiducials), np.max(self.fiducials)))

        hdfFile = parent

        # Get the point index
        if index is None:
            fiducials = StatArray.StatArray.fromHdf(parent['currentdatapoint/fiducial'])
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

        self.datapoint.writeHdf(hdfFile,'currentdatapoint',  index=i)

        self.best_datapoint.writeHdf(hdfFile,'bestd', withPosterior=False, index=i)

        self.model.writeHdf(hdfFile,'currentmodel', index=i)

        self.best_model.writeHdf(hdfFile,'bestmodel', withPosterior=False, index=i)


    def read_fromH5Obj(self, h5obj, fName, grpName, system_file_path = ''):
        """ Reads a data points results from HDF5 file """
        grp = h5obj.get(grpName)
        assert not grp is None, "ID "+str(grpName) + " does not exist in file " + fName
        self.fromHdf(grp, system_file_path)


    @classmethod
    def fromHdf(cls, hdfFile, system_file_path, index=None, fiducial=None):

        iNone = index is None
        fNone = fiducial is None

        assert not (iNone and fNone) ^ (not iNone and not fNone), Exception("Must specify either an index OR a fiducial.")

        fiducials = StatArray.StatArray.fromHdf(hdfFile['currentdatapoint/fiducial'])

        if not fNone:
            index = fiducials.searchsorted(fiducial)

        self = cls(None, None, kwargs=None)

        s = np.s_[index, :]

        self.fiducial = np.float64(fiducials[index])

        self._n_markov_chains = np.array(hdfFile.get('nmc'))
        self._update_plot_every = np.array(hdfFile.get('iplot'))
        self.interactive_plot = np.array(hdfFile.get('plotme'))

        tmp = hdfFile.get('limits')
        self.limits = None if tmp is None else np.array(tmp)
        self.reciprocateParameter = np.array(hdfFile.get('reciprocateParameter'))

        self.nSystems = np.array(hdfFile.get('nsystems'))
        self.acceptance_x = hdfRead.readKeyFromFile(hdfFile, '', '/', 'ratex')

        self.iteration = hdfRead.readKeyFromFile(
            hdfFile, '', '/', 'i', index=index)
        self.burned_in_iteration = hdfRead.readKeyFromFile(
            hdfFile, '', '/', 'iburn', index=index)
        self.burned_in = hdfRead.readKeyFromFile(
            hdfFile, '', '/', 'burnedin', index=index)
        # self.doi = hdfRead.readKeyFromFile(hdfFile,'','/','doi', index=index)
        self.multiplier = hdfRead.readKeyFromFile(
            hdfFile, '', '/', 'multiplier', index=index)
        self.acceptance_rate = hdfRead.readKeyFromFile(hdfFile, '', '/', 'rate', index=s)
        self.data_misfit_v = hdfRead.readKeyFromFile(
            hdfFile, '', '/', 'phids', index=s)

        self.best_datapoint = hdfRead.readKeyFromFile(
            hdfFile, '', '/', 'bestd', index=index, system_file_path=system_file_path)

        self.datapoint = hdfRead.readKeyFromFile(
            hdfFile, '', '/', 'currentdatapoint', index=index, system_file_path=system_file_path)

        self.model = hdfRead.readKeyFromFile(
            hdfFile, '', '/', 'currentmodel', index=index)

        self.Hitmap = self.model.par.posterior
        # self.currentModel._max_edge = np.log(self.Hitmap.y.centres[-1])
        # except:
        #     self.Hitmap = hdfRead.readKeyFromFile(hdfFile,'','/','hitmap', index=index)

        self.best_model = hdfRead.readKeyFromFile(
            hdfFile, '', '/', 'bestmodel', index=index)
        # self.bestModel._max_edge = np.log(self.Hitmap.y.centres[-1])

        self.invTime = np.array(hdfFile.get('invtime')[index])
        self.saveTime = np.array(hdfFile.get('savetime')[index])

        # Initialize a list of iteration number
        self.iRange = StatArray.StatArray(
            np.arange(2 * self.n_markov_chains), name="Iteration #", dtype=np.int64)

        self.verbose = False

        self.plotMe = True

        return self

    # def verbose(self):
    #     # if self.verbose & self.burnedIn:

    #     if self.verbose:
    #         self.verboseFigs = []
    #         self.verboseAxs = []

    #         # Posterior components
    #         fig = plt.figure(facecolor='white', figsize=(10,7))
    #         self.verboseFigs.append(fig)
    #         self.verboseAxs.append(fig.add_subplot(511))
    #         self.verboseAxs.append(fig.add_subplot(512))
    #         self.verboseAxs.append(fig.add_subplot(513))

    #         fig = plt.figure(facecolor='white', figsize=(10,7))
    #         self.verboseFigs.append(fig)
    #         for i in range(8):
    #             self.verboseAxs.append(fig.add_subplot(8, 1, i+1))

    #         # Cross Plots
    #         fig = plt.figure(facecolor='white', figsize=(10,7))
    #         self.verboseFigs.append(fig)
    #         for i in range(4):
    #             self.verboseAxs.append(fig.add_subplot(1, 4, i+1))

    #         # ratios vs iteration number
    #         fig = plt.figure(facecolor='white', figsize=(10,7))
    #         self.verboseFigs.append(fig)
    #         for i in range(5):
    #             self.verboseAxs.append(fig.add_subplot(5, 1, i+1))

    #         for ax in self.verboseAxs:
    #             cP.pretty(ax)

    #     plt.figure(self.verboseFigs[0].number)
    #     plt.sca(self.verboseAxs[0])
    #     plt.cla()
    #     self.allRelErr[0, :].plot(self.iRange, i=np.s_[:self.i], c='k')
    #     plt.sca(self.verboseAxs[1])
    #     plt.cla()
    #     self.allAddErr[0, :].plot(self.iRange, i=np.s_[:self.i], axis=1, c='k')
    #     plt.sca(self.verboseAxs[2])
    #     plt.cla()
    #     self.allZ.plot(x=self.iRange, i=np.s_[:self.i], marker='o', linestyle='none', markersize=2, alpha=0.3, markeredgewidth=1)

    #     # Posterior components plot Figure 1
    #     labels=['nCells','depth','parameter','gradient','relative','additive','height','calibration']
    #     plt.figure(self.verboseFigs[1].number)
    #     for i in range(8):
    #         plt.sca(self.verboseAxs[3 + i])
    #         plt.cla()
    #         self.posteriorComponents[i, :].plot(linewidth=0.5)
    #         plt.ylabel('')
    #         plt.title(labels[i])
    #         if labels[i] == 'gradient':
    #             plt.ylim([-30.0, 1.0])

    #     ira = self.iRange[:np.int(1.2*self.n_markov_chains)][self.accepted]
    #     irna = self.iRange[:np.int(1.2*self.n_markov_chains)][~self.accepted]

    #     plt.figure(self.verboseFigs[3].number)
    #     # Number of layers vs iteration
    #     plt.sca(self.verboseAxs[15])
    #     plt.cla()
    #     self.allK[~self.accepted].plot(x = irna, marker='o', markersize=1,  linestyle='None', alpha=0.3, color='k')
    #     self.allK[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
    #     plt.title('black = rejected')

    #     plt.figure(self.verboseFigs[2].number)
    #     # Cross plot of current vs candidate prior
    #     plt.sca(self.verboseAxs[11])
    #     plt.cla()
    #     x = StatArray.StatArray(self.ratioComponents[0, :], 'Candidate Prior')
    #     y = StatArray.StatArray(self.ratioComponents[1, :], 'Current Prior')

    #     x[x == -np.inf] = np.nan
    #     y[y == -np.inf] = np.nan
    #     x[~self.accepted].plot(x = y[~self.accepted], linestyle='', marker='.', color='k', alpha=0.3)
    #     x[self.accepted].plot(x = y[self.accepted], linestyle='', marker='.', alpha=0.3)
    #     # v1 = np.maximum(np.minimum(np.nanmin(x), np.nanmin(y)), -20.0)
    #     v2 = np.maximum(np.nanmax(x), np.nanmax(y))
    #     v1 = v2 - 25.0
    #     plt.xlim([v1, v2])
    #     plt.ylim([v1, v2])
    #     plt.plot([v1,v2], [v1,v2])

    #     # Prior ratio vs iteration
    #     plt.figure(self.verboseFigs[3].number)
    #     plt.sca(self.verboseAxs[16])
    #     plt.cla()
    #     r = x - y
    #     r[~self.accepted].plot(x = irna, marker='o', markersize=1, linestyle='None', alpha=0.3, color='k')
    #     r[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
    #     plt.ylim([v1, 5.0])
    #     cP.ylabel('Prior Ratio')

    #     plt.figure(self.verboseFigs[2].number)
    #     # Cross plot of the likelihood ratios
    #     plt.sca(self.verboseAxs[12])
    #     plt.cla()
    #     x = StatArray.StatArray(self.ratioComponents[2, :], 'Candidate Likelihood')
    #     y = StatArray.StatArray(self.ratioComponents[3, :], 'Current Likelihood')
    #     x[~self.accepted].plot(x = y[~self.accepted], linestyle='', marker='.', color='k', alpha=0.3)
    #     x[self.accepted].plot(x = y[self.accepted], linestyle='', marker='.', alpha=0.3)

    #     v2 = np.maximum(np.nanmax(x), np.nanmax(y)) + 5.0
    #     v1 = v2 - 200.0
    #     # v1 = -100.0
    #     # v2 = -55.0
    #     plt.xlim([v1, v2])
    #     plt.ylim([v1, v2])
    #     plt.plot([v1, v2], [v1, v2])
    #     plt.title('black = rejected')

    #     plt.figure(self.verboseFigs[3].number)
    #     # Likelihood ratio vs iteration
    #     plt.sca(self.verboseAxs[17])
    #     plt.cla()
    #     r = x - y
    #     r[~self.accepted].plot(x = irna, marker='o', markersize=1, linestyle='None', alpha=0.3, color='k')
    #     r[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
    #     cP.ylabel('Likelihood Ratio')
    #     plt.ylim([-20.0, 20.0])

    #     plt.figure(self.verboseFigs[2].number)
    #     # Cross plot of the proposal ratios
    #     plt.sca(self.verboseAxs[13])
    #     plt.cla()
    #     y = StatArray.StatArray(self.ratioComponents[4, :], 'Current Proposal')
    #     x = StatArray.StatArray(self.ratioComponents[5, :], 'Candidate Proposal')
    #     x[~self.accepted].plot(x = y[~self.accepted], linestyle='', marker='.', color='k', alpha=0.3)
    #     x[self.accepted].plot(x = y[self.accepted], linestyle='', marker='.', alpha=0.3)
    #     # v1 = np.maximum(np.minimum(np.nanmin(x), np.nanmin(y)), -200.0)
    #     v2 = np.maximum(np.nanmax(x), np.nanmax(y)) + 10.0
    #     v1 = v2 - 60.0
    #     v1 = -20.0
    #     v2 = 20.0
    #     # plt.plot([v1,v2], [v1,v2])
    #     plt.xlim([v1, v2])
    #     plt.ylim([v1, v2])

    #     plt.figure(self.verboseFigs[2].number)
    #     # Cross plot of the proposal ratios coloured by a change in dimension
    #     plt.sca(self.verboseAxs[14])
    #     plt.cla()
    #     y = StatArray.StatArray(self.ratioComponents[4, :], 'Current Proposal')
    #     x = StatArray.StatArray(self.ratioComponents[5, :], 'Candidate Proposal')
    #     x[~self.dimensionChange].plot(x = y[~self.dimensionChange], linestyle='', marker='.', color='k', alpha=0.3)
    #     x[self.dimensionChange].plot(x = y[self.dimensionChange], linestyle='', marker='.', alpha=0.3)
    #     # v1 = np.maximum(np.minimum(np.nanmin(x), np.nanmin(y)), -200.0)
    #     # v2 = np.maximum(np.nanmax(x), np.nanmax(y)) + 10.0
    #     # v1 = v2 - 60.0

    #     # plt.plot([v1,v2], [v1,v2])
    #     plt.xlim([v1, v2])
    #     plt.ylim([v1, v2])
    #     plt.title('black = no dimension change')

    #     plt.figure(self.verboseFigs[3].number)
    #     # Proposal ratio vs iteration
    #     plt.sca(self.verboseAxs[18])
    #     plt.cla()
    #     r = x - y
    #     r[~self.accepted].plot(x = irna, marker='o', markersize=1, linestyle='None', alpha=0.3, color='k')
    #     r[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
    #     cP.ylabel('Proposal Ratio')
    #     plt.ylim([v1, v2])

    #     # Acceptance ratio vs iteration
    #     plt.sca(self.verboseAxs[19])
    #     plt.cla()
    #     x = StatArray.StatArray(self.ratioComponents[6, :], 'Acceptance Ratio')
    #     x[~self.accepted].plot(x = irna, marker='o', markersize=1, linestyle='None', alpha=0.3, color='k')
    #     x[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
    #     plt.ylim([-20.0, 20.0])

    #     for fig in self.verboseFigs:
    #         fig.canvas.draw()
    #         fig.canvas.flush_events()

        # if verbose:
        #     n = np.int(1.2*self.n_markov_chains)
        #     self.allRelErr = StatArray.StatArray(np.full([self.nSystems, n], np.nan), name='$\epsilon_{Relative}x10^{2}$', units='%')
        #     self.allAddErr = StatArray.StatArray(np.full([self.nSystems, n], np.nan), name='$\epsilon_{Additive}$', units=dataPoint.data.units)
        #     self.allZ = StatArray.StatArray(np.full(n, np.nan), name='Height', units='m')
        #     self.allK = StatArray.StatArray(np.full(n, np.nan), name='Number of Layers')
        #     self.posteriorComponents = StatArray.StatArray(np.full([8, n], np.nan), 'Components of the posterior')
        #     self.ratioComponents = StatArray.StatArray(np.full([7, n], np.nan), 'log(Ratio Components)')
        #     self.accepted = StatArray.StatArray(np.zeros(n, dtype=bool), name='Accepted')
        #     self.dimensionChange = StatArray.StatArray(np.zeros(n, dtype=bool), name='Dimensions were changed')

    # if (self.verbose):
    #        fig = plt.figure(1)
    #        fig.set_size_inches(19, 11)
    #        figName = join(directory,str(fiducial) + '_rap.png')
    #        plt.savefig(figName, dpi=dpi)

    #        fig = plt.figure(2)
    #        fig.set_size_inches(19, 11)
    #        figName = join(directory,str(fiducial) + '_posterior_components.png')
    #        plt.savefig(figName, dpi=dpi)

    #        fig = plt.figure(3)
    #        fig.set_size_inches(19, 11)
    #        figName = join(directory,str(fiducial) + '_ratio_crossplot.png')
    #        plt.savefig(figName, dpi=dpi)

    #        fig = plt.figure(4)
    #        fig.set_size_inches(19, 11)
    #        figName = join(directory,str(fiducial) + '_ratios_vs_iteration.png')
    #        plt.savefig(figName, dpi=dpi)
