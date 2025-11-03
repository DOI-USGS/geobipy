""" @Inference1D
Class to store inversion results. Contains plotting and writing to file procedures
"""
from copy import deepcopy
from os.path import join
import traceback
from datetime import timedelta

from numpy import argwhere, asarray, reshape, size, int64, sum, linspace, float64, int32, uint8
from numpy import arange, inf, isclose, mod, s_, maximum, any, isnan, sort, nan
from numpy import max, min, log, log10, array, full, longdouble, exp, maximum, sqrt

from numpy.random import Generator
from numpy.linalg import norm

import matplotlib as mpl
import matplotlib.pyplot as plt

from ..base import plotting as cP
from ..base.utilities import expReal
from ..base.utilities import debug_print as dprint

import h5py
from ..classes.core.DataArray import DataArray
from ..classes.statistics.StatArray import StatArray
from ..classes.statistics.Distribution import Distribution
from ..classes.statistics.Histogram import Histogram
from ..classes.core.myObject import myObject
from ..classes.data.datapoint.DataPoint import DataPoint
# from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
# from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ..classes.model.Model import Model
from ..classes.core.Stopwatch import Stopwatch
from ..base.HDF import hdfRead
from cached_property import cached_property
from .user_parameters import user_parameters

class Inference1D(myObject):
    """Define the results for the Bayesian MCMC Inversion.

    Contains histograms and inversion related variables that can be updated as the Bayesian inversion progresses.

    Inference1D(saveMe, plotMe, savePNG, dataPoint, model, ID, **kwargs)

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

    def __init__(self,
                 covariance_scaling:float = 1.0,
                 high_variance:float = inf,
                 ignore_likelihood:bool = False,
                 interactive_plot:bool = True,
                 low_variance:float = -inf,
                 multiplier:float = 1.0,
                 n_markov_chains:int = 100000,
                 parameter_limits = None,
                 prng=None,
                 reciprocate_parameters:bool = False,
                 reset_limit:int = 1,
                 save_hdf5:bool = True,
                 save_png:bool = False,
                 solve_gradient:bool = True,
                 solve_parameter:bool = False,
                 update_plot_every:int = 5000,
                 minimum_burn_in:int = 5000,
                 world = None,
                 **kwargs):
        """ Initialize the results of the inversion """

        self.world = world

        self.options = kwargs

        self.prng = prng
        self.ignore_likelihood = ignore_likelihood
        self.n_markov_chains = n_markov_chains
        self.multiplier = multiplier
        self.solve_gradient = solve_gradient
        self.solve_parameter = solve_parameter
        self.save_hdf5 = save_hdf5
        self.interactive_plot = interactive_plot
        self.save_png = save_png
        self.update_plot_every = update_plot_every
        self.limits = parameter_limits
        self.reciprocate_parameter = reciprocate_parameters
        self.reset_limit = reset_limit
        self.low_variance = low_variance
        self.high_variance = high_variance
        self.covariance_scaling = covariance_scaling
        self.minimum_burn_in = minimum_burn_in

        assert self.interactive_plot or self.save_hdf5, Exception('You have chosen to neither view or save the inversion results!')

        self._n_zero_acceptance = 0
        self._n_resets = 0

        self.posterior_fig = None
        self.posterior_ax = None

    @property
    def acceptance_percent(self):
        if self.iteration > self.update_plot_every:
            s = sum(self.acceptance_v[self.iteration-self.update_plot_every:self.iteration]) / float64(self.update_plot_every)
        else:
            s = sum(self.acceptance_v[:self.iteration]) / float64(self.iteration)
        return 100.0 * s

    @property
    def covariance_scaling(self):
        return self.options['covariance_scaling']

    @covariance_scaling.setter
    def covariance_scaling(self, value):
        self.options['covariance_scaling'] = float64(value)

    @property
    def observed_datapoint(self):
        return self._observed_datapoint

    @observed_datapoint.setter
    def observed_datapoint(self, value):
        assert isinstance(value, DataPoint), TypeError("observed_datapoint must have type geobipy.Datapoint")
        self._observed_datapoint = deepcopy(value)

    @property
    def datapoint(self):
        return self._datapoint

    @datapoint.setter
    def datapoint(self, value):
        assert isinstance(value, DataPoint), TypeError("datapoint must have type geobipy.Datapoint")
        self._datapoint = value

    @property
    def high_variance(self):
        return self.options['high_variance']

    @high_variance.setter
    def high_variance(self, value):
        self.options['high_variance'] = float64(value)

    @cached_property
    def iz(self):
        return arange(self.model.values.posterior.y.nCells.item())

    @property
    def ignore_likelihood(self):
        return self.options['ignore_likelihood']

    @ignore_likelihood.setter
    def ignore_likelihood(self, value:bool):
        assert isinstance(value, bool), ValueError('ignore_likelihood must have type bool')
        self._options['ignore_likelihood'] = value

    @property
    def interactive_plot(self):
        return self.options['interactive_plot']

    @interactive_plot.setter
    def interactive_plot(self, value:bool):
        assert isinstance(value, bool), ValueError('interactive_plot must have type bool')
        if self.mpi_enabled:
            value = False
        self.options['interactive_plot'] = value

    @property
    def limits(self):
        return self.options['limits']

    @limits.setter
    def limits(self, values):
        if values is not None:
            assert size(values) == 2, ValueError("Limits must have length 2")
            values = sort(asarray(values, dtype=float64))
        self.options['limits'] = values

    @property
    def low_variance(self):
        return self.options['low_variance']

    @low_variance.setter
    def low_variance(self, value):
        self.options['low_variance'] = float64(value)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        assert isinstance(value, Model), TypeError("model must have type geobipy.Model")
        self._model = value

    @property
    def mpi_enabled(self):
        return self.world is not None

    @property
    def multiplier(self):
        return self.options['multiplier']

    @multiplier.setter
    def multiplier(self, value):
        self.options['multiplier'] = float64(value)

    @property
    def n_markov_chains(self):
        return self.options['n_markov_chains']

    @n_markov_chains.setter
    def n_markov_chains(self, value):
        self.options['n_markov_chains'] = int64(value)

    @property
    def reset_limit(self):
        return self.options['reset_limit']

    @reset_limit.setter
    def reset_limit(self, value):
        self.options['reset_limit'] = int64(value)

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, value):
        assert isinstance(value, dict), TypeError("options must have type dict")
        self._options = value

    @property
    def prng(self):
        return self.options['prng']

    @prng.setter
    def prng(self, value):
        assert isinstance(value, Generator), TypeError(("prng must have type np.random.Generator.\n"
                                                        "You can generate one using\n"
                                                        "from numpy.random import Generator\n"
                                                        "from numpy.random import PCG64DXSM\n"
                                                        "Generator(bit_generator)\n\n"
                                                        "Where bit_generator is one of the several generators from either numpy or randomgen"))

        self.options['prng'] = value

    @property
    def rank(self):
        if self.mpi_enabled:
            return self.world.rank
        else:
            return 1

    @property
    def reciprocate_parameters(self):
        return self.options['reciprocate_parameters']

    @reciprocate_parameters.setter
    def reciprocate_parameters(self, value:bool):
        assert isinstance(value, bool), ValueError('reciprocate_parameters must have type bool')
        self.options['reciprocate_parameters'] = value

    @property
    def save_hdf5(self):
        return self.options['save_hdf5']

    @save_hdf5.setter
    def save_hdf5(self, value:bool):
        assert isinstance(value, bool), ValueError('save_hdf5 must have type bool')
        self.options['save_hdf5'] = value

    @property
    def save_png(self):
        return self.options['save_png']

    @save_png.setter
    def save_png(self, value:bool):
        assert isinstance(value, bool), ValueError('save_png must have type bool')
        if self.mpi_enabled:
            value = False
        self.options['save_png'] = value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @property
    def solve_parameter(self):
        return self.options['solve_parameter']

    @solve_parameter.setter
    def solve_parameter(self, value:bool):
        assert isinstance(value, bool), ValueError('solve_parameter must have type bool')
        self.options['solve_parameter'] = value

    @property
    def solve_gradient(self):
        return self.options['solve_gradient']

    @solve_gradient.setter
    def solve_gradient(self, value:bool):
        assert isinstance(value, bool), ValueError('solve_gradient must have type bool')
        self.options['solve_gradient'] = value

    @property
    def stochastic_newton(self):
        return self.options['stochastic_newton']

    @property
    def update_plot_every(self):
        return self.options['update_plot_every']

    @update_plot_every.setter
    def update_plot_every(self, value):
        self.options['update_plot_every'] = int32(value)

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, value):
        self._world = value


    def initialize(self, datapoint):
        # Get the initial best fitting halfspace and set up
        # priors and posteriors using user parameters
        # ------------------------------------------------
        # Intialize the datapoint with the user parameters
        # ------------------------------------------------
        self.initialize_datapoint(datapoint, **self.options)

        # # Initialize the calibration parameters
        # if (kwargs.solveCalibration):
        #     datapoint.calibration.set_prior('Normal',
        #                            reshape(kwargs.calMean, size(kwargs.calMean), order='F'),
        #                            reshape(kwargs.calVar, size(kwargs.calVar), order='F'), prng=prng)
        #     datapoint.calibration[:] = datapoint.calibration.prior.mean
        #     # Initialize the calibration proposal
        #     datapoint.calibration.setProposal('Normal', datapoint.calibration, reshape(kwargs.propCal, size(kwargs.propCal), order='F'), prng=prng)

        # ---------------------------------
        # Set the earth model properties
        # ---------------------------------
        self.initialize_model(**self.options)

        # Compute the data misfit
        self.data_misfit = datapoint.data_misfit()

        self.data_misfit_v = StatArray(2 * self.n_markov_chains, name='Data Misfit')
        self.data_misfit_v[0] = self.data_misfit
        self._n_target_hits = 0
        self.data_misfit_v.prior = Distribution('chi2', df=sum(self.datapoint.active), prng=self.prng)

        # # Calibrate the response if it is being solved for
        # if (self.kwargs.solveCalibration):
        #     self.datapoint.calibrate()

        # Evaluate the prior for the current model
        self.prior = self.model.probability(self.solve_parameter,
                                            self.solve_gradient)

        self.prior += self.datapoint.probability

        # Initialize the burned in state
        self.burned_in_iteration = self.n_markov_chains
        self.burned_in = True

        # Add the likelihood function to the prior
        self.likelihood = 1.0
        if not self.ignore_likelihood:
            self.likelihood = self.datapoint.likelihood(log=True)
            self.burned_in = False
            self.burned_in_iteration = int64(0)

        self.posterior = self.likelihood + self.prior

        # Initialize the current iteration number
        # Current iteration number
        self.iteration = int64(0)

        # Initialize the vectors to save results
        # StatArray of the data misfit
        self.relative_chi_squared_fit = 100.0

        edges = DataArray(linspace(1, 2*sum(self.datapoint.active)))
        self.data_misfit_v.posterior = Histogram(mesh = RectilinearMesh1D(edges=edges))

        # Initialize a stopwatch to keep track of time
        self.clk = Stopwatch()
        self.invTime = float64(0.0)

        # Return none if important parameters are not used (used for hdf 5)
        if datapoint is None:
            return

        assert self.interactive_plot or self.save_hdf5, Exception(
            'You have chosen to neither view or save the inversion results!')

        # Set the ID for the data point the results pertain to

        # Set the increment at which to plot results
        # Increment at which to update the results

        # Set the display limits of the parameter in the HitMap
        # Display limits for parameters
        # Should we plot resistivity or Conductivity?
        # Logical whether to take the reciprocal of the parameters

        # Multiplier for discrepancy principle
        self.multiplier = float64(1.0)

        # Initialize the acceptance level
        # Model acceptance rate
        self.accepted = 0

        self.acceptance_v = DataArray(full(2 * self.n_markov_chains, fill_value=0, dtype=uint8), name='% Acceptance')

        n = 2 * int32(self.n_markov_chains / self.update_plot_every)
        self.acceptance_x = DataArray(arange(1, n + 1) * self.update_plot_every, name='Iteration #')
        self.acceptance_rate = DataArray(full(n, fill_value=nan), name='% Acceptance')

        self.iRange = DataArray(arange(2 * self.n_markov_chains), name="Iteration #", dtype=int64)

        # Initialize time in seconds
        self.inference_time = float64(0.0)

        # Initialize the best data, current data and best model
        self.best_model = deepcopy(self.model)
        self.best_datapoint = deepcopy(self.datapoint)
        self.best_posterior = self.posterior
        self.best_iteration = int64(0)

        # if self.interactive_plot:
        #     self._init_posterior_plots()
        #     plt.show(block=False)

    def initialize_datapoint(self, datapoint, **kwargs):

        self.datapoint = datapoint

        # ---------------------------------------
        # Set the statistical properties of the datapoint
        # ---------------------------------------
        # Set the prior on the data
        self.datapoint.initialize(**kwargs)
        self.observed_datapoint = deepcopy(datapoint)
        # Set the priors, proposals, and posteriors.
        self.datapoint.set_priors(**kwargs)
        self.datapoint.set_proposals(**kwargs)
        self.datapoint.set_posteriors()

    def initialize_model(self, **kwargs):
        # Find the conductivity of a half space model that best fits the data
        halfspace = self.datapoint.find_best_halfspace()

        # dprint('halfspace', halfspace.values)
        self.halfspace = halfspace.values

        # Create an initial model for the first iteration
        # Initialize a 1D model with the half space conductivity
        # Assign the depth to the interface as half the bounds
        self.model = deepcopy(halfspace)

        # Setup the model for perturbation
        self.model.set_priors(
            value_mean=kwargs.pop('value_mean', halfspace.values.item()),
            min_edge=kwargs['minimum_depth'],
            max_edge=kwargs['maximum_depth'],
            max_cells=kwargs['maximum_number_of_layers'],
            solve_value=True, #self.solve_parameter,
            # solve_gradient=self.solve_gradient,
            parameter_limits=self.limits,
            min_width=kwargs.get('minimum_thickness', None),
            # factor=kwargs.get('factor', 10.0),
            **kwargs
        )

        # Assign a Hitmap as a prior if one is given
        # if (not self.kwargs.referenceHitmap is None):
        #     Mod.setReferenceHitmap(self.kwargs.referenceHitmap)

        # Compute the predicted data
        self.datapoint.forward(self.model)

        observation = self.datapoint
        if self.ignore_likelihood:
            observation = None
        else:
            observation.sensitivity(self.model)

        self.model.set_proposal_weights(**kwargs)

        local_variance = self.model.local_variance(observation)

        # Instantiate the proposal for the parameters.
        parameterProposal = Distribution('MvLogNormal', mean=self.model.values, variance=local_variance, linearSpace=True, prng=self.prng)

        probabilities = [kwargs['probability_of_birth'],
                         kwargs['probability_of_death'],
                         kwargs['probability_of_perturb'],
                         kwargs['probability_of_no_change']]

        self.model.set_proposals(probabilities=probabilities, proposal=parameterProposal, **kwargs)

        self.model.set_posteriors(**kwargs)

    def accept_reject(self):
        """ Propose a new random model and accept or reject it """
        import numpy as np
        dprint(f'\n\n{self.iteration=}')

        dprint(f'{self.prng.random()=}')

        dprint(f'incoming {self.datapoint.data=}')
        dprint(f'incoming {self.datapoint.predicted_data=}')
        dprint(f'incoming {self.model.values=}')
        test_datapoint = deepcopy(self.datapoint)

        # Perturb the current model
        observation = None
        if self.stochastic_newton and not self.ignore_likelihood:
            observation = test_datapoint

        # try:
        remapped_model, test_model = self.model.perturb(observation, alpha = self.covariance_scaling)
        # except Exception:
        #     # print(f'singularity --line={observation.line_number.item()} --fiducial={observation.fiducial.item()} --jump={self.rank} iteration={self.iteration}', flush=True)
        #     print(traceback.format_exc())
        #     return True

        if remapped_model is None:
            self.accepted = False
            return

        # # Propose a new data point, using assigned proposal distributions
        test_datapoint.perturb()

        # Forward model the data from the candidate model
        test_datapoint.forward(test_model)

        # J is now centered on the perturbed
        test_data_misfit = test_datapoint.data_misfit()
        dprint(f"{test_data_misfit=}")

        # Evaluate the prior for the current data
        test_prior = test_datapoint.probability
        # Test for early rejection
        if (test_prior == -inf):
            self.accepted = False
            return

        # Evaluate the prior for the current model
        test_prior += test_model.probability(self.solve_parameter, self.solve_gradient)

        # Test for early rejection
        if (test_prior == -inf):
            self.accepted = False
            return

        # Compute the components of each acceptance ratio
        test_likelihood = 1.0
        observation = None
        if not self.ignore_likelihood:
            test_likelihood = test_datapoint.likelihood(log=True)
            observation = test_datapoint

        proposal, test_proposal = test_model.proposal_probabilities(remapped_model, observation, alpha = self.covariance_scaling)

        test_posterior = test_prior + test_likelihood

        prior_ratio = test_prior - self.prior

        likelihood_ratio = test_likelihood - self.likelihood

        proposal_ratio = proposal - test_proposal

        log_acceptance_ratio = prior_ratio + likelihood_ratio + proposal_ratio
        acceptance_probability = expReal(log_acceptance_ratio)

        # If we accept the model
        self.accepted = acceptance_probability > self.prng.uniform()

        if (self.accepted):
            # Compute the data misfit
            self.data_misfit = test_data_misfit
            self.prior = test_prior
            self.likelihood = test_likelihood
            self.posterior = test_posterior
            self.model = test_model
            self.datapoint = test_datapoint
            # Reset the sensitivity locally to the newly accepted model
            # self.datapoint.sensitivity(self.model, model_changed=False)

        dprint('accepted', self.accepted)

        # input('NEXT')

        return False

    def infer(self, hdf_file_handle):
        """ Markov Chain Monte Carlo approach for inversion of geophysical data
        userParameters: User input parameters object
        DataPoint: Datapoint to invert
        ID: Datapoint label for saving results
        pHDFfile: Optional HDF5 file opened using h5py.File('name.h5','w',driver='mpio', comm=world) before calling Inv_MCMC
        """

        Go = self.datapoint.n_active_channels > 0

        self.clk.start()

        failed = not Go
        while (Go):
            # Accept or reject the new model
            try:
                failed = self.accept_reject()
            except Exception as e:
                print(f'singularity --line={self.datapoint.line_number.item()} --fiducial={self.datapoint.fiducial.item()} --jump={self.rank} iteration={self.iteration}', flush=True)
                print(traceback.format_exc())
                failed = True

            self.update()

            if self.interactive_plot:
                self.plot_posteriors(axes=self.posterior_ax,
                                     title="Fiducial {}".format(self.datapoint.fiducial),
                                     increment=self.update_plot_every)

            Go = not failed and (self.iteration <= self.n_markov_chains + self.burned_in_iteration)

            if (not failed) and (not self.burned_in):
                Go = self.iteration < self.n_markov_chains
                if not Go:
                    failed = True


            # if self._n_resets == 3 and not self.burned_in:
            #     if self.low_variance == -inf:
            #         # If we reset 3 times, we might have either too low or high a proposal variance.
            #         # Add limiters and try again.
            #         self.low_variance = 0.1
            #         self.high_variance = 2.0
            #         self._n_resets = 0
            #         self.reset()

                # If we tried limiters and reset again 3 times, fail the datapoint.
                # else:
                #     Go = False
                #     failed = True

        self.clk.stop()

        if self.save_hdf5:
            self.writeHdf(hdf_file_handle)

        if self.save_png:
            self.plot_posteriors(axes = self.posterior_ax, fig=self.posterior_fig)
            self.toPNG(f'.//{self.datapoint.fiducial.item()}.png')

        return failed

    def __deepcopy__(self, memo={}):
        return None

    @property
    def hitmap(self):
        return self.model.values.posterior

    @cached_property
    def chisquare_pdf(self):
        return self.data_misfit_v.prior.probability(self.data_misfit_v.posterior.mesh.centres, log=False)

    @cached_property
    def norm_chisquare(self):
        return norm(self.chisquare_pdf)

    def update(self):
        """Update the posteriors of the McMC algorithm. """

        self.iteration += 1

        self.data_misfit_v[self.iteration - 1] = self.data_misfit

        # Check the fit of the chi square distribution with theoretical
        self.relative_chi_squared_fit = norm(self.data_misfit_v.posterior.pdf.values - self.chisquare_pdf)/self.norm_chisquare

        # Determine if we are burning in
        if (not self.burned_in):
            target_misfit = sum(self.datapoint.active)

            # # if self.data_misfit < target_misfit:
            # if (self.iteration > 10000) and (isclose(self.data_misfit, self.multiplier*target_misfit, rtol=1e-1, atol=1e-2)):
            #     self._n_target_hits += 1

            # converged = (((self.iteration > 10000) and (self.relative_chi_squared_fit < 1.0)) or
            #              ((self.iteration > 10000) and (self._n_target_hits > 1000)))

            converged = (self.iteration > self.minimum_burn_in) and (self.data_misfit < target_misfit)

            if converged:
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

        self.acceptance_v[self.iteration] = self.accepted

        if ((self.iteration > 0) and (mod(self.iteration, self.update_plot_every) == 0)):
            self.acceptance_rate[int32(self.iteration / self.update_plot_every)-1] = self.acceptance_percent

        if (mod(self.iteration, self.update_plot_every) == 0):
            time_per_model = self.clk.lap() / self.update_plot_every
            elapsed = self.clk.timeinSeconds()
            burned_in = "" if self.burned_in else "*"

            eta = "--:--:--"
            if self.burned_in:
                eta = str(timedelta(seconds=int(time_per_model * (self.n_markov_chains + self.burned_in_iteration - self.iteration))))

            tmp = "i=%i, k=%i, acc=%s%4.3f, %4.3f s/Model, %0.3f s Elapsed, eta=%s h:m:s\n" % (self.iteration, float64(self.model.nCells[0]), burned_in, self.acceptance_percent, time_per_model, elapsed, eta)
            if (self.rank == 1):
                print(tmp, flush=True)

            # Test resetting of the inversion.
            # if self.update_plot_every > 1:
            #     if not self.burned_in:
            #         if self.acceptance_percent == 0.0:

            #             self._n_zero_acceptance += 1

            #             # Reset if we have 3 zero acceptances
            #             if self._n_zero_acceptance == self.reset_limit:
            #                 self.reset()
            #                 self._n_zero_acceptance = 0
            #         else:
            #             self._n_zero_acceptance = 0
            #     else:
            #         if self.acceptance_percent == 0.0:
            #             self.low_variance = -inf
            #             self.high_variance = inf

            if (not self.burned_in and not self.datapoint.relative_error.hasPrior):
                self.multiplier *= self.options['multiplier']

        # Added the layer depths to a list, we histogram this list every
        # iPlot iterations
        self.model.update_posteriors(0.5)#self.user_options.clip_ratio)

        # Update the height posterior
        self.datapoint.update_posteriors()

    def _init_posterior_plots(self, **kwargs):
        """ Initialize the plotting region """
        # Setup the figure region. The figure window is split into a 4x3
        # region. Columns are able to span multiple rows

        fig = plt.figure(facecolor='white', figsize=(16, 9))

        if self.interactive_plot:
            plt.interactive(True)

        # fig.clf()

        gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=(1, 6))

        ax = {}
        ax['acceptance_rate'] = cP.pretty(plt.subplot(gs[0, 0]))

        splt = gs[0, 1].subgridspec(1, 2, width_ratios=[4, 1])

        ax['misfit'] = cP.pretty(plt.subplot(splt[0, 0]))
        ax['chisq'] = cP.pretty(plt.subplot(splt[0, 1]))

        ax['model'] = self.model._init_posterior_plots(gs[1, 0])

        ax['data'] = self.datapoint._init_posterior_plots(gs[1, 1])

        if self.interactive_plot:
            plt.show(block=False)

        self.posterior_fig, self.posterior_ax = fig, ax

        return fig, ax

    def plot_posteriors(self, title="", increment=None, **kwargs):
        """ Updates the figures for MCMC Inversion """
        # Plots that change with every iteration
        if self.iteration == 0:
            return

        plot = True
        if increment is not None:
            if (mod(self.iteration, increment) != 0):
                plot = False

        if not plot:
            return

        if self.posterior_ax is None:
            self._init_posterior_plots()

        self._plotAcceptanceVsIteration()

        # Update the data misfit vs iteration
        self._plotMisfitVsIteration()

        overlay = self.best_model if self.burned_in else self.model

        self.model.plot_posteriors(
            axes=self.posterior_ax['model'],
            # ncells_kwargs={
            #     'normalize': True},
            edges_kwargs={
                'transpose': True,
                'trim': False,
                'flipY': True},
            values_kwargs={
                'colorbar': False,
                'xscale': 'log',
                'credible_interval_kwargs': {
                }
            },
            overlay=overlay)


        overlay = self.best_datapoint if self.burned_in else self.datapoint


        # if self.datapoint.hasPosterior:
        self.datapoint.plot_posteriors(
            axes=self.posterior_ax['data'],
            # height_kwargs={
            #     'normalize': True},
            data_kwargs={},
            # rel_error_kwargs={
            #     'normalize': True},
            # add_error_kwargs={
            #     'normalize': True},
            overlay=overlay
        )

        if '_observed_datapoint' in self.__dict__:
            self.datapoint.overlay_on_posteriors(self.observed_datapoint, axes=self.posterior_ax['data'], linecolor='k')
        if self.burned_in:
            self.datapoint.overlay_on_posteriors(self.best_datapoint, axes=self.posterior_ax['data'])

        self.posterior_fig.suptitle(title)

        self.posterior_fig.canvas.draw()
        self.posterior_fig.canvas.flush_events()

        # cP.pause(1e-9)

    def _plotAcceptanceVsIteration(self, **kwargs):
        """ Plots the acceptance percentage against iteration. """

        # i = s_[:int64(self.iteration / self.update_plot_every)]

        # acceptance_rate = self.acceptance_v[:self.iteration]
        # i_positive = argwhere(acceptance_rate > 0.0)
        # i_zero = argwhere(acceptance_rate == 0.0)

        kwargs['ax'] = kwargs.get('ax', self.posterior_ax['acceptance_rate'])
        kwargs['marker'] = kwargs.get('marker', 'o')
        kwargs['alpha'] = kwargs.get('alpha', 0.7)
        kwargs['linestyle'] = kwargs.get('linestyle', 'none')
        kwargs['markeredgecolor'] = kwargs.get('markeredgecolor', 'k')

        i = s_[:int64(self.iteration / self.update_plot_every)]

        self.acceptance_rate.plot(x=self.acceptance_x, i=i, color='k', **kwargs)
        kwargs['ax'].axhline(y=23.4, color='#C92641', linestyle='dashed')


    def _plotMisfitVsIteration(self, **kwargs):
        """ Plot the data misfit against iteration. """

        m = kwargs.pop('marker', '.')
        a = kwargs.pop('alpha', 0.7)
        ls = kwargs.pop('linestyle', 'none')
        c = kwargs.pop('color', 'k')

        ax = self.posterior_ax['misfit']
        ax.cla()
        tmp_ax = self.data_misfit_v.plot(self.iRange, i=s_[:self.iteration], marker=m, alpha=a, linestyle=ls, color=c, ax=ax, **kwargs)
        ax.set_ylabel('Data Misfit')

        dum = self.multiplier * self.data_misfit_v.prior.df
        ax.axhline(dum, color='#C92641', linestyle='dashed')
        if (self.burned_in):
            ax.axvline(self.burned_in_iteration, color='#C92641', linestyle='dashed')

        ax.set_yscale('log')
        tmp_ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        ax.set_xlim([0, self.iRange[self.iteration]])

        if not self.burned_in:
            self.data_misfit_v.reset_posteriors()

        self.data_misfit_v.posterior.update(self.data_misfit_v[maximum(0, self.iteration-self.update_plot_every):self.iteration], trim=True)

        ax = self.posterior_ax['chisq']
        ax.cla()

        misfit_ax, _, _ = self.data_misfit_v.posterior.plot(transpose=True, ax=ax, normalize=True, **kwargs)
        ylim = misfit_ax.get_ylim()

        self.data_misfit_v.prior.plot_pdf(ax=ax, transpose=True, c='#C92641', linestyle='dashed')
        ax.hlines(self.data_misfit_v.prior.df, xmin=0.0, xmax=0.5*ax.get_xlim()[1], color='#C92641', linestyle='dashed')
        ax.set_ylim(ylim)

    # def _plotObservedPredictedData(self, **kwargs):
    #     """ Plot the observed and predicted data """
    #     if self.burnedIn:
    #         # self.datapoint.predicted_data.plot_posteriors(colorbar=False)
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

    def toPNG(self, filename, dpi=300):
       """ save a png of the results """
       self.posterior_fig.set_size_inches(19, 11)
       self.posterior_fig.savefig(filename, dpi=dpi)

    def read(self, fileName, system_file_path, fiducial=None, index=None):
        """ Reads a data point's results from HDF5 file """

        with h5py.File(fileName, 'r')as f:
            R = self.fromHdf(f, system_file_path, index=index, fiducial=fiducial)

        self.plotMe = True
        return self

    def reset(self):
        def clear(this):
            if isinstance(this, dict):
                for k, ax in this.items():
                    clear(ax)
            elif isinstance(this, list):
                for ax in this:
                    clear(ax)
            else:
                this.set_xscale('linear')
                this.cla()

        self._n_resets += 1
        self.initialize(self.datapoint)
        if self.interactive_plot:
            if self.posterior_ax is not None:
                for k, ax in self.posterior_ax.items():
                    clear(ax)

        self.clk.restart()


    def createHdf(self, parent, add_axis=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """

        assert self.datapoint is not None, ValueError("Inference needs a datapoint before creating HDF5 files.")

        if add_axis is not None:
            if not isinstance(add_axis, (int, int32, int64)):
                add_axis = size(add_axis)

        self.datapoint.createHdf(parent, 'data', add_axis=add_axis, fillvalue=nan)

        # Initialize and write the attributes that won't change
        parent.create_dataset('update_plot_every', data=self.update_plot_every)
        parent.create_dataset('interactive_plot', data=self.interactive_plot)
        parent.create_dataset('reciprocate_parameter', data=self.reciprocate_parameter)

        if not self.limits is None:
            parent.create_dataset('limits', data=self.limits)

        parent.create_dataset('n_markov_chains', data=self.n_markov_chains)
        parent.create_dataset('nsystems', data=self.datapoint.nSystems)

        # Initialize the attributes that will be written later
        s = add_axis
        if add_axis is None:
            s = 1

        parent.create_dataset('iteration', shape=(s), dtype=self.iteration.dtype, fillvalue=nan)
        parent.create_dataset('burned_in_iteration', shape=(s), dtype=self.burned_in_iteration.dtype, fillvalue=nan)
        parent.create_dataset('best_iteration', shape=(s), dtype=self.best_iteration.dtype, fillvalue=nan)
        parent.create_dataset('burned_in', shape=(s), dtype=type(self.burned_in), fillvalue=0)
        parent.create_dataset('multiplier',  shape=(s), dtype=self.multiplier.dtype, fillvalue=nan)
        parent.create_dataset('invtime',  shape=(s), dtype=float, fillvalue=nan)
        parent.create_dataset('savetime',  shape=(s), dtype=float, fillvalue=nan)

        self.acceptance_v.createHdf(parent, 'acceptance_rate', add_axis=add_axis, fillvalue=nan)
        self.data_misfit_v.createHdf(parent, 'phids', add_axis=add_axis, fillvalue=nan)
        self.halfspace.createHdf(parent, 'halfspace', add_axis=add_axis, fillvalue=nan)

        # Since the 1D models change size adaptively during the inversion, we need to pad the HDF creation to the maximum allowable number of layers.
        tmp = self.model.pad(self.model.mesh.max_cells)
        tmp.createHdf(parent, 'model', add_axis=add_axis, fillvalue=nan)

        return parent

    def writeHdf(self, parent, index=None):
        """ Given a HDF file initialized as line results, write the contents of results to the appropriate arrays """

        # Get the point index
        if index is None:
            fiducials = DataArray.fromHdf(parent['data/fiducial'])
            index = fiducials.searchsorted(self.datapoint.fiducial)

        # Add the iteration number
        parent['iteration'][index] = self.iteration

        # Add the burn in iteration
        parent['burned_in_iteration'][index] = self.burned_in_iteration

        # Add the burn in iteration
        parent['best_iteration'][index] = self.best_iteration

        # Add the burned in logical
        parent['burned_in'][index] = self.burned_in

        # Add the multiplierx
        parent['multiplier'][index] = self.multiplier

        # Add the acceptance rate
        self.acceptance_v.writeHdf(parent, 'acceptance_rate', index=index)

        # Add the data misfit
        self.data_misfit_v.writeHdf(parent, 'phids', index=index)

        # Write the data posteriors
        self.datapoint.writeHdf(parent,'data',  index=index)
        # Write the highest posterior data
        self.best_datapoint.writeHdf(parent,'data', withPosterior=False, index=index)

        self.halfspace.writeHdf(parent, 'halfspace', index=index)

        # Write the model posteriors
        self.model.writeHdf(parent,'model', index=index)

        # Write the highest posterior data
        self.best_model.writeHdf(parent,'model', withPosterior=False, index=index)


    def read_fromH5Obj(self, h5obj, fName, grpName, system_file_path = ''):
        """ Reads a data points results from HDF5 file """
        grp = h5obj.get(grpName)
        assert not grp is None, "ID "+str(grpName) + " does not exist in file " + fName
        self.fromHdf(grp, system_file_path)


    @classmethod
    def fromHdf(cls, hdfFile, prng, index=None, fiducial=None):

        iNone = index is None
        fNone = fiducial is None

        assert not (iNone and fNone) ^ (not iNone and not fNone), Exception("Must specify either an index OR a fiducial.")

        if not fNone:
            fiducials = DataArray.fromHdf(hdfFile['data/fiducial'])
            index = fiducials.searchsorted(fiducial)

        self = cls(
                interactive_plot = True,
                multiplier = hdfRead.readKeyFromFile(hdfFile, '', '/', 'multiplier', index=index),
                n_markov_chains = array(hdfFile.get('n_markov_chains', 100000)),
                parameter_limits = None if not 'limits' in hdfFile else hdfFile.get('limits'),
                reciprocate_parameters = array(hdfFile.get('reciprocate_parameter', False)),
                save_hdf5 = False,
                save_png = False,
                update_plot_every = array(hdfFile.get('update_plot_every', 5000)),
                dont_initialize = True,
                prng=prng)
        self.datapoint = hdfRead.readKeyFromFile(hdfFile, '', '/', 'data', index=index)

        s = s_[index, :]

        self.nSystems = array(hdfFile.get('nsystems'))

        key = 'iteration' if 'iteration' in hdfFile else 'i'
        self.iteration = hdfRead.readKeyFromFile(hdfFile, '', '/', key, index=index)

        key = 'burned_in_iteration' if 'burned_in_iteration' in hdfFile else 'iburn'
        self.burned_in_iteration = hdfRead.readKeyFromFile(hdfFile, '', '/', key, index=index)

        key = 'burned_in' if 'burned_in' in hdfFile else 'burnedin'
        self.burned_in = hdfRead.readKeyFromFile(hdfFile, '', '/', key, index=index)

        key = 'acceptance_rate' if 'acceptance_rate' in hdfFile else 'rate'
        self.acceptance_rate = hdfRead.readKeyFromFile(hdfFile, '', '/', key, index=s)

        # Compute the x axis for acceptance since its every X iterations.
        n = 2 * int32(self.n_markov_chains / self.update_plot_every)
        self.acceptance_x = DataArray(arange(1, n + 1) * self.update_plot_every, name='Iteration #')

        self.best_datapoint = self.datapoint

        self.data_misfit_v = StatArray(hdfRead.readKeyFromFile(hdfFile, '', '/', 'phids', index=s))
        self.data_misfit_v.prior = Distribution('chi2', df=sum(self.datapoint.active), prng=self.prng)
        edges = DataArray(linspace(1, 2*sum(self.datapoint.active)))
        self.data_misfit_v.posterior = Histogram(mesh = RectilinearMesh1D(edges=edges))


        self.model = hdfRead.readKeyFromFile(hdfFile, '', '/', 'model', index=index)

        self.best_model = self.model

        self.halfspace = hdfRead.readKeyFromFile(hdfFile, '', '/', 'halfspace', index=index)

        self.Hitmap = self.model.values.posterior

        self.invTime = array(hdfFile.get('invtime')[index])
        self.saveTime = array(hdfFile.get('savetime')[index])

        # Initialize a list of iteration number
        self.iRange = DataArray(arange(2 * self.n_markov_chains), name="Iteration #", dtype=int64)

        self.verbose = False


        return self
