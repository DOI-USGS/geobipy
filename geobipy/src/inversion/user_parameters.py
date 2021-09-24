""" @_userParameters_Class
Parent handler for user defined parameters. Checks that the input user parameters match for a given data point.
This provides a bit more robust checking when the user is new to the codes, and must modify an input parameter class file.
"""
#from ..base import Error as Err
from ..classes.core.myObject import myObject
from ..classes.core import StatArray
from ..classes.data.dataset.FdemData import FdemData
from ..classes.data.dataset.TdemData import TdemData
from ..classes.data.dataset.TempestData import TempestData
from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.data.datapoint.Tempest_datapoint import Tempest_datapoint
from ..classes.statistics.Hitmap2D import Hitmap2D
import numpy as np
from ..base.utilities import isInt


global_dict = {'FdemData': FdemData,
               'TdemData': TdemData,
               'TempestData':TempestData,
               'FdemDataPoint':FdemDataPoint,
               'TdemDataPoint':TdemDataPoint,
               'TempestDataPoint':Tempest_datapoint}


class user_parameters(myObject):
    """ Handler class to user defined parameters. Allows us to check a users input parameters in the backend """

    def __init__(self, **kwargs):

        self.stochastic_newton = True
        self.data_type = kwargs.get('data_type')
        self.data_filename = kwargs.get('data_filename')
        self.system_filename = kwargs.get('system_filename')

        self.n_markov_chains = kwargs.get('n_markov_chains')
        self.interactive_plot = kwargs.get('interactive_plot')
        self.update_plot_every = kwargs.get('update_plot_every')
        self.save_png = kwargs.get('save_png')
        self.save_hdf5 = kwargs.get('save_hdf5')


        self.solve_parameter = kwargs.get('solve_parameter')
        self.solve_gradient = kwargs.get('solve_gradient')
        self.solve_relative_error = kwargs.get('solve_relative_error')
        self.solve_additive_error = kwargs.get('solve_additive_error')
        self.solve_height = kwargs.get('solve_height')
        self.solve_calibration = kwargs.get('solve_calibration')


        self.maximum_number_of_layers = kwargs.get('maximum_number_of_layers')
        self.minimum_depth = kwargs.get('minimum_depth')
        self.maximum_depth = kwargs.get('maximum_depth')
        self.minimum_thickness = kwargs.get('minimum_thickness')

        if self.solve_relative_error:
            self.minimum_relative_error = kwargs.get('minimum_relative_error')
            self.maximum_relative_error = kwargs.get('maximum_relative_error')
            self.relative_error_proposal_variance = kwargs.get('relative_error_proposal_variance')
        self.initial_relative_error = kwargs.get('initial_relative_error')

        if self.solve_additive_error:
            self.minimum_additive_error = kwargs.get('minimum_additive_error')
            self.maximum_additive_error = kwargs.get('maximum_additive_error')
            self.additive_error_proposal_variance = kwargs.get('additive_error_proposal_variance')
        self.initial_additive_error = kwargs.get('initial_additive_error')

        if self.solve_height:
            self.maximum_height_change = kwargs.get('maximum_height_change')
            self.height_proposal_variance = kwargs.get('height_proposal_variance')


        self.probability_of_birth = kwargs.get('probability_of_birth')
        self.probability_of_death = kwargs.get('probability_of_death')
        self.probability_of_perturb = kwargs.get('probability_of_perturb')
        self.probability_of_no_change = kwargs.get('probability_of_no_change')


        self.factor = kwargs.get('factor', np.float64(10.0))
        self.gradient_standard_deviation = kwargs.get('gradient_standard_deviation')
        self.covariance_scaling = kwargs.get('covariance_scaling')
        self.multiplier = kwargs.get('multiplier')
        self.clip_ratio = kwargs.get('clip_ratio')
        self.ignore_likelihood = kwargs.get('ignore_likelihood')
        self.parameter_limits = kwargs.get('parameter_limits')
        self.reciprocate_parameters = kwargs.get('reciprocate_parameters')
        self.verbose = kwargs.get('verbose')



        # self.initialRelativeError = StatArray.StatArray(Datapoint.nSystems, 'Relative Error') + self.initialRelativeError
        # self.minimumRelativeError = StatArray.StatArray(Datapoint.nSystems, 'Minimum Relative Error') + self.minimumRelativeError
        # self.maximumRelativeError = StatArray.StatArray(Datapoint.nSystems, 'Maximum Relative Error') + self.maximumRelativeError


        # self.initialAdditiveError = StatArray.StatArray(Datapoint.nSystems, 'Additive Error') + self.initialAdditiveError
        # self.minimumAdditiveError = StatArray.StatArray(Datapoint.nSystems, 'Minimum Additive Error') + self.minimumAdditiveError
        # self.maximumAdditiveError = StatArray.StatArray(Datapoint.nSystems, 'Maximum Additive Error') + self.maximumAdditiveError

        # self.check(Datapoint)

    def __deepcopy__(self, memo={}):
        return None

    @property
    def data_type(self):
        return self._data_type
    @data_type.setter
    def data_type(self, value):
        self._data_type = value


    @property
    def data_filename(self):
        return self._data_filename
    @data_filename.setter
    def data_filename(self, value):
        self._data_filename = [value] if isinstance(value, str) else value

    @property
    def system_filename(self):
        return self._system_filename
    @system_filename.setter
    def system_filename(self, value):
        self._system_filename = [value] if isinstance(value , str) else value


    @property
    def n_markov_chains(self):
        return self._n_markov_chains
    @n_markov_chains.setter
    def n_markov_chains(self, value):
        self._n_markov_chains = np.int64(value)

    @property
    def interactive_plot(self):
        return self._interactive_plot
    @interactive_plot.setter
    def interactive_plot(self, value=False):
        assert isinstance(value, bool), ValueError("interactive_plot must have type bool")
        self._interactive_plot = value

    @property
    def update_plot_every(self):
        return self._update_plot_every
    @update_plot_every.setter
    def update_plot_every(self, value=5000):
        self._update_plot_every = np.int32(value)

    @property
    def save_png(self):
        return self._save_png
    @save_png.setter
    def save_png(self, value=False):
        assert isinstance(value, bool), ValueError("save_png must have type bool")
        self._save_png = value

    @property
    def save_hdf5(self):
        return self._save_hdf5
    @save_hdf5.setter
    def save_hdf5(self, value=True):
        assert isinstance(value, bool), ValueError("save_hdf5 must have type bool")
        self._save_hdf5 = value

    @property
    def solve_parameter(self):
        return self._solve_parameter
    @solve_parameter.setter
    def solve_parameter(self, value=False):
        assert isinstance(value, bool), ValueError("solve_parameter must have type bool")
        self._solve_parameter = value

    @property
    def solve_gradient(self):
        return self._solve_gradient
    @solve_gradient.setter
    def solve_gradient(self, value=True):
        assert isinstance(value, bool), ValueError("solve_gradient must have type bool")
        self._solve_gradient = value

    @property
    def solve_relative_error(self):
        return self._solve_relative_error
    @solve_relative_error.setter
    def solve_relative_error(self, value=False):
        assert isinstance(value, bool), ValueError("solve_relative_error must have type bool")
        self._solve_relative_error = value

    @property
    def solve_additive_error(self):
        return self._solve_additive_error
    @solve_additive_error.setter
    def solve_additive_error(self, value=False):
        assert isinstance(value, bool), ValueError("solve_additive_error must have type bool")
        self._solve_additive_error = value

    @property
    def solve_height(self):
        return self._solve_height
    @solve_height.setter
    def solve_height(self, value=False):
        assert isinstance(value, bool), ValueError("solve_height must have type bool")
        self._solve_height = value


    @property
    def maximum_number_of_layers(self):
        return self._maximum_number_of_layers
    @maximum_number_of_layers.setter
    def maximum_number_of_layers(self, value=30):
        self._maximum_number_of_layers = np.int32(value)

    @property
    def minimum_depth(self):
        return self._minimum_depth
    @minimum_depth.setter
    def minimum_depth(self, value):
        self._minimum_depth = np.float64(value)

    @property
    def maximum_depth(self):
        return self._maximum_depth
    @maximum_depth.setter
    def maximum_depth(self, value):
        self._maximum_depth = np.float64(value)

    @property
    def minimum_thickness(self):
        return self._minimum_thickness
    @minimum_thickness.setter
    def minimum_thickness(self, value):
        if value is not None:
            value = np.float64(value)
        self._minimum_thickness = value


    @property
    def initial_relative_error(self):
        return self._initial_relative_error
    @initial_relative_error.setter
    def initial_relative_error(self, value):
        if not value is None:
            value = np.asarray(value) if np.size(value) > 1 else np.float64(value)
        self._initial_relative_error = value

    @property
    def minimum_relative_error(self):
        return self._minimum_relative_error
    @minimum_relative_error.setter
    def minimum_relative_error(self, value):
        if not value is None:
            value = np.asarray(value) if np.size(value) > 1 else np.float64(value)
        self._minimum_relative_error = value

    @property
    def maximum_relative_error(self):
        return self._maximum_relative_error
    @maximum_relative_error.setter
    def maximum_relative_error(self, value):
        if not value is None:
            value = np.asarray(value) if np.size(value) > 1 else np.float64(value)
        self._maximum_relative_error = value


    @property
    def initial_additive_error(self):
        return self._initial_additive_error
    @initial_additive_error.setter
    def initial_additive_error(self, value):
        if not value is None:
            value = np.asarray(value) if np.size(value) > 1 else np.float64(value)
        self._initial_additive_error = value

    @property
    def minimum_additive_error(self):
        return self._minimum_additive_error
    @minimum_additive_error.setter
    def minimum_additive_error(self, value):
        if not value is None:
            value = np.asarray(value) if np.size(value) > 1 else np.float64(value)
        self._minimum_additive_error = value

    @property
    def maximum_additive_error(self):
        return self._maximum_additive_error
    @maximum_additive_error.setter
    def maximum_additive_error(self, value):
        if not value is None:
            if np.size(value) > 1:
                value = np.asarray(value) if np.size(value) > 1 else np.float64(value)
        self._maximum_additive_error = value


    @property
    def maximum_height_change(self):
        return self._maximum_height_change
    @maximum_height_change.setter
    def maximum_height_change(self, value):
        if not value is None:
            value = np.float64(value)
        self._maximum_height_change = value


    @property
    def relative_error_proposal_variance(self):
        return self._relative_error_proposal_variance
    @relative_error_proposal_variance.setter
    def relative_error_proposal_variance(self, value):
        if not value is None:
            value = np.asarray(value) if np.size(value) > 1 else np.float64(value)
        self._relative_error_proposal_variance = np.float64(value)

    @property
    def additive_error_proposal_variance(self):
        return self._additive_error_proposal_variance
    @additive_error_proposal_variance.setter
    def additive_error_proposal_variance(self, value):
        if not value is None:
            value = np.asarray(value) if np.size(value) > 1 else np.float64(value)
        self._additive_error_proposal_variance = np.float64(value)

    @property
    def height_proposal_variance(self):
        return self._height_proposal_variance
    @height_proposal_variance.setter
    def height_proposal_variance(self, value):
        self._height_proposal_variance = np.float64(value)


    @property
    def probability_of_birth(self):
        return self._probability_of_birth
    @probability_of_birth.setter
    def probability_of_birth(self, value):
        self._probability_of_birth = np.float64(value)

    @property
    def probability_of_death(self):
        return self._probability_of_death
    @probability_of_death.setter
    def probability_of_death(self, value):
        self._probability_of_death = np.float64(value)

    @property
    def probability_of_perturb(self):
        return self._probability_of_perturb
    @probability_of_perturb.setter
    def probability_of_perturb(self, value):
        self._probability_of_perturb = np.float64(value)

    @property
    def probability_of_no_change(self):
        return self._probability_of_no_change
    @probability_of_no_change.setter
    def probability_of_no_change(self, value):
        self._probability_of_no_change = np.float64(value)

    @property
    def factor(self):
        return self._factor
    @factor.setter
    def factor(self, value=np.float64(10.0)):
        value = np.float64(10.0) if value is None else value
        self._factor = value

    @property
    def gradient_standard_deviation(self):
        return self._gradient_standard_deviation
    @gradient_standard_deviation.setter
    def gradient_standard_deviation(self, value=np.float64(1.5)):
        value = np.float64(1.5) if value is None else value
        self._gradient_standard_deviation = value

    @property
    def covariance_scaling(self):
        return self._covariance_scaling
    @covariance_scaling.setter
    def covariance_scaling(self, value=np.float64(1.65)):
        value = np.float64(1.65) if value is None else value
        self._covariance_scaling = value

    @property
    def multiplier(self):
        return self._multiplier
    @multiplier.setter
    def multiplier(self, value=np.float64(1.02)):
        value = np.float64(1.02) if value is None else value
        self._multiplier = value

    @property
    def clip_ratio(self):
        return self._clip_ratio
    @clip_ratio.setter
    def clip_ratio(self, value=np.float64(0.5)):
        value = np.float64(0.5) if value is None else value
        self._clip_ratio = value

    @property
    def ignore_likelihood(self):
        return self._ignore_likelihood
    @ignore_likelihood.setter
    def ignore_likelihood(self, value=False):
        assert isinstance(value, bool), ValueError("ignore_likelihood must have type bool")
        self._ignore_likelihood = value
        self.stochasticNewton = ~self.ignore_likelihood

    @property
    def parameter_limits(self):
        return self._parameter_limits
    @parameter_limits.setter
    def parameter_limits(self, value=None):
        if value is not None:
            assert np.size(value) == 2, ValueError("parameter_limits must have 2 entries")
            values = np.sort(values)
        self._parameter_limits = value

    @property
    def reciprocate_parameters(self):
        return self._reciprocate_parameters
    @reciprocate_parameters.setter
    def reciprocate_parameters(self, value=False):
        value = False if value is None else value
        self._reciprocate_parameters = value

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, value=False):
        value = False if value is None else value
        self._verbose = value


    @classmethod
    def read(cls, filename):
        options = {}
        # Load user parameters
        with open(filename, 'r') as f:
            f = '\n'.join(f.readlines())
            exec(f, global_dict, options)

        return cls(**options)


    def check(self, DataPoint):
        """ Check that the specified input parameters match the type of data point"""

        assert isinstance(DataPoint, (FdemDataPoint, TdemDataPoint)), TypeError('Invalid DataPoint type used')

        # Check the number of Markov chains
        self.nMarkovChains = np.int32(self.nMarkovChains)
        assert isInt(self.nMarkovChains), TypeError('nMC must be a numpy integer')
        assert self.nMarkovChains >= 1000, ValueError('Number of Markov Chain iterations nMC must be >= 1000')

        # # Check the reference hitmap if given
        # if (not self.referenceHitmap is None):
        #     assert isinstance(self.referenceHitmap, Hitmap2D), TypeError('referenceHitmap must be of type geobipy.Hitmap2D')

        # # Check the relative Error
        # assert self.initialRelativeError.size == DataPoint.nSystems, ValueError('Initial relative error must have size {}'.format(DataPoint.nSystems))

        # # Check the minimum relative error
        # assert self.minimumRelativeError.size == DataPoint.nSystems, ValueError('Minimum relative error must be size {}'.format(DataPoint.nSystems))

        # # Check the maximum relative error
        # assert self.maximumRelativeError.size == DataPoint.nSystems, ValueError('Maximum Relative error must be size {}'.format(DataPoint.nSystems))

        # Check the error floor
        check = (DataPoint.nSystems, DataPoint.nChannels)
        assert (self.initialAdditiveError.size in check), ValueError('Initial additive error size must equal one of these {}'.format(check))

        if self.solveAdditiveError:
            # Check the minimum relative error
            assert self.minimumAdditiveError.size == DataPoint.nSystems, ValueError('Minimum additive error must be size {}'.format(DataPoint.nSystems))

            # Check the maximum relative error
            assert self.maximumAdditiveError.size == DataPoint.nSystems, ValueError('Maximum additive error must be size {}'.format(DataPoint.nSystems))



        # Check the relative error proposal variance
        assert self.relativeErrorProposalVariance.size == DataPoint.nSystems, ValueError('Proposal additive error variance must be size {}'.format(DataPoint.nSystems))

        # Check the relative error proposal variance
        assert self.additiveErrorProposalVariance.size == DataPoint.nSystems, ValueError('Proposal additive error variance must be size {}'.format(DataPoint.nSystems))

        # Check the calibration proposal variance if they are used
        # if (self.solveCalibration):
        #     assert self.propCal.shape == [N1, nCalibration], ValueError('Proposal Calibration variance must have shape {}'.forma([N1, nCalibration]))

        # Check the covariance scaling parameter
        assert isinstance(self.parameterCovarianceScaling, float), TypeError('Covariance scaling must be a float (preferably np.float64)')

        # Check the data misfit multiplier factor
        assert isinstance(self.multiplier, float), TypeError('Data misfit multiplier must be a float (preferably np.float64)')

        # Checking the Probability Wheel
        assert isinstance(self.pBirth, float), TypeError('Probability of birth must be a float (preferably np.float64)')
        assert isinstance(self.pDeath, float), TypeError('Probability of death must be a float (preferably np.float64)')
        assert isinstance(self.pPerturb, float), TypeError('Probability of perturb must be a float (preferably np.float64)')
        assert isinstance(self.pNochange, float), TypeError('Probability of no change must be a float (preferably np.float64)')

        if self.ignoreLikelihood:
            self.stochasticNewton = False

