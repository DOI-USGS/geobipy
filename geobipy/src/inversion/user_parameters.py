""" @_userParameters_Class
Parent handler for user defined parameters. Checks that the input user parameters match for a given data point.
This provides a bit more robust checking when the user is new to the codes, and must modify an input parameter class file.
"""
from os.path import join
from numpy import float64, asarray

from ..classes.core.myObject import myObject
from copy import deepcopy
from ..classes.statistics import StatArray
from ..classes.data.dataset.FdemData import FdemData
from ..classes.data.dataset.TdemData import TdemData
from ..classes.data.dataset.TempestData import TempestData
from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.data.datapoint.Tempest_datapoint import Tempest_datapoint

import numpy as np
from ..base.utilities import isInt
from pprint import pprint


global_dict = {'FdemData': FdemData,
               'TdemData': TdemData,
               'TempestData':TempestData,
               'FdemDataPoint':FdemDataPoint,
               'TdemDataPoint':TdemDataPoint,
               'TempestDataPoint':Tempest_datapoint}


class user_parameters(dict):
    """ Handler class to user defined parameters. Allows us to check a users input parameters in the backend """

    def __init__(self, **kwargs):

        missing = [ x for x in self.required_keys if not x in kwargs ]
        if len(missing) > 0:
            raise ValueError("Missing {} from the user parameter file".format(missing))

        kwargs.pop('join', None)

        kwargs = self.optional_arguments(**kwargs)

        # kwargs['stochastic_newton'] = not kwargs.get('ignore_likelihood', False)

        kwargs['data_filename'] = join(kwargs['data_directory'], kwargs['data_filename'])
        if isinstance(kwargs['system_filename'], list):
            kwargs['system_filename'] = [join(kwargs['data_directory'], x) for x in kwargs['system_filename']]
        else:
            kwargs['system_filename'] = join(kwargs['data_directory'], kwargs['system_filename'])

        for key, value in kwargs.items():
            self[key] = value

        self._data_filename = [value] if isinstance(value, str) else value

    def __deepcopy__(self, memo={}):
        return deepcopy(self)

    def optional_arguments(self, **kwargs):
        kwargs = self.assign_default('parameter_standard_deviation', float64(2.39), **kwargs)
        kwargs = self.assign_default('gradient_standard_deviation', float64(1.5), **kwargs)
        kwargs = self.assign_default('multiplier', float64(1.0), **kwargs)
        kwargs = self.assign_default('factor', float64(10.0), **kwargs)
        kwargs = self.assign_default('covariance_scaling', float64(1.0), **kwargs)
        kwargs = self.assign_default('parameter_weight', float64(1.0), **kwargs)
        kwargs = self.assign_default('gradient_weight', float64(0.0), **kwargs)
        kwargs = self.assign_default('minimum_burn_in', float64(5000.0), **kwargs)
        kwargs = self.assign_default('parameter_limits', np.r_[1e-20, 1e20], **kwargs)
        kwargs = self.assign_default('stochastic_newton', True, **kwargs)

        if "number_of_depth_bins" in kwargs:
            kwargs["number_of_edge_bins"] = kwargs.pop("number_of_depth_bins")
        return kwargs

    def assign_default(self, key, value, **kwargs):
        kwargs[key] = value if kwargs.get(key) is None else kwargs.get(key)
        return kwargs

    @property
    def required_keys(self):
        return ('data_type',
                'data_filename',
                'system_filename',
                'n_markov_chains',
                'interactive_plot',
                'update_plot_every',
                'save_png',
                'save_hdf5',
                'solve_parameter',
                'solve_gradient',
                'maximum_number_of_layers',
                'minimum_depth',
                'maximum_depth',
                'probability_of_birth',
                'probability_of_death',
                'probability_of_perturb',
                'probability_of_no_change'
                )

    @classmethod
    def read(cls, filename, **kwargs):
        options = {}
        # Load user parameters
        with open(filename, 'r') as f:
            f = '\n'.join(f.readlines())
            exec(f, global_dict, options)

        for key, value in options.items():
            if isinstance(value, list):
                if not isinstance(value[0], str):
                    options[key] = asarray(value)

        for key, value in kwargs.items():
            if value is not None:
                options[key] = value

        return cls(**options)
