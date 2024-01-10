""" @_userParameters_Class
Parent handler for user defined parameters. Checks that the input user parameters match for a given data point.
This provides a bit more robust checking when the user is new to the codes, and must modify an input parameter class file.
"""
from os.path import join
from numpy import float64, asarray

from ..classes.core.myObject import myObject
from copy import deepcopy
from ..classes.core import StatArray
from ..classes.data.dataset.FdemData import FdemData
from ..classes.data.dataset.TdemData import TdemData
from ..classes.data.dataset.TempestData import TempestData
from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.data.datapoint.Tempest_datapoint import Tempest_datapoint

import numpy as np
from ..base.utilities import isInt


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
        kwargs['gradient_standard_deviation'] = 1.5 if kwargs.get('gradient_standard_deviation') is None else kwargs.get('gradient_standard_deviation')
        kwargs['multiplier'] = float64(1.0) if kwargs.get('multiplier') is None else kwargs.get('multiplier')
        kwargs['stochastic_newton'] = not kwargs.get('ignore_likelihood', False)
        kwargs['factor'] = float64(10.0) if kwargs.get('factor') is None else kwargs.get('factor')
        kwargs['covariance_scaling'] = float64(1.0) if kwargs.get('covariance_scaling') is None else kwargs.get('covariance_scaling')
        # kwargs['parameter_limits'] = np.r_[1e-10, 1e10] if kwargs.get('parameter_limits') is None else kwargs.get('parameter_limits')

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
