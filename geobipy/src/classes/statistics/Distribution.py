""" @Distribution_Class
Module describing statistical distributions
"""
#from ...base import Erro
# r as Err
from copy import deepcopy
from numpy.random import RandomState
from .baseDistribution import baseDistribution
from .CategoricalDistribution import Categorical
from .NormalDistribution import Normal
from .ChiSquaredDistribution import ChiSquared
from .LogNormalDistribution import LogNormal
from .MvNormalDistribution import MvNormal
from .MvLogNormalDistribution import MvLogNormal
from .UniformDistribution import Uniform
from .GammaDistribution import Gamma
from .OrderStatistics import Order
from .StudentT import StudentT


def Distribution(distributionType, *args, **kwargs):
    """Instantiate a statistical distribution

    Parameters
    ----------
    distributionType : str or subclass of baseDistribution
        If distributionType is str, choose between {Normal, MvNormal, Uniform, Gamma, Order, Categorical}
        if distributionType is subclass of baseDistribution, a copy is made

    Returns
    -------
    out : The distribution requested
        Subclass of baseDistribution

    Example
    -------
    >>> from geobipy import Distribution
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> D = Distribution('Normal', 0.0, 1.0)
    >>> x = np.linspace(-5.0,5.0,100)
    >>> y = D.probability(x)
    >>> plt.figure()
    >>> plt.plot(x,y)
    >>> plt.show()
    >>> # To create a Distribution using a specific pseudo random number generator
    >>> prng = np.random.RandomState()
    >>> D = Distribution('Normal', 0.0, 1.0, prng=prng)

    See Also
    --------
    geobipy.src.classes.statistics.NormalDistribution
    geobipy.src.classes.statistics.MvNormalDistribution
    geobipy.src.classes.statistics.UniformDistribution
    geobipy.src.classes.statistics.GammaDistribution
    geobipy.src.classes.statistics.OrderStatistics
    geobipy.src.classes.statistics.CategoricalDistribution

    """
    if (isinstance(distributionType, baseDistribution)):
        return deepcopy(distributionType)

    tName = distributionType.lower()
    if (tName == 'uniform'):
        return Uniform(*args, **kwargs)

    elif (tName == 'chi2'):
        return ChiSquared(*args, **kwargs)

    elif (tName == 'normal'):
        return Normal(*args, **kwargs)

    elif (tName == 'lognormal'):
        return LogNormal(*args, **kwargs)

    elif (tName == 'mvnormal'):
        return MvNormal(*args, **kwargs)

    elif (tName == 'mvlognormal'):
        return MvLogNormal(*args, **kwargs)

    elif (tName == 'studentt'):
        return StudentT(*args, **kwargs)

    elif (tName == 'gamma'):
        return Gamma(*args, **kwargs)

    elif (tName == 'poisson'):
        assert False, ('Poisson not implemented yet')

    elif (tName == 'laplace'):
        assert False, ('Laplace not implemented yet')

    elif (tName == 'order'):
        return Order(*args, **kwargs)

    elif (tName == 'categorical'):
        return Categorical(*args, **kwargs)

    else:
        assert False, Exception('Please choose an appropriate distribution [Uniform, Normal, MvNormal, Gamma, Order, Categorical]')