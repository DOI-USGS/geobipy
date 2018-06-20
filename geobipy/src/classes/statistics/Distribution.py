""" @Distribution_Class
Module describing statistical distributions
"""
#from ...base import Error as Err
from numpy.random import RandomState
from .baseDistribution import baseDistribution
from .NormalDistribution import Normal
from .MvNormalDistribution import MvNormal
from .NormalLoggedDistribution import NormalLog
from .MvNormalLoggedDistribution import MvNormalLog
from .UniformDistribution import Uniform
from .GammaDistribution import Gamma
from .OrderStatistics import Order


def Distribution(distributionType, *args, **kwargs):
    """Instantiate a statistical distribution

    Parameters
    ----------
    distributionType : str or subclass of baseDistribution
        If distributionType is str, choose between {Normal, MvNormal, NormalLogged, MvNormalLogged, Uniform, Gamma, Order}
        if distributionType is subclass of baseDistribution, a copy is made
    *args : See the documentation for each distribution type to determine what *args could be
    *kwargs : See the documentation for each distribution type to determine what key words could be

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
    geobipy.src.classes.statistics.NormalLoggedDistribution
    geobipy.src.classes.statistics.MvNormalLoggedDistribution
    geobipy.src.classes.statistics.UniformDistribution
    geobipy.src.classes.statistics.GammaDistribution
    geobipy.src.classes.statistics.OrderStatistics

    """

    #if (not 'prng' in kwargs):
    #    kwargs['prng'] = RandomState()

    if (isinstance(distributionType, baseDistribution)):
        return distributionType.deepcopy()
    tName = distributionType.lower()
    if (tName == 'uniform'):
        assert (len(args)+len(kwargs) >= 2), 'Please enter a minimum and maximum for the uniform distribtion'
        out = Uniform(*args, **kwargs)
        return out
    elif (tName == 'normal' or tName == 'gaussian'):
        assert (len(args)+len(kwargs) >= 2), 'Please enter a mean and variance for the Normal distribtion'
        out = Normal(*args, **kwargs)
    elif (tName == 'mvnormal' or tName == 'mvgaussian'):
        assert (len(args)+len(kwargs) >= 2), 'Please enter a mean and variance for the MvNormal distribtion'
        out = MvNormal(*args, **kwargs)
    elif (tName == 'normallog' or tName == 'gaussianlog'):
        assert (len(args)+len(kwargs) >= 2), 'Please enter a mean and variance for the NormalLog distribtion'
        out = NormalLog(*args, **kwargs)
        return out
    elif (tName == 'mvnormallog' or tName == 'mvgaussianlog'):
        assert (len(args)+len(kwargs) >= 2), 'Please enter a mean and variance for the MvNormalLog distribtion'
        out = MvNormalLog(*args, **kwargs)
        return out
    elif (tName == 'gamma'):
        assert (len(args)+len(kwargs) >= 2), 'Please enter a scale > 0.0, shape > 0.0, and prng for the gamma distribtion'
        out = Gamma(*args, **kwargs)
    elif (tName == 'poisson'):
        assert False, ('Poisson not implemented yet')
    elif (tName == 'laplace'):
        assert False, ('Laplace not implemented yet')
    elif (tName == 'order'):
     #   tmp = kwargs.pop('prng')
        out = Order(*args, **kwargs)
        return out
    else:
        assert False, 'Please choose an appropriate distribution [Uniform, Normal, MvNormal, NormalLog, MvNormalLog, Gamma, Order]'

    return out

