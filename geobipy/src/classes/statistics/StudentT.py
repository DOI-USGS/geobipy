""" @NormalDistribution
Module defining a normal distribution with statistical procedures
"""
#from copy import deepcopy
import numpy as np
from ...base.logging import myLogger
from .baseDistribution import baseDistribution
from ...base.HDF.hdfWrite import writeNumpy
#from .MvNormalDistribution import MvNormal
from scipy.stats import t
from ...base import customPlots as cP
from ..core import StatArray

class StudentT(baseDistribution):
    """Univariate normal distribution

    Normal(mean, variance)

    Parameters
    ----------
    mean : numpy.float
        The mean of the distribution
    variance : numpy.float
        The variance of the distribution
    degrees : numpy.float
        The degrees of freedom

    """
    def __init__(self, mean, variance, degrees, prng=None):
        """Instantiate a Normal distribution """
        # assert np.size(mean) == 1, 'Univariate Normal mean must have size = 1'
        # assert np.size(variance) == 1, 'Univariate Normal variance must have size = 1'
        super().__init__(prng)
        self.mean = np.asarray(mean)
        self.variance = np.asarray(variance)
        self.degrees = np.asarray(degrees)

    
    @property
    def ndim(self):
        return 1

    @property
    def multivariate(self):
        return True


    def deepcopy(self):
        """Create a deepcopy

        Returns
        -------
        out
            Normal

        """
        # return deepcopy(self)
        return StudentT(self.mean, self.variance, self.degrees, self.prng)


    def rng(self, size=1):
        """Generate random numbers

        Parameters
        ----------
        N : int or sequence of ints
            Number of samples to generate

        Returns
        -------
        out
            numpy.ndarray

        """
        size = (size, self.mean.size)
        return np.squeeze(self.prng.standard_t(df=self.degrees, size=size, loc=self.mean, scale=self.variance))


    def probability(self, x):
        """ For a realization x, compute the probability """
        return StatArray.StatArray(t.pdf(x, df=self.degrees, loc = self.mean, scale = self.variance))