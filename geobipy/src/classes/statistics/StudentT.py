""" @NormalDistribution
Module defining a normal distribution with statistical procedures
"""
from copy import deepcopy
import numpy as np
from .baseDistribution import baseDistribution
from scipy.stats import t
from ...base import plotting as cP
from ..core.DataArray import DataArray

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
        self._mean = np.asarray(mean).copy()
        self._variance = np.asarray(variance).copy()
        self._degrees = np.asarray(degrees).copy()


    @property
    def ndim(self):
        return 1

    @property
    def multivariate(self):
        return True

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def degrees(self):
        return self._degrees

    def deepcopy(self):
        """Create a deepcopy

        Returns
        -------
        out
            Normal

        """
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


    def probability(self, x, log):
        """ For a realization x, compute the probability """
        if log:
            return DataArray(t.logpdf(x, self.degrees, loc = self._mean, scale = self.variance), "Probability Density")
        else:
            return DataArray(t.pdf(x, self.degrees, loc = self._mean, scale = self.variance), "Probability Density")