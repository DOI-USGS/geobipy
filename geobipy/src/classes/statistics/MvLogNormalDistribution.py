""" @MvNormalDistribution
Module defining a multivariate normal distribution with statistical procedures
"""
#from copy import deepcopy
from numpy import empty, exp, float64, linspace, squeeze
from numpy import log as nplog

from ...base  import utilities
from .baseDistribution import baseDistribution
from ..core.DataArray import DataArray
from .MvNormalDistribution import MvNormal
from scipy.stats import multivariate_normal

class MvLogNormal(MvNormal):
    """Class extension to geobipy.baseDistribution

    Handles a multivariate lognormal distribution.  Uses Scipy to evaluate probabilities,
    but Numpy to generate random samples since scipy is slow.

    MvLogNormal(mean, variance, ndim, linearSpace, prng)

    Parameters
    ----------
    mean : scalar or array_like
        Mean(s) for each dimension
    variance : scalar or array_like
        Variance of the logged values for each dimension
    ndim : int, optional
        The number of dimensions in the multivariate normal.
        Only used if mean and variance are scalars that are constant for all dimensions
    linearSpace : bool, optional
        If False, any input and output is in log space.
        If True, input and output is in linear space.
            Inputs are internally logged, and the exponential of any output is returned
    prng : numpy.random.RandomState, optional
        A random state to generate random numbers. Required for parallel instantiation.

    Returns
    -------
    out : MvLogNormal
        Multivariate lognormal distribution.

    """
    def __init__(self, mean, variance, ndim=None, linearSpace=False, prng=None):
        """ Initialize a multivariate lognormal distribution. """
        if linearSpace:
            mean = nplog(mean)
        self.linearSpace = linearSpace
        super().__init__(mean, variance, ndim, prng=prng)

    @property
    def mean(self):
        return exp(self._mean) if self.linearSpace else self._mean

    @mean.setter
    def mean(self, values):
        self._mean[:] = nplog(values) if self.linearSpace else values

    def __deepcopy__(self, memo={}):
        """ Define a deepcopy routine """
        if self._constant:
            return MvLogNormal(mean=self.mean[0], variance=self.variance[0, 0], ndim=self.ndim, linearSpace=self.linearSpace, prng=self.prng)
        else:
            return MvLogNormal(mean=self.mean, variance=self.variance, linearSpace=self.linearSpace, prng=self.prng)

    def derivative(self, x, order):
        if self.linearSpace:
            x = nplog(x)
        return super().derivative(x, order)

    def deviation(self, x):
        if self.linearSpace:
            x = nplog(x)
        return super().deviation(x)

    def rng(self, size = 1):
        return exp(super().rng(size)) if self.linearSpace else super().rng(size)

    def probability(self, x, log, axis=None, **kwargs):
        if self.linearSpace:
            x = nplog(x)

        return super().probability(x=x, log=log, axis=axis)

    def mahalanobis(self, x):
        if self.linearSpace:
            x = nplog(x)
        return super().mahalanobis(x=x)

    def bins(self, nBins=99, nStd=4.0, axis=None, relative=False):
        """Discretizes a range given the mean and variance of the distribution

        Parameters
        ----------
        nBins : int, optional
            Number of bins to return.
        nStd : float, optional
            The bin edges = mean +- nStd * variance.
        dim : int, optional
            Get the bins of this dimension, if None, returns bins for all dimensions.

        Returns
        -------
        bins : geobipy.StatArray
            The bin edges.

        """
        bins = super().bins(nBins, nStd, axis, relative)
        return exp(bins) if self.linearSpace else bins

    @property
    def summary(self):
        msg =  "{}\n".format(type(self).__name__)
        if self.linearSpace:
            msg += '    Mean:log{}\n'.format(self.mean)
        else:
            msg += '    Mean:{}\n'.format(self.mean)
        msg += 'Variance:{}\n'.format(self._variance)
        return msg