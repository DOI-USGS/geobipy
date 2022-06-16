""" @MvNormalDistribution
Module defining a multivariate normal distribution with statistical procedures
"""
#from copy import deepcopy
import numpy as np
from ...base  import utilities as cf
from .baseDistribution import baseDistribution
from ..core import StatArray
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
            mean = np.log(mean)
        super().__init__(mean, variance, ndim, prng=prng)
        self.linearSpace = linearSpace

    @property
    def mean(self):
        return np.exp(self._mean) if self.linearSpace else self._mean

    @mean.setter
    def mean(self, values):
        self._mean[:] = np.log(values) if self.linearSpace else values

    def __deepcopy__(self, memo={}):
        """ Define a deepcopy routine """
        if self._constant:
            return MvLogNormal(mean=self.mean[0], variance=self.variance[0, 0], ndim=self.ndim, linearSpace=self.linearSpace, prng=self.prng)
        else:
            return MvLogNormal(mean=self.mean, variance=self.variance, linearSpace=self.linearSpace, prng=self.prng)

    def derivative(self, x, order):

        assert order in [1, 2], ValueError("Order must be 1 or 2.")
        if order == 1:
            if self.linearSpace:
                x = np.log(x)
            return cf.Ax(self.precision, x-self._mean)

        elif order == 2:
            return self.precision

    def rng(self, size = 1):
        return np.exp(super().rng(size)) if self.linearSpace else super().rng(size)

    def probability(self, x, log, axis=None):
        if self.linearSpace:
            x = np.log(x)

        return super().probability(x=x, log=log, axis=axis)

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
        # if not self.linearSpace:
        #     return super().bins(nBins, nStd, axis)

        nStd = np.float64(nStd)
        nD = self.ndim
        if (nD > 1):
            if axis is None:
                bins = np.empty([nD, nBins+1])
                for i in range(nD):
                    tmp = np.squeeze(nStd * self.std[axis, axis])
                    t = np.linspace(-tmp, tmp, nBins+1)
                    if not relative:
                        t += self._mean[i]
                    bins[i, :] = t
            else:
                bins = np.empty(nBins+1)
                tmp = np.squeeze(nStd * self.std[axis, axis])
                t = np.linspace(-tmp, tmp, nBins+1)
                if not relative:
                    t += self._mean[axis]
                bins[:] = t

        else:
            tmp = nStd * self.std
            bins = np.squeeze(np.linspace(-tmp, tmp, nBins+1))
            if not relative:
                bins += self._mean

        return StatArray.StatArray(np.exp(bins)) if self.linearSpace else StatArray.StatArray(bins)
