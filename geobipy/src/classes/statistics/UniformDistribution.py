""" @UniformDistribution
Module defining a uniform distribution with statistical procedures
"""
from copy import deepcopy
from .baseDistribution import baseDistribution
from ...base.HDF.hdfWrite import writeNumpy
from ...base import customPlots as cP
import numpy as np
from scipy.stats import uniform
from ..core import StatArray


class Uniform(baseDistribution):
    """ Class defining a uniform distribution """

    def __init__(self, min, max, log=False, prng=None):
        """ Initialize a uniform distribution
        xmin:  :Minimum value
        xmax:  :Maximum value
        """

        assert np.all(max > min), ValueError("Maximum must be > minimum")
        super().__init__(prng)
        # Minimum
        self._min = np.log(min) if log else deepcopy(min)
        # Maximum
        self._max = np.log(max) if log else deepcopy(max)
        self.log = log
        # Mean
        self._mean = 0.5 * (self._max + self._min)
        self._scale = self._max - self._min
        # Variance
        self._variance = (1.0 / 12.0) * self.scale**2.0
        

    @property
    def ndim(self):
        return np.size(self.min)

    @property
    def multivariate(self):
        return True if self.ndim > 1 else False

    @property
    def min(self):
        return np.exp(self._min) if self.log else self._min

    @property
    def max(self):
        return np.exp(self._max) if self.log else self._max

    @property
    def mean(self):
        return self._mean

    @property
    def scale(self):
        return self._scale

    @property
    def variance(self):
        return self._variance


    def deepcopy(self):
        """ Define a deepcopy routine """
        return Uniform(self.min, self.max, self.log, self.prng)


    def cdf(self, x, log=False):
        """ Get the value of the cumulative distribution function for a x """
        if self.log:
            x = np.log(x)
        if log:
            return uniform.logcdf(x, self._min, self.scale)
        else:
            return uniform.cdf(x, self._min, self.scale)


    def plotPDF(self, log=False, **kwargs):
        bins = self.bins()
        t = r"$\tilde{U}("+str(self.min)+","+str(self.max)+")$"
        cP.plot(bins, self.probability(bins, log=log), label=t, **kwargs)


    def probability(self, x, log):

        if self.log:
            x = np.log(x)

        if log:
            out = np.squeeze(uniform.logpdf(x, self._min, self.scale))
            return np.sum(out) if self.multivariate else out
        else:
            out = np.squeeze(uniform.pdf(x, self._min, self.scale))
            return np.prod(out) if self.multivariate else out


    def rng(self, size=1):
        values = self.prng.uniform(self._min, self._max, size=size)
        return np.exp(values) if self.log else values


    def summary(self, out=False):
        msg = 'Uniform Distribution: \n'
        msg += '  Min: :' + str(self.min) + '\n'
        msg += '  Max: :' + str(self.max) + '\n'
        if (out):
            return msg
        print(msg)


    def bins(self, nBins=100, dim=None):
        """Discretizes a range given the min and max of the distribution 
        
        Parameters
        ----------
        nBins : int, optional
            Number of bins to return.
        dim : int, optional
            Get the bins of this dimension, if None, returns bins for all dimensions.
        
        Returns
        -------
        bins : array_like
            The bin edges.

        """
        
        nD = self.ndim
        if (nD > 1):
            if dim is None:
                bins = np.empty([nD, nBins+1])
                for i in range(nD):
                    bins[i, :] = np.linspace(self._min[i], self._max[i], nBins+1)
                values = StatArray.StatArray(np.squeeze(bins))
            else:
                bins = np.empty(nBins+1)
                bins[:] = np.linspace(self._min[dim], self._max[dim], nBins+1)
                values = StatArray.StatArray(np.squeeze(bins))

        else:
            values = StatArray.StatArray(np.squeeze(np.linspace(self._min, self._max, nBins+1)))

        return np.exp(values) if self.log else values
