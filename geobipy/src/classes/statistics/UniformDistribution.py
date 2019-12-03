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

    def __init__(self, min, max, prng=None):
        """ Initialize a uniform distribution
        xmin:  :Minimum value
        xmax:  :Maximum value
        """

        assert max > min, ValueError("Maximum must be > minimum")
        super().__init__(prng)
        # Minimum
        self._min = deepcopy(min)
        # Maximum
        self._max = deepcopy(max)
        # Mean
        self._mean = 0.5 * (max + min)
        self._scale = max - min
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
        return self._min

    @property
    def max(self):
        return self._max

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
        return Uniform(self.min, self.max, self.prng)


    def getPdf(self, x=0):
        """ get the PDF, for a uniform distribution this does not need a procedure, however other distributions might, and we will need a function """
        return np.sum(self.pdf)


    def cdf(self, x, log=False):
        """ Get the value of the cumulative distribution function for a x """
        if log:
            return uniform.logcdf(x, self.min, self.scale)
        else:
            return uniform.cdf(x, self.min, self.scale)


    def plotPDF(self, **kwargs):
        bins = self.bins()
        t = r"$\tilde{U}("+str(self.min)+","+str(self.max)+")$"
        cP.plot(bins, np.repeat(self.pdf, np.size(bins)), label=t, **kwargs)


    def probability(self, x, log):
        if log:
            return np.squeeze(uniform.logpdf(x, self.min, self.scale))
        else:
            return np.squeeze(uniform.pdf(x, self.min, self.scale))


    def rng(self, size=1):
        return self.prng.uniform(self.min, self.max, size=size)


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
                    bins[i, :] = np.linspace(self.min[i], self.max[i], nBins+1)
                return StatArray.StatArray(np.squeeze(bins))
            else:
                bins = np.empty(nBins+1)
                bins[:] = np.linspace(self.min[dim], self.max[dim], nBins+1)
                return StatArray.StatArray(np.squeeze(bins))

        return StatArray.StatArray(np.squeeze(np.linspace(self.min, self.max, nBins+1)))
