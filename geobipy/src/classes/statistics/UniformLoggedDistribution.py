""" @UniformDistribution
Module defining a uniform distribution with statistical procedures
"""
from copy import deepcopy
from .baseDistribution import baseDistribution
from .UniformDistribution import Uniform
from ...base.HDF.hdfWrite import writeNumpy
from ...base import customPlots as cP
import numpy as np


class UniformLog(Uniform):
    """ Class defining a uniform distribution """

    def __init__(self, min, max, prng=None):
        """ Initialize a uniform distribution
        xmin:  :Minimum value
        xmax:  :Maximum value
        """
        baseDistribution.__init__(self, prng)
        # Minimum
        self.min = deepcopy(min)
        # Maximum
        self.max = deepcopy(max)
        # Mean
        self.mean = 0.5 * (max + min)
        tmp = max - min
        # Variance
        self.variance = (1.0 / 12.0) * tmp**2.0

        assert np.any(max - min != 1.0), ValueError("Difference between max and min must != 1.0")

        # Set the pdf
        self.pdf = np.log(np.float64(1.0 / tmp))

    @property
    def ndim(self):
        return self.min.size

    def deepcopy(self):
        """ Define a deepcopy routine """
        return UniformLog(self.min, self.max, self.prng)


    def probability(self, x):
        if np.any(x < self.min):
            return -np.infty
        if np.any(x > self.max):
            return -np.infty
        return np.sum(self.pdf)


    def summary(self, out=False):
        msg = 'Uniform Logged Distribution: \n'
        msg += '  Min: :' + str(self.min) + '\n'
        msg += '  Max: :' + str(self.max) + '\n'
        if (out):
            return msg
        print(msg)


    def getBinEdges(self, nBins=100, dim=None):
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
                return bins
            else:
                bins = np.empty(nBins+1)
                bins[:] = np.linspace(self.min[dim], self.max[dim], nBins+1)
                return bins

        return np.linspace(self.min, self.max, nBins+1)
