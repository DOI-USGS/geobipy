""" @UniformDistribution
Module defining a uniform distribution with statistical procedures
"""
from copy import deepcopy
from .baseDistribution import baseDistribution
from ...base import plotting as cP
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

        self.log = log

        self.min = min
        self.max = max

        # Mean
        self._mean = 0.5 * (self._max + self._min)
        # Variance
        self._variance = (1.0 / 12.0) * self.scale**2.0

    @property
    def addressof(self):
        msg =  "{} {}\n".format(type(self).__name__, hex(id(self)))
        msg += 'Min:{}\n'.format(hex(id(self._min)))
        msg += 'Max:{}\n'.format(hex(id(self._max)))
        return msg

    @property
    def ndim(self):
        return np.size(self.min)

    @property
    def multivariate(self):
        return True if self.ndim > 1 else False

    @property
    def min(self):
        return np.exp(self._min) if self.log else self._min

    @min.setter
    def min(self, values):
        values = np.asarray(values)
        self._min = np.log(values) if self.log else deepcopy(values)

    @property
    def max(self):
        return np.exp(self._max) if self.log else self._max

    @max.setter
    def max(self, values):
        values = np.asarray(values)
        self._max = np.log(values) if self.log else deepcopy(values)

    @property
    def mean(self):
        return self._mean

    @property
    def scale(self):
        return self._max - self._min

    @property
    def variance(self):
        return self._variance


    def __deepcopy__(self, memo={}):
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


    def plot_pdf(self, log=False, **kwargs):
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


    @property
    def summary(self):
        msg = 'Uniform Distribution: \n'
        msg += '  Min: :' + str(self.min) + '\n'
        msg += '  Max: :' + str(self.max) + '\n'
        return msg


    def bins(self, nBins=99, dim=None):
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
