""" @UniformDistribution
Module defining a uniform distribution with statistical procedures
"""
from copy import deepcopy

from numpy import asarray, empty, exp, linspace, prod, s_, size, squeeze, sum, hstack
from numpy import log as nplog
from numpy import all as npall

from .baseDistribution import baseDistribution
from ...base import plotting as cP

from scipy.stats import uniform
from ..core.DataArray import DataArray

class Uniform(baseDistribution):
    """ Class defining a uniform distribution """

    def __init__(self, min=0.0, max=1.0, log=False, prng=None):
        """ Initialize a uniform distribution
        xmin:  :Minimum value
        xmax:  :Maximum value
        """

        assert npall(max > min), ValueError("Maximum must be > minimum")
        super().__init__(prng)

        self.log = log

        self.min = min
        self.max = max

        # Mean
        self._mean = 0.5 * (self._max + self._min)
        # Variance
        self._variance = (1.0 / 12.0) * self.scale**2.0

    @property
    def address(self):
        return hstack([hex(id(self)), hex(id(self._min)), hex(id(self._max))])

    @property
    def addressof(self):
        msg =  "{} {}\n".format(type(self).__name__, hex(id(self)))
        msg += 'Min:{}\n'.format(hex(id(self._min)))
        msg += 'Max:{}\n'.format(hex(id(self._max)))
        return msg

    @property
    def ndim(self):
        return size(self.min)

    @property
    def multivariate(self):
        return True if self.ndim > 1 else False

    @property
    def min(self):
        return exp(self._min) if self.log else self._min

    @min.setter
    def min(self, values):
        values = asarray(values)
        self._min = nplog(values) if self.log else deepcopy(values)

    @property
    def max(self):
        return exp(self._max) if self.log else self._max

    @max.setter
    def max(self, values):
        values = asarray(values)
        self._max = nplog(values) if self.log else deepcopy(values)

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
            x = nplog(x)
        if log:
            return uniform.logcdf(x, self._min, self.scale)
        else:
            return uniform.cdf(x, self._min, self.scale)


    def plot_pdf(self, log=False, **kwargs):
        bins = self.bins()
        t = r"$\tilde{U}("+str(self.min)+","+str(self.max)+")$"
        cP.plot(bins, self.probability(bins, log=log), label=t, **kwargs)


    def probability(self, x, log, i=s_[:]):

        if self.log:
            x = nplog(x)

        if log:
            out = squeeze(uniform.logpdf(x, self._min, self.scale))
            probability = sum(out[i]) if self.multivariate else out

        else:
            out = squeeze(uniform.pdf(x, self._min, self.scale))
            probability =  prod(out[i]) if self.multivariate else out

        return probability


    def rng(self, size=1):
        values = self.prng.uniform(low=self._min, high=self._max, size=size)
        return exp(values) if self.log else values


    @property
    def summary(self):
        msg =  'Uniform Distribution: \n'
        msg += 'Min: {}\n'.format(self.min)
        msg += 'Max: {}\n'.format(self.max)
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
                bins = empty([nD, nBins+1])
                for i in range(nD):
                    bins[i, :] = linspace(self._min[i], self._max[i], nBins+1)
                values = DataArray(squeeze(bins))
            else:
                bins = empty(nBins+1)
                bins[:] = linspace(self._min[dim], self._max[dim], nBins+1)
                values = DataArray(squeeze(bins))

        else:
            values = DataArray(squeeze(linspace(self._min, self._max, nBins+1)))

        return exp(values) if self.log else values
