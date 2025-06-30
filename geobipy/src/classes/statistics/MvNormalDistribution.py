""" @MvNormalDistribution
Module defining a multivariate normal distribution with statistical procedures
"""
from copy import deepcopy
from numpy import all, atleast_1d, diag, diag_indices, dot, empty, exp, float64, full, hstack
from numpy import int32, linspace, maximum, newaxis, pi, prod, r_, repeat, size, squeeze, sqrt, zeros
from numpy import ndim as npndim
from numpy import log as nplog
from numpy.linalg import inv, slogdet
from scipy.stats import multivariate_normal
from ...base import utilities as cf
from ...base import plotting as cP
from .baseDistribution import baseDistribution
from .NormalDistribution import Normal
from ..core.DataArray import DataArray

class MvNormal(baseDistribution):
    """Class extension to geobipy.baseDistribution

    Handles a multivariate normal distribution.  Uses Scipy to evaluate probabilities,
    but Numpy to generate random samples since scipy is slow.

    MvNormal(mean, variance, ndim, prng)

    Parameters
    ----------
    mean : scalar or array_like
        Mean(s) for each dimension
    variance : scalar or array_like
        Variance for each dimension
    ndim : int, optional
        The number of dimensions in the multivariate normal.
        Only used if mean and variance are scalars that are constant for all dimensions
    prng : numpy.random.RandomState, optional
        A random state to generate random numbers. Required for parallel instantiation.

    Returns
    -------
    out : MvNormal
        Multivariate normal distribution.

    """


    def __init__(self, mean, variance, ndim=None, prng=None, **kwargs):
        """ Initialize a normal distribution
        mu:     :Mean of the distribution
        sigma:  :Standard deviation of the distribution

        """

        super().__init__(prng)

        if ndim is None:
            self._mean = atleast_1d(mean).copy()

            self.variance = variance

            self._constant = False

        else:

            assert size(mean) == 1, ValueError("When specifying ndim, mean must be a scalar.")
            assert size(variance) == 1, ValueError("When specifying ndim, variance must be a scalar.")

            ndim = int32(maximum(1, ndim))
            self._constant = True
            self._mean = full(ndim, fill_value=mean)
            self._variance = full(ndim, fill_value=variance)

    @property
    def address(self):
        return hstack([hex(id(self)), hex(id(self._mean)), hex(id(self._variance))])

    @property
    def addressof(self):
        msg =  "{} {}\n".format(type(self).__name__, hex(id(self)))
        msg += '    Mean:{}\n'.format(hex(id(self._mean)))
        msg += 'Variance:{}\n'.format(hex(id(self._variance)))
        return msg

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, values):
        self._mean[:] = values

    @property
    def multivariate(self):
        return True

    @property
    def ndim(self):
        return size(self.mean)

    @ndim.setter
    def ndim(self, newDimension):
        newDimension = int32(newDimension)
        if newDimension == self.ndim:
            return
        assert newDimension > 0, ValueError("Cannot have zero dimensions.")
        assert self._constant, ValueError("Cannot change the dimension of a non-constant multivariate distribution.")
        if npndim(self.mean) == 0:
            mean = self._mean
        else:
            mean = self._mean[0]

        variance = self.variance[0, 0]

        self._mean = full(newDimension, fill_value=mean)
        self._variance = full(newDimension, fill_value=variance)

    @property
    def std(self):
        return sqrt(self.variance)

    @property
    def variance(self):

        if npndim(self._variance) < 2:
            return diag(self._variance)
        return self._variance

    @variance.setter
    def variance(self, values):
        self._variance = atleast_1d(values).copy()
        # self._variance = zeros((self.ndim, self.ndim))

        # # Variance
        # nd = npndim(values)
        # if nd == 0:
        #     self._variance[diag_indices(self.ndim)] = values

        # elif nd == 1:
        #     assert size(values) == self.ndim, Exception('Mismatch in size of mean and variance')
        #     self._variance[diag_indices(self.ndim)] = values

        # elif nd == 2:
        #     self._variance[:, :] = values

    @property
    def precision(self):
        return cf.inv(self._variance)

    def __deepcopy__(self, memo={}):
        """ Define a deepcopy routine """
        if self._constant:
            return MvNormal(mean=self.mean[0], variance=self.variance[0, 0], ndim=self.ndim, prng=self.prng)
        else:
            return MvNormal(mean=self.mean, variance=self.variance, prng=self.prng)

    def derivative(self, x, order):

        assert order in [1, 2], ValueError("Order must be 1 or 2.")
        if order == 1:
            return cf.Ax(self.precision, x - self._mean)

        elif order == 2:
            return self.precision

    def deviation(self, x):
        return x - self._mean

    # def derivative(self, x, order):

    #     assert order in [1, 2], ValueError("Order must be 1 or 2.")
    #     if order == 1:
    #         return cf.Ax(self.inverseVariance, (x - self._mean)) * self.probability(x)
    #     elif order == 2:
    #         return cf.Ax(self.inverseVariance, self.probability(x))

    def mahalanobis(self, x):
        tmp = x - self.mean
        return sqrt(dot(tmp, dot(self.precision, tmp)))

    def rng(self, size=1):
        """  """

        return atleast_1d(squeeze(self.prng.multivariate_normal(self._mean, self.variance, size=size)))

    def plot_pdf(self, log=False, **kwargs):
        bins = self.bins()
        t = r"$\tilde{N}(\mu="+str(self.mean)+r", \sigma^{2}="+str(self.variance)+")$"

        p = self.probability(bins, log=log)

        cP.plot(bins, p, label=t, **kwargs)

    def probability(self, x, log, axis=None, **kwargs):
        """ For a realization x, compute the probability """

        if axis is not None:
            d = Normal(mean=self._mean[axis], variance=self.variance[axis, axis])
            return d.probability(x, log)

        import numpy as np
        N = size(x); nsd = np.ndim(x)
        nD = self.ndim

        if nsd != nD:
            if N != nD:
                x = np.repeat(x[:, np.newaxis], nD, 1)

        mean = self._mean
        if (nD == 1):
            mean = repeat(self._mean, N)

        pdf = multivariate_normal.logpdf if log else multivariate_normal.pdf

        return DataArray(pdf(x, mean=mean, cov=self.variance, allow_singular=True), name='Probability Density')

    @property
    def summary(self):
        msg =  "{}\n".format(type(self).__name__)
        msg += '    Mean:{}\n'.format(self._mean)
        msg += 'Variance:{}\n'.format(self._variance)
        return msg

    def pad(self, N):
        """ Pads the mean and variance to the given size
        N: Padded size
        """
        if (self._variance.ndim == 1):
            return MvNormal(zeros(N, dtype=self.mean.dtype), zeros(N, dtype=self.variance.dtype), prng=self.prng)
        if (self._variance.ndim == 2):
            return MvNormal(zeros(N, dtype=self.mean.dtype), zeros([N, N], dtype=self.variance.dtype), prng=self.prng)

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
        import numpy as np
        nStd = float64(nStd)
        nD = self.ndim
        if (nD > 1):
            std = diag(self.std)
            if axis is None:
                tmp = np.outer(r_[-1.0, 1.0], (nStd*std))
                if not relative:
                    tmp += self._mean
                tmp = np.r_[np.min(tmp), np.max(tmp)]
                bins = linspace(*tmp, nBins+1)

            else:
                bins = empty(nBins+1)
                tmp = squeeze(nStd * std[axis])
                t = linspace(-tmp, tmp, nBins+1)
                if not relative:
                    t += self._mean[axis]
                bins[:] = t

        else:
            tmp = nStd * self.std
            bins = squeeze(linspace(-tmp, tmp, nBins+1))
            if not relative:
                bins += self._mean

        return DataArray(bins)
