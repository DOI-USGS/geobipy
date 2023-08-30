""" @MvNormalDistribution
Module defining a multivariate normal distribution with statistical procedures
"""
from copy import deepcopy
from numpy import all, atleast_1d, diag, diag_indices, dot, empty, exp, float64, full
from numpy import int32, linspace, maximum, pi, prod, repeat, size, squeeze, sqrt, zeros
from numpy import ndim as npndim
from numpy import log as nplog
from numpy.linalg import inv, slogdet
from ...base import utilities as cf
from .baseDistribution import baseDistribution
from .NormalDistribution import Normal
from ..core import StatArray
from scipy.stats import multivariate_normal

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

        if (type(variance) is float):
            variance = float64(variance)

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
            self._variance = diag(full(ndim, fill_value=variance))

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

        # if ndim(self.variance) == 0:
        #     variance = self.variance
        # else:
        variance = self.variance[0, 0]

        self._mean = full(newDimension, fill_value=mean)
        self._variance = diag(full(newDimension, fill_value=variance))

    @property
    def std(self):
        return sqrt(self.variance)

    @property
    def variance(self):
        return self._variance

    @variance.setter
    def variance(self, values):

        self._variance = zeros((self.ndim, self.ndim))

        # Variance
        nd = npndim(values)
        if nd == 0:
            self._variance[diag_indices(self.ndim)] = values

        elif nd == 1:
            assert size(values) == self.ndim, Exception('Mismatch in size of mean and variance')
            self._variance[diag_indices(self.ndim)] = values

        elif nd == 2:
            self._variance[:, :] = values

    @property
    def precision(self):
        return inv(self.variance)

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

    def probability(self, x, log, axis=None, **kwargs):
        """ For a realization x, compute the probability """

        if axis is not None:
            d = Normal(mean=self._mean[axis], variance=self.variance[axis, axis])
            return d.probability(x, log)

        N = size(x)
        nD = size(self.mean)

        if N != nD:
            probability = empty((nD, *x.shape))
            for i in range(nD):
                d = Normal(mean=self._mean[i], variance=self.variance[i, i])

                probability[i, :] = d.probability(x, log)
            return probability

        if log:
            # assert (N == nD), TypeError(
            #     'size of samples {} must equal number of distribution dimensions {} for a multivariate distribution'.format(N, nD))

            mean = self._mean
            if (nD == 1):
                mean = repeat(self._mean, N)

            dv = 0.5 * prod(slogdet(self.variance))
            # subtract the mean from the samples
            xMu = x - mean
            # Start computing the exponent term
            # e^(-0.5*(x-mu)'*inv(cov)*(x-mu))                        (1)
            # Compute the multiplication on the right of equation 1
            # Probability Density Function
            return -(0.5 * N) * nplog(2.0 * pi) - dv - 0.5 * dot(xMu, dot(self.precision, xMu))

        else:

            if N != nD:
                probability = empty((nD, *x.shape))
                for i in range(nD):
                    probability[i, :] = self.probability(x, log, axis=i)
                return probability


            # assert (N == nD), TypeError(
            #     'size of samples {} must equal number of distribution dimensions {} for a multivariate distribution'.format(N, nD))
            # For a diagonal matrix, the determinant is the product of the diagonal
            # entries
            # subtract the mean from the samples.
            xMu = x - self._mean
            # Take the inverse of the variance
            expo = exp(-0.5 * dot(xMu, dot(self.precision, xMu)))
            # Probability Density Function
            prob = (1.0 / sqrt(((2.0 * pi)**N) * cf.Det(self.variance))) * expo
            return prob

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
        if (self.variance.ndim == 1):
            return MvNormal(zeros(N, dtype=self.mean.dtype), zeros(N, dtype=self.variance.dtype), prng=self.prng)
        if (self.variance.ndim == 2):
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
        nStd = float64(nStd)
        nD = self.ndim
        if (nD > 1):
            if axis is None:
                bins = empty([nD, nBins+1])
                for i in range(nD):
                    tmp = squeeze(nStd * self.std[axis, axis])
                    t = linspace(-tmp, tmp, nBins+1)
                    if not relative:
                        t += self._mean[i]
                    bins[i, :] = t
            else:
                bins = empty(nBins+1)
                tmp = squeeze(nStd * self.std[axis, axis])
                t = linspace(-tmp, tmp, nBins+1)
                if not relative:
                    t += self._mean[axis]
                bins[:] = t

        else:
            tmp = nStd * self.std
            bins = squeeze(linspace(-tmp, tmp, nBins+1))
            if not relative:
                bins += self._mean

        return StatArray.StatArray(bins)
