""" @NormalDistribution
Module defining a normal distribution with statistical procedures
"""
#from copy import deepcopy
from numpy import array, exp, linspace, size, squeeze, sqrt
from numpy import log as nplog
from .NormalDistribution import Normal
from scipy.stats import norm
from ...base import plotting as cP
from ..core.DataArray import DataArray

class LogNormal(Normal):
    """Univariate normal distribution

    Normal(mean, variance)

    Parameters
    ----------
    mean : numpy.float
        The mean of the distribution
    variance : numpy.float
        The variance of the distribution

    """
    def __init__(self, mean, variance, linearSpace=False, prng=None):
        """Instantiate a Normal distribution """

        if linearSpace:
            mean = nplog(mean)
        super().__init__(mean, variance, prng=prng)
        self.linearSpace = linearSpace

    @property
    def mean(self):
        return exp(self._mean) if self.linearSpace else self._mean

    @mean.setter
    def mean(self, value):
        self._mean = nplog(value) if self.linearSpace else value

    @property
    def ndim(self):
        return 1

    @property
    def multivariate(self):
        return False

    @property
    def variance(self):
        return self._variance


    def cdf(self, x):
        """ For a realization x, compute the probability """
        if self.linearSpace:
            x = nplog(x)
        return DataArray(norm.cdf(x, loc = self._mean, scale = self.variance), "Cumulative Density")


    def __deepcopy__(self, memo={}):
        """Create a deepcopy

        Returns
        -------
        out
            Normal

        """
        return LogNormal(self.mean, self.variance, self.linearSpace, self.prng)


    def derivative(self, x, moment):
        assert 0 <= moment <= 1, ValueError("Must have 0 <= moment < 2")

        if self.linearSpace:
            x = nplog(x)

        if moment == 0:
            return ((x - self._mean) / self.variance)
        else:
            return (0.5 / self.variance**2.0) * ((x - self._mean)**2.0 - self.variance)



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
        size = (size, size(self.mean))
        values = squeeze(self.prng.normal(loc=self._mean, scale=self.variance, size=size))

        return exp(values) if self.linearSpace else values


    def plot_pdf(self, log=False, **kwargs):


        bins = self.bins()
        t = r"$\tilde{N}(\mu="+str(self.mean)+r", \sigma^{2}="+str(self.variance)+")$"

        cP.plot(bins, self.probability(bins, log=log), label=t, **kwargs)


    def probability(self, x, log):
        """ For a realization x, compute the probability """

        if self.linearSpace:
            x = nplog(x)

        if log:
            return DataArray(norm.logpdf(x, loc = self._mean, scale = self.variance), "Probability Density")
        else:
            return DataArray(norm.pdf(x, loc = self._mean, scale = self.variance), "Probability Density")

#    def hdfName(self):
#        """ Create the group name for an HDF file """
#        return('Distribution("Normal",0.0,1.0)')
#
#    def toHdf(self, h5obj, myName):
#        """ Write the object to an HDF file """
#        grp = h5obj.create_group(myName)
#        grp.attrs["repr"] = self.hdfName()
#        grp.create_dataset('mean', data=self.mean)
#        grp.create_dataset('variance', data=self.variance)
#
#    def createHdf(self, parent, myName):
#        """ Create the hdf group metadata in file """
#        grp = parent.create_group(myName)
#        grp.attrs["repr"] = self.hdfName()
#        grp.create_dataset('mean', (1,), dtype=self.mean.dtype)
#        grp.create_dataset('variance', (1,), dtype=self.variance.dtype)
#
#    def writeHdf(self, parent, myName, create=True):
#        """ Write the StatArray to an HDF object
#        parent: Upper hdf file or group
#        myName: object hdf name. Assumes createHdf has already been called
#        """
#        # create a new group inside h5obj
#        if (create):
#            self.createHdf(parent, myName)
#
#        grp = parent.get(myName)
#        writeNumpy(self.mean,grp,'mean')
#        writeNumpy(self.variance,grp,'variance')
#
#    def fromHdf(self, h5grp):
#        """ Reads the Uniform Distribution from an HDF group """
#        T1 = array(h5grp.get('mean'))
#        T2 = array(h5grp.get('variance'))
#        return MvNormal(T1, T2)

    def bins(self, nBins = 99, nStd=4.0):
        """ Discretizes a range given the mean and variance of the distribution """
        tmp = nStd * sqrt(self.variance)
        values = linspace(self._mean - tmp, self._mean + tmp, nBins+1)

        return DataArray(exp(values)) if self.linearSpace else DataArray(values)
