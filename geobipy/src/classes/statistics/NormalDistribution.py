""" @NormalDistribution
Module defining a normal distribution with statistical procedures
"""
#from copy import deepcopy
import numpy as np
from ...base.logging import myLogger
from .baseDistribution import baseDistribution
from ...base.HDF.hdfWrite import writeNumpy
#from .MvNormalDistribution import MvNormal
from scipy.stats import norm
from ...base import customPlots as cP
from ..core import StatArray

class Normal(baseDistribution):
    """Univariate normal distribution

    Normal(mean, variance)

    Parameters
    ----------
    mean : numpy.float
        The mean of the distribution
    variance : numpy.float
        The variance of the distribution

    """
    def __init__(self, mean, variance, prng=None):
        """Instantiate a Normal distribution """
        assert np.size(mean) == 1, 'Univariate Normal mean must have size = 1'
        assert np.size(variance) == 1, 'Univariate Normal variance must have size = 1'
        baseDistribution.__init__(self, prng)
        self.mean = mean
        self.variance = variance


    @property
    def ndim(self):
        return 1


    def deepcopy(self):
        """Create a deepcopy

        Returns
        -------
        out
            Normal

        """
        # return deepcopy(self)
        return Normal(self.mean, self.variance, self.prng)


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
        return self.prng.normal(size=size, loc=self.mean, scale=self.variance)

    
    def plotPDF(self, **kwargs):

        
        bins = self.getBinEdges()
        t = r"$\tilde{N}(\mu="+str(self.mean)+", \sigma^{2}="+str(self.variance)+")$"

        cP.plot(bins, self.probability(bins), label=t, **kwargs)


    def probability(self, x):
        """ For a realization x, compute the probability """
        return norm.pdf(x,loc=self.mean, scale = self.variance)
        

    def summary(self, out=False):
        msg = 'Normal Distribution: \n'
        msg += '    Mean: :' + str(self.mean) + '\n'
        msg += 'Variance: :' + str(self.variance) + '\n'
        if (out):
            return msg
        print(msg)

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
#        T1 = np.array(h5grp.get('mean'))
#        T2 = np.array(h5grp.get('variance'))
#        return MvNormal(T1, T2)

    def getBinEdges(self, nBins = 100, nStd=4.0):
        """ Discretizes a range given the mean and variance of the distribution """
        tmp = nStd * np.sqrt(self.variance)
        return StatArray.StatArray(np.linspace(self.mean - tmp, self.mean + tmp, nBins+1))

