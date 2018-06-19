""" @MvNormalLoggedDistribution
Module defining a multivariate normal distribution for log(parameter) with statistical procedures
"""
from copy import deepcopy
from ...base.logging import myLogger
import numpy as np
from .baseDistribution import baseDistribution
from ...base import customFunctions as cf
from ...base.HDF.hdfWrite import writeNumpy

class MvNormalLog(baseDistribution):
    """ Class defining a normal distribution """

    def __init__(self, mean, variance, prng=None):
        """ Initialize a normal distribution
        mu:     :Mean of the distribution
        sigma:  :Standard deviation of the distribution
        """
        if (type(mean) is float): mean=np.float64(mean)
        if (type(variance) is float): variance=np.float64(variance)
        baseDistribution.__init__(self, prng)
        # Mean
        self.mean = deepcopy(mean)
        # Variance
        self.variance = deepcopy(variance)

        self.multivariate = True


    def deepcopy(self):
        """ Define a deepcopy routine """
        return MvNormalLog(self.mean, self.variance, self.prng)

#    def getPdf(self, x):
#        """ get the PDF, for a normal logged distribution
#        Currently assumes only variances and constant variance
#        """
#        return self.getPdf(x)


    def probability(self, samples):
        """  """
        N = np.size(samples)
        nD = self.mean.size

        mean = self.mean
        if (nD == 1):
            mean = np.repeat(self.mean, N)
        else:
            assert (N == nD), TypeError('size of samples {} must equal number of distribution dimensions {} for a multivariate distribution'.format(N, nD))
        
        # For a diagonal matrix, the log determinant is the cumulative sum of
        # the log of the diagonal entries

        tmp = cf.LogDet(self.variance, N)
        dv = 0.5 * tmp
        # subtract the mean from the samples
        xMu = samples - mean
        # Start computing the exponent term
        # e^(-0.5*(x-mu)'*inv(cov)*(x-mu))                        (1)
        # Take the inverse of the variance
        tmp = cf.Inv(self.variance)
        # Compute the multiplication on the right of equation 1
        iv = cf.Ax(tmp, xMu)
        tmp = 0.5 * np.dot(xMu, iv)
        # Probability Density Function
        prob = -(0.5 * N) * np.log(2.0 * np.pi) - dv - tmp
        tmp = np.float64(prob)
        return tmp

    def summary(self, out=False):
        msg = 'MV Normal Logged Distribution: \n'
        msg += '  Mean:      :' + str(self.mean) + '\n'
        msg += '  Variance:  :' + str(self.variance) + '\n'
        if (out):
            return msg
        print(msg)

    def pad(self, N):
        """ Pads the mean and variance to the given size
        N: Padded size
        """
        if (self.variance.ndim == 1):
            return MvNormalLog(np.zeros(N,dtype=self.mean.dtype),np.zeros(N, dtype=self.variance.dtype), self.prng)
        if (self.variance.ndim == 2):
            return MvNormalLog(np.zeros(N,dtype=self.mean.dtype),np.zeros([N,N], dtype=self.variance.dtype), self.prng)

#    def hdfName(self):
#        """ Create the group name for an HDF file """
#        return('Distribution("MvNormalLog",0.0,1.0)')
#
#    def createHdf(self, parent, myName):
#        """ Create the hdf group metadata in file """
#        grp = parent.create_group(myName)
#        grp.attrs["repr"] = self.hdfName()
#        grp.create_dataset('mean', self.mean.shape, dtype=self.mean.dtype)
#        grp.create_dataset('variance', self.variance.shape, dtype=self.variance.dtype)
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
#    def toHdf(self, h5obj, myName):
#        """ Write the object to an HDF file """
#        grp = h5obj.create_group(myName)
#        grp.attrs["repr"] = self.hdfName()
#        grp.create_dataset('mean', data=self.mean)
#        grp.create_dataset('variance', data=self.variance)
#
#    def fromHdf(self, h5grp):
#        """ Reads the Uniform Distribution from an HDF group """
#        T1 = np.array(h5grp.get('mean'))
#        T2 = np.array(h5grp.get('variance'))
#        return MvNormalLog(T1, T2)
