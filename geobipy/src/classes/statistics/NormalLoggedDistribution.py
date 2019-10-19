""" @NormalLoggedDistribution
Module defining a normal distribution for log(parameter) with statistical procedures
"""
from copy import deepcopy
from ...base.logging import myLogger
from .baseDistribution import baseDistribution
import numpy as np
from ...base.HDF.hdfWrite import writeNumpy
from .MvNormalLoggedDistribution import MvNormalLog
from ..core import StatArray

class NormalLog(baseDistribution):
    """ Class defining a normal distribution """

    def __init__(self, mean, variance, prng=None):
        """ Initialize a normal distribution
        mu:     :Mean of the distribution
        sigma:  :Standard deviation of the distribution
        """
        assert np.size(mean) == 1, 'Univariate Normal Logged mean must have size = 1'
        assert np.size(variance) == 1, 'Univariate Normal Logged variance must have size = 1'
        super().__init__(prng)
        # Mean
        self.mean = deepcopy(mean)
        # Variance
        self.variance = deepcopy(variance)


    @property
    def ndim(self):
        return 1

    @property
    def multivariate(self):
        return False


    def deepcopy(self):
        """ Define a deepcopy routine """
        return NormalLog(self.mean, self.variance, self.prng)
#
#    def getPdf(self, x):
#        """ get the PDF, for a normal logged distribution
#        Currently assumes only variances and constant variance
#        """

    def probability(self, x):
        # ####lg.myLogger("Global");####lg.indent()
        ####lg.info('Getting PDF for Normal Log distribution')
#        N = np.size(x)
        ####lg.debug('Variance: '+str(self.variance))
        # For a diagonal matrix, the log determinant is the cumulative sum of
        # the log of the diagonal entries
        tmp = self.variance #cf.LogDet(self.variance, N)
        ####lg.info('LogDeterminant: '+str(tmp))
        dv = 0.5 * tmp
        # subtract the mean from the samples
        ####lg.debug('mean: '+str(self.mean))
        ####lg.debug('   x: '+str(x))
        xMu = x - self.mean
#   ####lg.info('xMu: '+str(xMu))
        # Start computing the exponent term
        # e^(-0.5*(x-mu)'*inv(cov)*(x-mu))                        (1)
        # Take the inverse of the variance
        tmp = 1.0/self.variance #cf.Inv(self.variance)
#   ####lg.info('tmp:'+str(tmp))
        # Compute the multiplication on the right of equation 1
#        iv = cf.Ax(tmp, xMu)
#   ####lg.info('iv: '+str(iv))
        tmp = 0.5 * xMu * tmp * xMu #np.dot(xMu, iv)
#   ####lg.info('tmp: '+str(tmp))
        # Probability Density Function
        prob = -(0.5) * np.log(2.0 * np.pi) - dv - tmp
#   ####lg.info('pdf: '+str(self.pdf))
        # ####lg.dedent()
        return np.float64(prob)

    def summary(self, out=False):
        msg = 'Normal Logged Distribution: \n'
        msg += '  Mean:      :' + str(self.mean) + '\n'
        msg += '  Variance:  :' + str(self.variance) + '\n'
        if (out):
            return msg
        print(msg)


    def getBinEdges(self, nBins = 100, nStd=4.0):
        """ Discretizes a range given the mean and variance of the distribution """
        tmp = nStd * np.sqrt(self.variance)
        return StatArray.StatArray(np.linspace(self.mean - tmp, self.mean + tmp, nBins+1))

#    def hdfName(self):
#        """ Create the group name for an HDF file """
#        return('Distribution("NormalLog",0.0,1.0)')
#
#    def createHdf(self, parent, myName):
#        """ Create the hdf group metadata in file """
#        grp = parent.create_group(myName)
#        grp.attrs["repr"] = self.hdfName()
#        grp.create_dataset('mean', (1,), dtype=self.mean.dtype)
#        grp.create_dataset('variance', (1,), dtype=self.var.dtype)
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
#        writeNumpy(self.var,grp,'variance')
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
