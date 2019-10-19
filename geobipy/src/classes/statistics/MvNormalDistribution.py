""" @MvNormalDistribution
Module defining a multivariate normal distribution with statistical procedures
"""
#from copy import deepcopy
import numpy as np
from ...base  import customFunctions as cf
from ...base.logging import myLogger
from .baseDistribution import baseDistribution
from .NormalDistribution import Normal
from ...base.HDF.hdfWrite import writeNumpy
from ..core import StatArray

class MvNormal(baseDistribution):
    """Multivariate normal distribution """

    def __init__(self, mean, variance, prng=None):
        """ Initialize a normal distribution
        mu:     :Mean of the distribution
        sigma:  :Standard deviation of the distribution

        """
        #assert (np.ndim(mean) > 0 and np.ndim(variance) > 0), ValueError("mean and variance must be > 1 dimension")

        if (type(variance) is float): variance = np.float64(variance)

        baseDistribution.__init__(self, prng)
        # Mean
        if np.ndim(mean) == 0:
            self.mean = np.float64(mean)
        else:
            self.mean = np.copy(mean)

        # Variance
        if np.ndim(variance) == 0:
            self.variance = np.zeros(np.size(mean))
            self.variance[:] = variance
            return
        if (np.ndim(variance) == 1):
            assert np.size(variance) == np.size(mean), 'Mismatch in size of mean and variance'
            self.variance = np.zeros(np.size(mean))
            self.variance[:] += variance
            return
        assert (variance.shape[0] == mean.size and variance.shape[1] == mean.size), 'Covariance must have same dimensions as the mean'
        self.variance = np.asarray(variance)

    @property
    def ndim(self):
        return self.mean.size

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def multivariate(self):
        return True


    def deepcopy(self):
        """ Define a deepcopy routine """
        # return deepcopy(self)
        return MvNormal(self.mean, self.variance, self.prng)


    # def getPdf(self, x):
    #     """ Get the PDF of Normal Distribution for the values in x """
    #     N = np.size(x)
    #     pdf = np.zeros(N)
    #     for i in range(N):
    #         pdf[i] = self.probability(x[i])
    #     return pdf

    def rng(self, size = 1):
        """  """

        variance = np.squeeze(self.variance)
        if variance.ndim == 0:
            return np.sqrt(variance) * self.prng.randn(size) + self.mean
        if (variance.ndim == 1):
            tmp = self.prng.multivariate_normal(self.mean, np.diag(variance), size)
            return tmp
        else:
            return self.prng.multivariate_normal(self.mean, variance, size)

    def probability(self, samples):
        """ For a realization x, compute the probability """
        N = samples.size
        nD = self.mean.size

        assert (N == nD), TypeError('size of samples {} must equal number of distribution dimensions {} for a multivariate distribution'.format(N, nD))
        # For a diagonal matrix, the determinant is the product of the diagonal
        # entries
        dv = cf.Det(self.variance)
        # subtract the mean from the samples.
        xMu = samples - self.mean
        # Take the inverse of the variance
        tmp = cf.Inv(self.variance)
        iv = cf.Ax(tmp, xMu)
        exp = np.exp(-0.5 * np.dot(xMu, iv))
        # Probability Density Function
        prob = (1.0 / np.sqrt(((2.0 * np.pi)**N) * dv)) * exp
        return prob

    def summary(self, out=False):
        msg = 'MV Normal Distribution: \n'
        msg += '    Mean: ' + str(self.mean) + '\n'
        msg += '    Variance: ' + str(self.variance) + '\n'
        
        return msg if out else print(msg)
    

    def pad(self, N):
        """ Pads the mean and variance to the given size
        N: Padded size
        """
        if (self.variance.ndim == 1):
            return MvNormal(np.zeros(N,dtype=self.mean.dtype),np.zeros(N, dtype=self.variance.dtype), self.prng)
        if (self.variance.ndim == 2):
            return MvNormal(np.zeros(N,dtype=self.mean.dtype),np.zeros([N,N], dtype=self.variance.dtype), self.prng)

#    def hdfName(self):
#        """ Create the group name for an HDF file """
#        return('Distribution("MvNormal",0.0,1.0)')
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
#        return MvNormal(T1, T2)

    def getBinEdges(self, nBins=100, nStd=4.0, dim=None):
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
        
        nStd = np.float64(nStd)
        nD = self.ndim
        if (nD > 1):
            if dim is None:
                bins = np.empty([nD, nBins+1])
                for i in range(nD):
                    tmp = nStd * np.sqrt(self.variance[i])
                    bins[i, :] = np.linspace(self.mean[i] - tmp, self.mean[i] + tmp, nBins+1)
                return StatArray.StatArray(bins)
            else:
                bins = np.empty(nBins+1)
                tmp = nStd * np.sqrt(self.variance[dim])
                bins[:] = np.linspace(self.mean[dim] - tmp, self.mean[dim] + tmp, nBins+1)
                return StatArray.StatArray(bins)

        tmp = nStd * np.sqrt(self.variance)
        return StatArray.StatArray(np.linspace(self.mean - tmp, self.mean + tmp, nBins+1))
