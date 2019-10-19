""" @UniformDistribution
Module defining a uniform distribution with statistical procedures
"""
from copy import deepcopy
from .baseDistribution import baseDistribution
from ...base.HDF.hdfWrite import writeNumpy
from ...base import customPlots as cP
import numpy as np
from scipy.stats import uniform
from ..core import StatArray


class Uniform(baseDistribution):
    """ Class defining a uniform distribution """

    def __init__(self, min, max, prng=None):
        """ Initialize a uniform distribution
        xmin:  :Minimum value
        xmax:  :Maximum value
        """

        assert max > min, ValueError("Maximum must be > minimum")
        super().__init__(prng)
        # Minimum
        self.min = deepcopy(min)
        # Maximum
        self.max = deepcopy(max)
        # Mean
        self.mean = 0.5 * (max + min)
        self.scale = max - min
        # Variance
        self.variance = (1.0 / 12.0) * self.scale**2.0
        
        


    @property
    def ndim(self):
        return np.size(self.min)

    @property
    def multivariate(self):
        return True if self.ndim > 1 else False


    def deepcopy(self):
        """ Define a deepcopy routine """
        return Uniform(self.min, self.max, self.prng)


    def getPdf(self, x=0):
        """ get the PDF, for a uniform distribution this does not need a procedure, however other distributions might, and we will need a function """
        return np.sum(self.pdf)


    def cdf(self, x):
        """ Get the value of the cumulative distribution function for a x """
        if (x < self.min):
            return 0.0
        if (x >= self.max):
            return 1.0
        return self.pdf * (x - self.min)


    def plotPDF(self, **kwargs):
        bins = self.getBinEdges()
        t = r"$\tilde{U}("+str(self.min)+","+str(self.max)+")$"
        cP.plot(bins, np.repeat(self.pdf, np.size(bins)), label=t, **kwargs)


    def probability(self, x):
        return uniform.pdf(x, self.min, self.scale)

    def rng(self, size=1):
        return self.prng.uniform(self.min, self.max, size=size)

    def summary(self, out=False):
        msg = 'Uniform Distribution: \n'
        msg += '  Min: :' + str(self.min) + '\n'
        msg += '  Max: :' + str(self.max) + '\n'
        if (out):
            return msg
        print(msg)

#    def hdfName(self):
#        """ Create the group name for an HDF file """
#        return('Distribution("Uniform",0.0,1.0,False)')
#
#    def createHdf(self, parent, myName):
#        """ Create the hdf group metadata in file """
#        grp = parent.create_group(myName)
#        grp.attrs["repr"] = self.hdfName()
#        grp.create_dataset('min', (1,), dtype=self.min.dtype)
#        grp.create_dataset('max', (1,), dtype=self.max.dtype)
#        grp.create_dataset('isLogged', (1,), dtype=bool)
#
#
#    def writeHdf(self, parent, myName, create=True):
#        """ Write the object to an HDF group
#        parent: Upper hdf file or group
#        myName: object hdf name. Assumes createHdf has already been called
#        """
#        # create a new group inside h5obj
#        if (create):
#            self.createHdf(parent, myName)
#
#        grp = parent.get(myName)
#        writeNumpy(self.min,grp,'min')
#        writeNumpy(self.max,grp,'max')
#        writeNumpy(self.logged,grp,'isLogged')
#
#    def toHdf(self, h5obj, myName):
#        """ Write the object to an HDF file """
#        grp = h5obj.create_group(myName)
#        grp.attrs["repr"] = self.hdfName()
#        grp.create_dataset('min', data=self.min)
#        grp.create_dataset('max', data=self.max)
#        grp.create_dataset('isLogged', data=self.logged, dtype=bool)
#
#    def fromHdf(self, h5grp):
#        """ Reads the Uniform Distribution from an HDF group """
#        minT = np.array(h5grp.get('min'))
#        maxT = np.array(h5grp.get('max'))
#        ilT = np.array(h5grp.get('isLogged'))
#        return Uniform(minT, maxT, ilT)


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
                return StatArray.StatArray(bins)
            else:
                bins = np.empty(nBins+1)
                bins[:] = np.linspace(self.min[dim], self.max[dim], nBins+1)
                return StatArray.StatArray(bins)

        return StatArray.StatArray(np.linspace(self.min, self.max, nBins+1))
