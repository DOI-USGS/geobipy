""" @OrderStatistics
Module defining modified order statistics as in Malinverno2002Parsimonious Bayesian Markov chain Monte Carlo inversion
in a nonlinear geophysical problem,Geophysical Journal International
"""
from copy import deepcopy
import numpy as np
from scipy.special import factorial


class Order(object):
    """ Class defining Order Statistics
    Specific application to Bayesian inversion of EM data
    """

    def __init__(self, zmin, zmax, hmin, kmax, *args, **kwargs):
        """ Initialize the order statistics
        zmin:  :log(minimum depth)
        zmax:  :log(maximum depth)
        hmin:  :log(minimum thickness)
        kmax:  :Maximum number of possible samples
        """
        # Probability Density Function
        if (zmin is None): return
        i=np.arange(kmax)
        dz = np.log((np.exp(zmax) - np.exp(zmin) - 2.0 * i * np.exp(hmin)))
        tmp = np.cumprod(dz)
        self.pdf = factorial(i) / tmp


    def deepcopy(self):
        """ Define a deepcopy routine """
        tmp = Order(None, 0, 0, 1)
        tmp.pdf = deepcopy(self.pdf)
        return tmp


    def getPdf(self, *args):
        """ Get the pdf value for a given number of layers """
        return self.pdf[args[0] - 1]  # Subtracts 1 for python


    def probability(self, x):
        tmp = np.int32(x)
        return self.pdf[tmp]


    def summary(self, out=False):
        msg = 'Order Statistics: \n'
        msg += str(self.pdf) + '\n'
        if (out):
            return msg
        print(msg)


    def hdfName(self):
        """ Create the group name for an HDF file """
        return('Distribution("Order",None,1.0,0.1,1)')


    def toHdf(self, h5obj, myName):
        """ Write the object to an HDF file """
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        grp.create_dataset('pdf', data=self.pdf)


    def fromHdf(self, h5grp):
        """ Reads the Uniform Distribution from an HDF group """
        T = np.array(h5grp.get('pdf'))
        self.pdf = T
        return self
