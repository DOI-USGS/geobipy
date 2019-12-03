""" @OrderStatistics
Module defining modified order statistics as in Malinverno2002Parsimonious Bayesian Markov chain Monte Carlo inversion
in a nonlinear geophysical problem,Geophysical Journal International
"""
from copy import deepcopy
import numpy as np
from scipy.special import factorial
from ..core import StatArray



class Order(object):
    """ Class defining Order Statistics
    Specific application to Bayesian inversion of EM data
    """

    def __init__(self, denominator, **kwargs):
        """ Initialize the order statistics
        
        """
        if (denominator is None): return
        
        i = np.arange(np.size(denominator))
        self.denominator = deepcopy(denominator)
        tmp = np.cumprod(denominator)
        self.pdf = factorial(i) / tmp


    @property
    def multivariate(self):
        return False


    def deepcopy(self):
        """ Define a deepcopy routine """
        return Order(self.denominator)


    def probability(self, x, log):
        tmp = np.squeeze(self.pdf[np.int(x)])

        return np.log(tmp) if log else tmp
        # print('tmp {}'.format(tmp))
        # if log:
        #     return StatArray.StatArray(np.log(tmp), "Log Probability Density")
        # else:
        #     return StatArray.StatArray(tmp, "Probability Density")


    def summary(self, out=False):
        msg = 'Order Statistics: \n'
        msg += str(self.pdf) + '\n'
        if (out):
            return msg
        print(msg)


    # def hdfName(self):
    #     """ Create the group name for an HDF file """
    #     return('Distribution("Order",None,1.0,0.1,1)')


    # def toHdf(self, h5obj, myName):
    #     """ Write the object to an HDF file """
    #     grp = h5obj.create_group(myName)
    #     grp.attrs["repr"] = self.hdfName()
    #     grp.create_dataset('pdf', data=self.pdf)


    # def fromHdf(self, h5grp):
    #     """ Reads the Uniform Distribution from an HDF group """
    #     T = np.array(h5grp.get('pdf'))
    #     self.pdf = T
    #     return self        
