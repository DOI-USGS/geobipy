""" @OrderStatistics
Module defining modified order statistics as in Malinverno2002Parsimonious Bayesian Markov chain Monte Carlo inversion
in a nonlinear geophysical problem,Geophysical Journal International
"""
from copy import deepcopy
from numpy import arange, cumprod, int32, size, squeeze, hstack
from numpy import log as nplog
from scipy.special import factorial
from . import StatArray
from .baseDistribution import baseDistribution


class Order(baseDistribution):
    """ Class defining Order Statistics
    Specific application to Bayesian inversion of EM data
    """

    def __init__(self, denominator, **kwargs):
        """ Initialize the order statistics

        """
        if (denominator is None): return

        i = arange(size(denominator))
        self.denominator = deepcopy(denominator)
        tmp = cumprod(denominator)
        self.pdf = factorial(i) / tmp

    @property
    def address(self):
        return hstack([hex(id(self)), hex(id(self.denominator))])

    @property
    def addressof(self):
        msg =  "{} {}\n".format(type(self).__name__, hex(id(self)))
        msg += 'Denominator:{}\n'.format(hex(id(self.denominator)))
        return msg

    @property
    def multivariate(self):
        return False


    def __deepcopy__(self, memo={}):
        """ Define a deepcopy routine """
        return Order(self.denominator)


    def probability(self, x, log, **kwargs):
        tmp = squeeze(self.pdf[int32(x)])
        return nplog(tmp) if log else tmp

    @property
    def summary(self):
        msg = ('Order Statistics: \n'
               '{} \n').format(None)
        return msg


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
    #     T = array(h5grp.get('pdf'))
    #     self.pdf = T
    #     return self
