""" @GammaDistribution
Module defining a gamma distribution with statistical procedures
"""
from copy import deepcopy
from numpy import exp
import scipy.special as sp


class Gamma(object):
    """ Class defining a normal distribution """

    def __init__(self, *args):
        """ Initialize a normal distribution
        args[0]:     :Scale of the distribution
        args[1]:     :Shape of the distribution
        arge[2]:Optional: Array to compute the pdf
        """
        # Mean
        self.shape = deepcopy(args[0])
        # Variance
        self.scale = deepcopy(args[1])
        # Initialize private variables for faster computation
        self._a = (self.shape - 1.0)
        self._b = -1.0 / self.scale
        self._c = 1.0 / (sp.gamma(self.shape) * (self.scale**self.shape))
        if (len(args) == 3):
            self.pdf(args[2])

    @property
    def ndim(self):
        return 1

    @property
    def multivariate(self):
        return False

    def pdf(self, x):
        """ set the PDF, for a gamma distribution """
        self.pdf = x**self._a * exp(x * self._b) * self._c
        return self.pdf

    @property
    def summary(self):
        msg = 'Gamma Distribution: \n'
        msg += '  Scale: :' + str(self.shape) + '\n'
        msg += '  Shape:  :' + str(self.shape) + '\n'
        return msg
