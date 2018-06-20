from geobipy.src.classes.core.myObject import myObject
from numpy.random import random as rng

class baseDistribution(myObject):
    """ Define an abstract base distribution class """

    def __init__(self, prng=None):
        self.multivariate = False

        if (prng is None):
            self.prng = rng.__self__
        else:
            self.prng = prng

    def getBins(self, size=100):
        """ Place Holder for children """
        assert False, 'Should not calling '+__name__+'.getBins()'

    def deepcopy():
        """ Place holder for children """
        assert False, 'Should not calling '+__name__+'.deepcopy()'
