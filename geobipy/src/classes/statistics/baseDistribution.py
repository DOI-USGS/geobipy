from geobipy.src.classes.core.myObject import myObject
from numpy.random import random as rng

class baseDistribution(myObject):
    """ Define an abstract base distribution class """

    def __init__(self, prng=None):
        
        if (prng is None):
            self.prng = rng.__self__
        else:
            self.prng = prng

    @property
    def ndim(self):
        """ Place Holder for children """
        assert False, 'Should not calling '+__name__+'.ndim'

    def bins(self):
        """ Place Holder for children """
        assert False, 'Should not calling '+__name__+'.bins()'

    def deepcopy():
        """ Place holder for children """
        assert False, 'Should not calling '+__name__+'.deepcopy()'
