from ..core.myObject import myObject
from numpy.random import Generator
from randomgen import Xoshiro256

class baseDistribution(myObject):
    """ Define an abstract base distribution class """

    def __init__(self, prng=None):

        if (prng is None):
            self.prng = Generator(Xoshiro256())
        else:
            self.prng = prng

    @property
    def moment(self):
        """ Place Holder for children """
        assert False, 'Should not calling '+__name__+'.moment'

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
