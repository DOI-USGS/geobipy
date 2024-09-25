from ..core.myObject import myObject
from numpy.random import Generator

class baseDistribution(myObject):
    """ Define an abstract base distribution class """

    def __init__(self, prng):
        self.prng = prng

    @property
    def moment(self):
        """ Place Holder for children """
        assert False, 'Should not calling '+__name__+'.moment'

    @property
    def ndim(self):
        """ Place Holder for children """
        assert False, 'Should not calling '+__name__+'.ndim'

    @property
    def prng(self):
        return self._prng

    @prng.setter
    def prng(self, value):
        # from ...base.MPI import get_prng

        # if value is None:
        #     import time
        #     value = get_prng(time.time)

        assert isinstance(value, Generator), TypeError(("prng must have type np.random.Generator.\n"
                                                        "You can generate one using\n"
                                                        "from numpy.random import Generator\n"
                                                        "from numpy.random import PCG64DXSM\n"
                                                        "Generator(bit_generator)\n\n"
                                                        "Where bit_generator is one of the several generators from either numpy or randomgen"))

        self._prng = value

    def bins(self):
        """ Place Holder for children """
        assert False, 'Should not calling '+__name__+'.bins()'

    def deepcopy():
        """ Place holder for children """
        assert False, 'Should not calling '+__name__+'.deepcopy()'
