from copy import deepcopy
import numpy as np
from ...base import MPI as myMPI
from ...classes.core.myObject import myObject
from ...base.HDF.hdfWrite import writeNumpy
from abc import ABC, abstractclassmethod

class EmLoop(myObject, ABC):
    """Defines a loop in an EM system e.g. transmitter or reciever

    This is an abstract base class and should not be instantiated

    EmLoop()

    
    """

    def __init__(self):
        raise NotImplementedError("Abstract base class, not implemented")


    @abstractclassmethod
    def deepcopy(self):
        """Required by subclasses"""
        raise NotImplementedError("Abstract base class, not implemented")
        # return deepcopy(self)


    @abstractclassmethod
    def __deepcopy__(self, memo):
        """Required by subclasses"""
        raise NotImplementedError("Abstract base class, not implemented")
        # tmp = EmLoop(self.orient, self.moment, self.tx, self.ty, self.off, self.pitch, self.roll, self.yaw, self.radius)

