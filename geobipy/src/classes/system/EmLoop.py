from copy import deepcopy
import numpy as np
from ...base import MPI as myMPI
from ...classes.pointcloud.Point import Point
from abc import ABC, abstractclassmethod

class EmLoop(Point, ABC):
    """Defines a loop in an EM system e.g. transmitter or reciever

    This is an abstract base class and should not be instantiated

    EmLoop()


    """

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

