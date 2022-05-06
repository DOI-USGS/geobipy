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
    @property
    def summary(self):
        """Print a summary"""
        msg = super().summary

        msg += "orientation:\n{}\n".format("|   "+(self.orientation.replace("\n", "\n|   ")))
        msg += "moment:\n{}".format("|   "+(self.moment.summary.replace("\n", "\n|   "))[:-4])
        msg += "pitch:\n{}".format("|   "+(self.pitch.summary.replace("\n", "\n|   "))[:-4])
        msg += "roll:\n{}".format("|   "+(self.roll.summary.replace("\n", "\n|   "))[:-4])
        msg += "yaw:\n{}".format("|   "+(self.yaw.summary.replace("\n", "\n|   "))[:-4])

        return msg
