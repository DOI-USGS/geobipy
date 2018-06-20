"""
.. module:: myObject
   :platform: Unix, Windows
   :synopsis: abstract base class

.. moduleauthor:: Leon Foks

"""

import sys
from numpy import log10

class myObject(object):
    """Abstract base class """

    def __getsizeof__(self):
        """Get the size of the object in memory """
        i = 0
        for k, v in self.__dict__.items():
            i += sys.getsizeof(v)
        return i

    def getsizeof(self):
        """Get the size of the object in memory with nice output """
        i = self.__getsizeof__()
        j = int(log10(i))
        if (j < 3):
            return (str(i) + ' B')
        if (j < 6):
            return (str(i / 1024) + ' KB')
        if (j < 9):
            return (str(i / (1024**2)) + ' MB')
        if (j < 12):
            return (str(i / (1024**3)) + ' GB')
        if (j < 15):
            return (str(i / (1024**4)) + ' TB')
