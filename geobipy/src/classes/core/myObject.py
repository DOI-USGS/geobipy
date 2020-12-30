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


    @property
    def hdf_name(self):
        return str(self.__class__)


    def create_hdf_group(self, h5obj, name):
        grp = h5obj.create_group(name)
        grp.attrs["repr"] = self.hdfName()

        return grp


    def toHdf(self, h5obj, myName):
        """Create and write to HDF.

        Parameters
        ----------
        h5obj : h5py._hl.files.File or h5py._hl.group.Group
            A HDF file or group object to write the contents to.
        myName : str
            The name of the group to write the StatArray to.

        """

        if isinstance(h5obj, str):
            with h5py.File(h5obj, 'w') as f:
                self.createHdf(f, myName)
                self.writeHdf(f, myName)
                return

        self.createHdf(h5obj, myName)
        self.writeHdf(h5obj, myName)
