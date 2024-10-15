"""
o"""

import sys
from abc import ABC, abstractmethod
from numpy import log10
import h5py

class myObject(ABC):

    __slots__ = ()
    """Abstract base class """

    # def __init__(self, **kwargs):
    #     raise NotImplementedError()

    def __getsizeof__(self):
        """Get the size of the object in memory """
        i = 0
        for k, v in self.__dict__.items():
            i += sys.getsizeof(v)
        return i

    @abstractmethod
    def __deepcopy__(self, memo={}):
        return None

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
        return type(self).__name__

    def create_hdf_group(self, h5obj, name, hdf_name=None, **kwargs):
        grp = h5obj.create_group(name)
        if hdf_name is None:
            hdf_name = self.hdf_name
        grp.attrs["repr"] = hdf_name
        return grp

    def pyvista_plotter(self, plotter=None, **kwargs):

        import pyvista as pv

        if plotter is None:
            plotter = pv.Plotter()
        plotter.add_mesh(self.pyvista_mesh(), show_edges=True)

        labels = dict(xlabel=self.x.label, ylabel=self.y.label, zlabel=self.z.label)
        plotter.show_grid()
        plotter.add_axes(**labels)

        return plotter

    def to_vtk(self, filename):
        mesh = self.pyvista_mesh()
        mesh.save(filename)

    def toHdf(self, h5obj, name, withPosterior=False):
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
                self.createHdf(f, name, withPosterior=withPosterior)
                self.writeHdf(f, name, withPosterior=withPosterior)
            return

        self.createHdf(h5obj, name, withPosterior=withPosterior)
        self.writeHdf(h5obj, name, withPosterior=withPosterior)
