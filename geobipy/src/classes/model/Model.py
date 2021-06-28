""" @Model_Class
Module describing a Model
"""
import numpy as np
from ..core.myObject import myObject
from ..core import StatArray


class Model(myObject):
    """Abstract Model Class

    This is an abstract base class for additional model classes

    See Also
    ----------
    geobipy.Model1D
    geobipy.Model2D

    """

    def __init__(self, mesh, values=None):
        """ Instantiate a 2D histogram """
        if (mesh is None):
            return
        # Instantiate the parent class
        self._mesh = mesh
        # Assign the values
        self.values = values

    @property
    def shape(self):
        return self.mesh.shape

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        if values is None:
            self._values = StatArray.StatArray(self.shape)
            return

        assert np.all(values.shape == self.shape), ValueError("values must have shape {}".format(self.shape))
        self._values = StatArray.StatArray(values)

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        assert isinstance(value, Mesh), TypeError

    @property
    def nCells(self):
        return self.mesh.nCells

    @property
    def x(self):
        return self.mesh.x

    @property
    def y(self):
        return self.mesh.y

    def interpolate_centres_to_nodes(self, kind='cubic'):
        return self.mesh.interpolate_centres_to_nodes(self.values, kind=kind, fill_value="extrapolate")

    def pcolor(self, **kwargs):
        """Plot the Histogram2D as an image

        Other Parameters
        ----------------
        alpha : scalar or array_like, optional
            If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
            If array_like, each pixel is given an individual alpha value.
        log : 'e' or float, optional
            Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
            Values in c that are <= 0 are masked.
        equalize : bool, optional
            Equalize the histogram of the colourmap so that all colours have an equal amount.
        nbins : int, optional
            Number of bins to use for histogram equalization.
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'.
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        grid : bool, optional
            Plot the grid
        noColorbar : bool, optional
            Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.

        """
        self.mesh.pcolor(values=self.values, **kwargs)

    def pyvista_mesh(self, **kwargs):
        mesh = self.mesh.pyvista_mesh(**kwargs)
        mesh.cell_arrays[self.values.label] = self.values.flatten()

        return mesh

    def to_vtk(self, filename):
        mesh = self.pyvista_mesh()
        mesh.save(filename)

    def resample(self, dx, dy):
        mesh, values = self.mesh.resample(dx, dy, self.values, kind='cubic')

        return Model(mesh, values)