""" @Model2D_Class
Module describing a 2 Dimensional Model with two axes
"""

import numpy as np
from ..core import StatArray
from .Model import Model

""" @Histogram_Class
Module describing an efficient histogram class
"""
from ..statistics.baseDistribution import baseDistribution
from ..statistics.Histogram1D import Histogram1D
from ..mesh.RectilinearMesh2D import RectilinearMesh2D
from ..mesh.TopoRectilinearMesh2D import TopoRectilinearMesh2D
from ...base import plotting as cP
from ...base.utilities import _log
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Model2D(Model):
    """2D Model class.

    Class extension to the [Topo]RectilinearMesh2D.

    Model2D(x, y, name, units)

    Parameters
    ----------
    x : array_like or geobipy.RectilinearMesh1D, optional
        If array_like, defines the centre x locations of each element of the 2D hitmap array.
    y : array_like or geobipy.RectilinearMesh1D, optional
        If array_like, defines the centre y locations of each element of the 2D hitmap array.
    name : str
        Name of the hitmap array, default is 'Frequency'.
    units : str
        Units of the hitmap array, default is none since counts are unitless.

    Returns
    -------
    out : Histogram2D
        2D histogram

    """

    def __init__(self, mesh, values=None, name=None, units=None):
        """ Instantiate a 2D histogram """
        if (mesh is None):
            return
        # Instantiate the parent class
        self._mesh = mesh
        # Assign the values
        if values is None:
            self._values = StatArray.StatArray(self.mesh.shape, name=name, units=units)
        else:
            self._values = StatArray.StatArray(values, name, units)


    @property
    def mesh(self):
        return self._mesh


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
        self.mesh.pcolor(x=self.x.edges, y=self.y.edges, values=values, **kwargs)



















