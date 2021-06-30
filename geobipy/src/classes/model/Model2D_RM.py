""" @Histogram_Class
Module describing an efficient histogram class
"""
from copy import deepcopy
from ...classes.mesh.TopoRectilinearMesh2D import TopoRectilinearMesh2D
from ...classes.core import StatArray
from ...base import plotting as cP
from ...base import utilities as cF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Model2D_RM(TopoRectilinearMesh2D):
    """ 2D Histogram class that can update and plot efficiently.

    Class extension to the RectilinearMesh2D.  The mesh defines the x and y axes, while the Histogram2D manages the counts.

    Histogram2D(x, y, name, units)

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

    def __init__(self, xCentres=None, xEdges=None, yCentres=None, yEdges=None, zCentres=None, zEdges=None, heightCentres=None, heightEdges=None, values=None, **kwargs):
        """ Instantiate a 2D histogram """

        # Instantiate the parent class
        super().__init__(xCentres, xEdges, yCentres, yEdges, zCentres, zEdges, heightCentres, heightEdges, **kwargs)
        self.values = values


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


    def __getitem__(self, slic):
        """Allow slicing.

        """
        assert np.shape(slic) == (2,), ValueError("slic must be over 2 dimensions.")

        slic0 = slic

        slic = []
        axis = -1
        for i, x in enumerate(slic0):
            if not isinstance(x, int):
                if isinstance(x.stop, int):
                    tmp = slice(x.start, x.stop+1, x.step) # If a slice, add one to the end for bins.
            else:
                tmp = x
                axis = i


            slic.append(tmp)
        slic = tuple(slic)

        if axis == -1:
            if self.xyz:
                out = type(self)(xEdges=self.x.edges[slic[1]], yBins=self.y.edges[slic[1]], zBins=self.z.edges[slic[0]])
            else:
                out = type(self)(xBins=self.x.edges[slic[1]], yBins=self.z.edges[slic[0]])

            out._counts += self.values[slic0]
            return out

        if axis == 0:
            out = type(self)(bins=self.x.edges[slic[1]])
            out._counts += np.squeeze(self.values[slic0])
        elif axis == 1:
            out = type(self)(bins=self.y.edges[slic[0]])
            out._counts += np.squeeze(self.values[slic0])
        return out




    def mean(self, log=None, axis=0):
        """Gets the mean along the given axis.

        This is not the true mean of the original samples. It is the best estimated mean using the binned counts multiplied by the axis bin centres.

        Parameters
        ----------
        log : 'e' or float, optional.
            Take the log of the mean to base "log"
        axis : int
            Axis to take the mean along.

        Returns
        -------
        out : geobipy.StatArray
            The means along the axis.

        """

        return super().mean(self.values, log, axis)


    def median(self, log=None, axis=0):
        """Gets the median for the specified axis.

        Parameters
        ----------
        log : 'e' or float, optional
            Take the log of the median to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the median.

        Returns
        -------
        out : array_like
            The medians along the specified axis. Has size equal to arr.shape[axis].

        """
        return super().median(self.values, log, axis)




    def deepcopy(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        """ Define the deepcopy. """

        if self.xyz:
            out = type(self)(xEdges=self.xEdges, yEdges=self.yEdges, zEdges=self.Edges)
        else:
            out = type(self)(xEdges=self.xEdges, yEdges=self.yEdges)
        out._values = self.values.deepcopy()

        return out



    def plot(self, **kwargs):
        return self.pcolor(**kwargs)


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
        return super().pcolor(values=self.values, **kwargs)


    def createHdf(self, parent, name, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = super().createHdf(parent, name, withPosterior, nRepeats, fillvalue)
        self.values.createHdf(grp, 'values', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        return grp


    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        super().writeHdf(parent, name, withPosterior, index)
        self.values.writeHdf(parent, name+'/values',  withPosterior=withPosterior, index=index)


    def fromHdf(self, grp, index=None):
        """ Reads in the object from a HDF file """
        super().fromHdf(grp, index)
        self.values = StatArray.StatArray().fromHdf(grp['values'], index)
        return self


















