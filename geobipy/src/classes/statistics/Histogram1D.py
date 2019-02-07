""" @Histogram_Class
Module describing an efficient histogram class
"""
from .baseDistribution import baseDistribution
from ...classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ...classes.core.StatArray import StatArray
from ...base.customFunctions import safeEval
from ...base import customPlots as cP
import numpy as np
import matplotlib.pyplot as plt
import sys

class Histogram1D(RectilinearMesh1D):
    """1D Histogram class that can update and plot efficiently.
    
    Fast updating relies on knowing the bins ahead of time.

    Histogram1D(values, nBins, bins, name, units)

    Parameters
    ----------
    bins : array_like, optional
        Specify the bins for the histogram. Can be regular or irregularly sized.
    values : array_like, optional
        Initial values to input into the histogram.
    nBins : int, optional
        Used if bins is not given. Used to divide the range of values.  
    name : str, optional
        Name of what the histogram characterizes.
    units : str, optional
        Units of what the histogram characterizes.

    Returns
    -------
    out : Histogram1D
        1D histogram

    Raises
    ------
    ValueError
        If bins is not given, values must be specified.
    ValueError
        If bins is not given, nbins must be specified.
    
    """

    def __init__(self, bins=None, binCentres=None):
        """ Initialize a histogram """

        # Allow an null instantiation
        if (bins is None and binCentres is None):
            return

        # Initialize the parent class
        RectilinearMesh1D.__init__(self, cellEdges=bins, cellCentres=binCentres)

        self._counts = StatArray(self.nCells, 'Frequency', dtype=np.int64)


    @property
    def counts(self):
        return self._counts

    
    @property
    def bins(self):
        return self.cellEdges


    @property
    def binCentres(self):
        return self.cellCentres


    @property
    def nBins(self):
        return self.nCells


    def update(self, values, clip=False):
        """Update the histogram by counting the entry of values into the bins of the histogram.
        
        Parameters
        ----------
        values : array_like
            Increments the count for the bin that each value falls into.
        clip : bool
            A negative index which would normally wrap will clip to 0 and self.bins.size instead.
        """
        iBin = np.atleast_1d(self.cellIndex(values.flatten(), clip=clip))
        tmp = np.bincount(iBin, minlength = self.nBins)
        
        self._counts += tmp


    def pcolor(self, **kwargs):
        """pcolor the histogram

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
            Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.   
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.

        See Also
        --------
        geobipy.customPlots.pcolor : For additional keywords

        """
        return super().pcolor(self._counts, **kwargs)
        

    def plot(self, rotate=False, flipX=False, flipY=False, trim=True, normalize=False, **kwargs):
        """ Plots the histogram """

        cP.hist(self.counts, self.bins, rotate=rotate, flipX=flipX, flipY=flipY, trim=trim, normalize=normalize, **kwargs)


    def hdfName(self):
        """ Reprodicibility procedure """
        return('Histogram1D()')


    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.bins.toHdf(grp, 'bins')
        self._counts.createHdf(grp, 'counts', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        self._counts.writeHdf(parent, myName+'/counts', index=index)


    def toHdf(self, h5obj, myName):
        """ Write the StatArray to an HDF object
        h5obj: :An HDF File or Group Object.
        """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.bins.toHdf(grp, 'bins')
        self._counts.toHdf(grp, 'counts')


    def fromHdf(self, grp, index=None):
        """ Reads in the object froma HDF file """
        item = grp.get('bins')
        obj = eval(safeEval(item.attrs.get('repr')))
        bins = obj.fromHdf(item)

        item = grp.get('counts')
        obj = eval(safeEval(item.attrs.get('repr')))
        if (index is None):
            counts = obj.fromHdf(item)
        else:
            counts = obj.fromHdf(item, index=np.s_[index,:])

        if bins.size == counts.size:
            Hist = Histogram1D(binCentres = bins)
        else:
            Hist = Histogram1D(bins = bins)

        Hist._counts = counts
        return Hist


    def summary(self):
        RectilinearMesh1D.summary(self)
        self.counts.summary()