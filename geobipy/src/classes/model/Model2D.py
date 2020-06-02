""" @Model2D_Class
Module describing a 2 Dimensional Model with two axes
"""

import numpy as np
from ...classes.core import StatArray
from .Model import Model

""" @Histogram_Class
Module describing an efficient histogram class
"""
from .baseDistribution import baseDistribution
from ...classes.statistics.Histogram1D import Histogram1D
from ...classes.mesh.RectilinearMesh2D import RectilinearMesh2D
from ...classes.core import StatArray
from ...base import customPlots as cP
from ...base.customFunctions import _logSomething
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Model2D(Model):
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

    def __init__(self, mesh, values=None, name=None, units=None):
        """ Instantiate a 2D histogram """
        if (mesh is None):
            return
        # Instantiate the parent class
        self._mesh = mesh
        # Assign the values
        if values is None:
            self._values = StatArray.StatArray([self.y.nCells, self.x.nCells], name=name, units=units)
        else:
            self._values = StatArray.StatArray(values, name, units)


    @property
    def mesh(self):
        return self._mesh


    def axisMean(self, log=None, axis=0):
        """Gets the mean along the given axis.

        This is not the true mean of the original samples. It is the best estimated mean using the binned counts multiplied by the axis bin centres.

        Parameters
        ----------
        axis : int
            Axis to take the mean along.

        Returns
        -------
        out : geobipy.StatArray
            The means along the axis.

        """

        if axis == 0:
            t = np.sum(np.repeat(self.x.cellCentres[np.newaxis, :], self.y.nCells, 0) * self._values, 1)
        else:
            t = np.sum(np.repeat(self.y.cellCentres[:, np.newaxis], self.x.nCells, 1) * self._values, 0)
        s = self._values.sum(axis = 1 - axis)

        i = np.where(s > 0.0)[0]
        tmp = np.zeros(t.size)
        tmp[i] = t[i] / s[i]

        if log:
            tmp, dum = _logSomething(tmp, log=log)

        return tmp


    def axisMedian(self, log=None, axis=0):
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
        return super().axisMedian(self._values, log, axis)


    def axisOpacity(self, percent=95.0, axis=0):
        """Return an opacity between 0 and 1 based on the difference between confidence invervals of the hitmap.

        Higher ranges in confidence map to less opaqueness.

        Parameters
        ----------
        percent : float
            Confidence percentage.
        axis : int
            Along which axis to obtain the interval locations.

        Returns
        -------
        out : array_like
            Opacity along the axis.

        """

        return 1.0 - self.axisTransparency(percent, axis)


    def axisPercentage(self, percent, log=None, axis=0):
        """Gets the percentage of the CDF for the specified axis.

        Parameters
        ----------
        percent : float
            Confidence percentage.
        log : 'e' or float, optional
            Take the log to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the CDF percentage.

        Returns
        -------
        out : array_like
            The CDF percentage along the specified axis. Has size equal to arr.shape[axis].

        """

        total = self._values.sum(axis=1-axis)
        p = 0.01 * percent
        tmp = np.cumsum(self._values, axis=1-axis)

        if axis == 0:
            tmp = tmp / np.repeat(total[:, np.newaxis], self._values.shape[1], 1)
        else:
            tmp = tmp / np.repeat(total[np.newaxis, :], self._values.shape[0], 0)

        ix2 = np.apply_along_axis(np.searchsorted, 1-axis, tmp, p)

        if axis == 0:
            out = self.x.cellCentres[ix2]
        else:
            out = self.y.cellCentres[ix2]

        if (not log is None):
            out, dum = _logSomething(out, log=log)

        return out


    def axisTransparency(self, percent=95.0, axis=0):
        """Return a transparency value between 0 and 1 based on the difference between confidence invervals of the hitmap.

        Higher ranges in confidence are mapped to more transparency.

        Parameters
        ----------
        percent : float
            Confidence percentage.
        axis : int
            Along which axis to obtain the interval locations.

        Returns
        -------
        out : array_like
            Transparency along the axis.

        """

        out = self.confidenceRange(percent=percent)
        maxes = np.max(out)
        if (maxes == 0.0): return out
        out /= maxes
        return out


    def comboPlot(self, **kwargs):
        """Combination plot using the 2D histogram and two axis histograms

        """

        self.gs = gridspec.GridSpec(5, 5)
        self.gs.update(wspace=0.3, hspace=0.3)
        plt.subplot(self.gs[1:, :4])
        self.pcolor(noColorbar = True, **kwargs)

        ax = plt.subplot(self.gs[:1, :4])
        h = self.axisHistogram(0).plot(**kwargs)
        plt.xlabel(''); plt.ylabel('')
        plt.xticks([]); plt.yticks([])
        ax.spines["left"].set_visible(False)

        ax = plt.subplot(self.gs[1:, 4:])
        h = self.axisHistogram(0).plot(rotate=True, **kwargs)
        plt.ylabel(''); plt.xlabel('')
        plt.yticks([]); plt.xticks([])
        ax.spines["bottom"].set_visible(False)



    def intervalMean(self, intervals, axis=0, statistic='mean'):
        """Compute the mean of an array between the intervals given along dimension dim.

        Returns
        -------
        out : geobipy.Histogram2D
            2D histogram with the new intervals.

        See Also
        --------
        geobipy.RectilinearMesh2D.intervalMean : for parameter information
        scipy.stats.binned_statistic : for more information

        """

        counts = super().intervalMean(self._values, intervals, axis, statistic)
        if axis == 0:
            out = Histogram2D(xBins = self.x.cellEdges, yBins = StatArray.StatArray(np.asarray(intervals), name=self.y.name(), units=self.y.units()))
            out._values[:] = counts
        else:
            out = Histogram2D(xBins = StatArray.StatArray(np.asarray(intervals), name=self.x.name(), units=self.x.units()), yBins = self.y.cellEdges)
            out._values[:] = counts
        return out


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
            Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.

        """
        self._values.pcolor(x=self.x.cellEdges, y=self.y.cellEdges, **kwargs)



















