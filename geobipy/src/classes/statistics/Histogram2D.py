""" @Histogram_Class
Module describing an efficient histogram class
"""
from .baseDistribution import baseDistribution
from ...classes.statistics.Histogram1D import Histogram1D
from ...classes.mesh.RectilinearMesh2D import RectilinearMesh2D
from ...classes.core.StatArray import StatArray
from ...base import customPlots as cP
from ...base.customFunctions import _logSomething
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Histogram2D(RectilinearMesh2D):
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

    def __init__(self, xBins=None, xBinCentres=None, yBins=None, yBinCentres=None, values=None):
        """ Instantiate a 2D histogram """
        if (xBins is None and xBinCentres is None):
            return
        # Instantiate the parent class
        RectilinearMesh2D.__init__(self, xCentres=xBinCentres, xEdges=xBins, yCentres=yBinCentres, yEdges=yBins)
        # Point counts to self.arr to make variable names more intuitive
        self._counts = StatArray([self.y.nCells, self.x.nCells], name='Frequency', dtype=np.int64)

        # Add the incoming values as counts to the histogram
        if (not values is None):
            self.update(values)


    @property
    def xBins(self):
        return self.x.cellEdges

    @property
    def yBins(self):
        return self.y.cellEdges

    @property
    def xBinCentres(self):
        return self.x.cellCentres
    
    @property
    def yBinCentres(self):
        return self.y.cellCentres

    @property
    def counts(self):
        return self._counts


    def axisHistogram(self, axis=0):
        """Get the histogram along an axis

        Parameters
        ----------
        axis : int
            Axis along which to get the histogram.

        Returns
        -------
        out : geobipy.Histogram1D

        """

        s = np.sum(self._counts, axis=axis)
        if axis == 0:
            out = Histogram1D(bins = self.x.cellEdges)
        else:
            out = Histogram1D(bins = self.y.cellEdges)
        out._counts += s
        return out

    
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
            t = np.sum(np.repeat(self.x.cellCentres[np.newaxis, :], self.y.nCells, 0) * self._counts, 1)
        else:
            t = np.sum(np.repeat(self.y.cellCentres[:, np.newaxis], self.x.nCells, 1) * self._counts, 0)
        s = self._counts.sum(axis = 1 - axis)

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
        return super().axisMedian(self._counts, log, axis)


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

        total = self._counts.sum(axis=1-axis)
        p = 0.01 * percent
        tmp = np.cumsum(self._counts, axis=1-axis)

        if axis == 0:
            tmp = tmp / np.repeat(total[:, np.newaxis], self._counts.shape[1], 1)
        else:
            tmp = tmp / np.repeat(total[np.newaxis, :], self._counts.shape[0], 0)

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


    def confidenceIntervals(self, percent=95.0, log=None, axis=0):
        """Gets the confidence intervals for the specified axis.
        
        Parameters
        ----------
        percent : float
            Confidence percentage.
        log : 'e' or float, optional
            Take the log of the confidence intervals to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the interval locations.

        Returns
        -------
        med : array_like
            Contains the locations of the medians along the specified axis. Has size equal to arr.shape[axis].
        low : array_like
            Contains the locations of the lower interval along the specified axis. Has size equal to arr.shape[axis].
        high : array_like
            Contains the locations of the upper interval along the specified axis. Has size equal to arr.shape[axis].

        """

        total = self._counts.sum(axis=1-axis)
        p = 0.01 * percent
        tmp = np.cumsum(self._counts, axis=1-axis)

        if axis == 0:
            tmp = tmp / np.repeat(total[:, np.newaxis], self._counts.shape[1], 1)
        else:
            tmp = tmp / np.repeat(total[np.newaxis, :], self._counts.shape[0], 0)

        ixM = np.apply_along_axis(np.searchsorted, 1-axis, tmp, 0.5)
        ix1 = np.apply_along_axis(np.searchsorted, 1-axis, tmp, (1.0 - p))
        ix2 = np.apply_along_axis(np.searchsorted, 1-axis, tmp, p)

        if axis == 0:
            med = self.x.cellCentres[ixM]
            low = self.x.cellCentres[ix1]
            high = self.x.cellCentres[ix2]
        else:
            med = self.y.cellCentres[ixM]
            low = self.y.cellCentres[ix1]
            high = self.y.cellCentres[ix2]

        if (not log is None):
            med, dum = _logSomething(med, log=log)
            low, dum = _logSomething(low, log=log)
            high, dum = _logSomething(high, log=log)

        return (med, low, high)


    def confidenceRange(self, percent=95.0, log=None, axis=0):
        """ Get the range of confidence with depth """
        sMed, sLow, sHigh = self.confidenceIntervals(percent, log=log, axis=axis)

        return sHigh - sLow


    def create2DjointProbabilityDistribution(self, H1, H2):
        """Given two histograms each of a single variable, regrid them to the
        same number of bins if necessary and take their outer product to obtain
         a 2D joint probability density """
        assert H1.bins.size == H2.bins.size, "Cannot do unequal bins yet"
        assert isinstance(H1, Histogram1D), TypeError("H1 must be a Histogram1D")
        assert isinstance(H2, Histogram1D), TypeError("H2 must be a Histogram1D")

        self.__init__(x=H1.bins, y=H2.bins)
        self._counts[:,:] = np.outer(H1.counts, H2.counts)


    def divideBySum(self, axis):
        """Divide by the sum along an axis.
        
        Parameters
        ----------
        axis : int
            Axis to sum along and then divide by.
                
        """
        s = np.sum(self._counts, axis)
        if (axis == 0):
            self._counts /= np.repeat(s[np.newaxis, :], np.size(self._counts, axis), axis)
        else:
            self._counts /= np.repeat(s[:, np.newaxis], np.size(self._counts, axis), axis)


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

        counts = super().intervalMean(self._counts, intervals, axis, statistic)
        if axis == 0:
            out = Histogram2D(xBins = self.x.cellEdges, yBins = StatArray(np.asarray(intervals), name=self.y.name(), units=self.y.units()))
            out._counts[:] = counts
        else:
            out = Histogram2D(xBins = StatArray(np.asarray(intervals), name=self.x.name(), units=self.x.units()), yBins = self.y.cellEdges)
            out._counts[:] = counts
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
        self._counts.pcolor(x=self.x.cellEdges, y=self.y.cellEdges, **kwargs)


    def plotConfidenceIntervals(self, percent=95.0, log=None, axis=0, **kwargs):

        med, low, high = self.confidenceIntervals(percent, log, axis)

        c = kwargs.pop('color', '#5046C8')
        ls = kwargs.pop('linestyle', 'dashed')
        lw = kwargs.pop('linewidth', 2)
        a = kwargs.pop('alpha', 0.6)

        if axis == 0:
            cP.plot(low, self.y.cellCentres, color=c, linestyle=ls, linewidth=lw, alpha=a, **kwargs)
            cP.plot(high, self.y.cellCentres, color=c, linestyle=ls, linewidth=lw, alpha=a, **kwargs)
        else:
            cP.plot(self.x.cellCentres, low, color=c, linestyle=ls, linewidth=lw, alpha=a, **kwargs)
            cP.plot(self.x.cellCentres, high, color=c, linestyle=ls, linewidth=lw, alpha=a, **kwargs)

    
    def plotMean(self, log=None, axis=0, **kwargs):

        m = self.axisMean(log=log, axis=axis)

        if axis == 0:
            cP.plot(m, self.y.cellCentres, **kwargs)
        else:
            cP.plot(self.x.cellCentres, m, **kwargs)


    def plotMedian(self, log=None, axis=0, **kwargs):

        m = self.axisMedian(log=log, axis=axis)

        if axis == 0:
            cP.plot(m, self.y.cellCentres, **kwargs)
        else:
            cP.plot(self.x.cellCentres, m, **kwargs)


    def update(self, values, clip=False):
        """Update the histogram by counting the entry of values into the bins of the histogram.
        
        Parameters
        ----------
        values : array_like
            Increments the count for the bin that each value falls into. Array should be 2D with first dimension == 2.
        
        """
      
        ixBin = self.x.cellIndex(values[0, :], clip=clip)
        # xTmp = np.bincount(ixBin, minlength = self.x.nCells-1)
        iyBin = self.y.cellIndex(values[1, :], clip=clip)
        # yTmp = np.bincount(iyBin, minlength = self.y.nCells-1)

        for a, b in zip(ixBin, iyBin):       
            self._counts[b, a] += 1



















