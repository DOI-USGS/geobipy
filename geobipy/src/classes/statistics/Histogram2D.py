""" @Histogram_Class
Module describing an efficient histogram class
"""
from copy import deepcopy
from ...classes.statistics.Histogram1D import Histogram1D
from ...classes.mesh.RectilinearMesh2D import RectilinearMesh2D
from ...classes.core import StatArray
from ...base import customPlots as cP
from ...base import customFunctions as cF
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

    def __init__(self, xBins=None, xBinCentres=None, yBins=None, yBinCentres=None, zBins=None, zBinCentres=None, values=None):
        """ Instantiate a 2D histogram """
        if (xBins is None and xBinCentres is None):
            return
        # Instantiate the parent class
        RectilinearMesh2D.__init__(self, xCentres=xBinCentres, xEdges=xBins, yCentres=yBinCentres, yEdges=yBins, zCentres=zBinCentres, zEdges=zBins)

        # Point counts to self.arr to make variable names more intuitive
        self._counts = StatArray.StatArray([self.y.nCells, self.x.nCells], name='Frequency', dtype=np.int64)

        # Add the incoming values as counts to the histogram
        if (not values is None):
            self.update(values)


    @property
    def xBins(self):
        return self.x.cellEdges

    @property
    def xBinCentres(self):
        return self.x.cellCentres

    @property
    def yBins(self):
        return self.y.cellEdges
    
    @property
    def yBinCentres(self):
        return self.y.cellCentres

    @property
    def zBins(self):
        return self.z.cellEdges
    
    @property
    def zBinCentres(self):
        return self.z.cellCentres

    @property
    def counts(self):
        return self._counts


    def __getitem__(self, slic):
        """Allow slicing of the histogram.

        """
        assert np.shape(slic) == (2,), ValueError("slic must be over two dimensions.")
        if np.any(slic == 1):
            # 1D Histogram

            print(1)

        else:
            # 2D Histogram
            if self.xyz:
                out = Histogram2D(xBinCentres=self._x[slic[1]], yBinCentres=self._y[slic[1]], zBinCentres=self._z[slic[0]])
            else:
                out = Histogram2D(xBinCentres=self._x[slic[1]], yBinCentres=self._y[slic[0]])
            out._counts += self.counts[slic]

            return out


    def axisConfidenceIntervals(self, percent=95.0, log=None, axis=0):
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
        cs = np.cumsum(self._counts, axis=1-axis)

        if axis == 0:
            tmp = np.divide(cs, total[:, np.newaxis])
            # tmp = tmp / np.repeat(total[:, np.newaxis], self._counts.shape[1], 1)
        else:
            tmp = np.divide(cs, total[np.newaxis, :])
            # tmp = tmp / np.repeat(total[np.newaxis, :], self._counts.shape[0], 0)

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
            med, dum = cF._log(med, log=log)
            low, dum = cF._log(low, log=log)
            high, dum = cF._log(high, log=log)

        return (med, low, high)


    def axisConfidenceRange(self, percent=95.0, log=None, axis=0):
        """ Get the range of confidence with depth """
        sMed, sLow, sHigh = self.axisConfidenceIntervals(percent, log=log, axis=axis)

        return sHigh - sLow



    def axisHistogram(self, between=None, axis=0, log=None):
        """Get the histogram along an axis

        Parameters
        ----------
        intervals : array_like
            Array of size 2 containing lower and upper limits between which to count.
        axis : int
            Axis along which to get the histogram.
        log : 'e' or float, optional
            Entries are given in linear space, but internally bins and values are logged.
            Plotting is in log space.

        Returns
        -------
        out : geobipy.Histogram1D

        """
        assert 0 <= axis <= 1, ValueError("0 <= axis <= 1")

        bins = self.x if axis == 0 else self.y

        if intervals is None:
            s = np.sum(self._counts, axis=axis)
        else:
            assert np.size(intervals) == 2, ValueError("intervals must have size equal to 2")
            assert intervals[1] > intervals[0], ValueError("intervals must be monotonically increasing")
            if axis == 0:
                iBins = self.y.cellCentres.searchsorted(intervals)
                s = np.sum(self._counts[iBins[0]:iBins[1], :], axis=axis)
            else:
                iBins = self.x.cellCentres.searchsorted(intervals)
                s = np.sum(self._counts[:, iBins[0]:iBins[1]], axis=axis)
                
        out = Histogram1D(bins = bins.cellEdges, log=log)
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
            tmp, dum = cF._log(tmp, log=log)

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

    
    def axisMode(self, log=None, axis=0):
        """Gets the mode for the specified axis.
        
        Parameters
        ----------
        log : 'e' or float, optional
            Take the log of the mode to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the mode.

        Returns
        -------
        med : array_like
            Contains the modes along the specified axis. Has size equal to arr.shape[axis].

        """

        iMode = np.argmax(self._counts, axis = 1-axis)
    
        if axis == 0:
            mode = self.x.cellCentres[iMode]
        else:
            mode = self.z.cellCentres[iMode]

        if (not log is None):
            mode, dum = cF._log(mode, log=log)

        return mode


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


    def axisCdf(self, axis=0):

        total = self._counts.sum(axis=1-axis)
        cdf = np.cumsum(self._counts, axis=1-axis)

        if axis == 0:
            cdf = cdf / np.repeat(total[:, np.newaxis], self._counts.shape[1], 1)
        else:
            cdf = cdf / np.repeat(total[np.newaxis, :], self._counts.shape[0], 0)

        return cdf


    def axisPdf(self, axis=0):

        total = self._counts.sum(axis=1-axis)

        if axis == 0:
            pdf = (self._counts.T / total).T
        else:
            pdf = self._counts /  total

        return pdf


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

        tmp = self.axisCdf(axis)

        p = 0.01 * percent

        ix2 = np.apply_along_axis(np.searchsorted, 1-axis, tmp, p)

        if axis == 0:
            out = self.x.cellCentres[ix2]
        else:
            out = self.y.cellCentres[ix2]

        if (not log is None):
            out, dum = cF._log(out, log=log)

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

        out = self.axisConfidenceRange(percent=percent, axis=axis)
        maxes = np.max(out)
        if (maxes == 0.0): return out
        out /= maxes
        return out


    def comboPlot(self, **kwargs):
        """Combination plot using the 2D histogram and two axis histograms

        """

        self.gs = gridspec.GridSpec(5, 5)
        self.gs.update(wspace=0.3, hspace=0.3)
        ax = [plt.subplot(self.gs[1:, :4])]
        self.pcolor(noColorbar = True, **kwargs)

        ax.append(plt.subplot(self.gs[:1, :4]))
        h = self.axisHistogram(axis=0).plot()
        plt.xlabel(''); plt.ylabel('')
        plt.xticks([]); plt.yticks([])
        ax[-1].spines["left"].set_visible(False)

        ax.append(plt.subplot(self.gs[1:, 4:]))
        h = self.axisHistogram(axis=0).plot(rotate=True)
        plt.ylabel(''); plt.xlabel('')
        plt.yticks([]); plt.xticks([])
        ax[-1].spines["bottom"].set_visible(False)


    def create2DjointProbabilityDistribution(self, H1, H2):
        """Create 2D joint distribution from two Histograms.
        
        Given two histograms each of a single variable, regrid them to the 
        same number of bins if necessary and take their outer product to obtain
        a 2D joint probability density.

        Parameters
        ----------
        H1 : geobipy.Histogram1D
            First histogram.
        H2 : geobipy.Histogram1D
            Second histogram.
         
        """
        assert H1.bins.size == H2.bins.size, "Cannot do unequal bins yet"
        assert isinstance(H1, Histogram1D), TypeError("H1 must be a Histogram1D")
        assert isinstance(H2, Histogram1D), TypeError("H2 must be a Histogram1D")

        self.__init__(x=H1.bins, y=H2.bins)
        self._counts[:,:] = np.outer(H1.counts, H2.counts)

    
    def deepcopy(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        """ Define the deepcopy. """

        if self.xyz:
            out = Histogram2D(xBins=self.xBins, yBins=self.yBins, zBins=self.zBins)
        else:
            out = Histogram2D(xBins=self.xBins, yBins=self.yBins)
        out._counts = self._counts.deepcopy()

        return out


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


    def fitMajorPeaks(self, intervals, axis=0, **kwargs):
        """Find peaks in the histogram along an axis.

        Parameters
        ----------
        intervals : array_like, optional
            Accumulate the histogram between these invervals before finding peaks
        axis : int, optional
            Axis along which to find peaks.

        """
        counts = super().intervalStatistic(self._counts, intervals, axis, 'sum')

        distributions = []
        if axis == 0:
            h = Histogram1D(bins = self.xBins)
            for i in range(intervals.size - 1):
                h._counts = counts[i, :]
                distributions.append(h.fitMajorPeaks(**kwargs))
        else:
            h = Histogram1D(bins = self.yBins)
            for i in range(intervals.size - 1):
                h._counts = counts[:, i]
                distributions.append(h.fitMajorPeaks(**kwargs))

        return distributions

    def intervalStatistic(self, intervals, axis=0, statistic='mean'):
        """Compute the statistic of an array between the intervals given along dimension dim.

        Returns
        -------
        out : geobipy.Histogram2D
            2D histogram with the new intervals.
        
        See Also
        --------
        geobipy.RectilinearMesh2D.intervalMean : for parameter information
        scipy.stats.binned_statistic : for more information

        """

        counts = super().intervalStatistic(self._counts, intervals, axis, statistic)

        if axis == 0:
            out = Histogram2D(xBins = self.x.cellEdges, yBins = StatArray.StatArray(np.asarray(intervals), name=self.y.name(), units=self.y.units()))
            out._counts[:] = counts
        else:
            out = Histogram2D(xBins = StatArray.StatArray(np.asarray(intervals), name=self.x.name(), units=self.x.units()), yBins = self.y.cellEdges)
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
        ax = self._counts.pcolor(x=self.x.cellEdges, y=self.y.cellEdges, **kwargs)
        return ax


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


    def update(self, xValues, yValues=None, trim=False):
        """Update the histogram by counting the entry of values into the bins of the histogram.
        
        Parameters
        ----------
        xValues : array_like
            * If xValues is 1D, yValues must be given.
            * If xValues is 2D, the first dimension must have size = 2. yValues will be ignored.
        yValues : array_like, optional
            Added to the second dimension of the histogram
            Ignored if xValues is 2D.
        clip : bool
            Values outside the histogram axes are clipped to the upper and lower bins.
        
        """

        if yValues is None:
            assert xValues.ndim == 2, ValueError("If yValues is not given, xValues must be 2D.")
            assert np.shape(xValues)[0] == 2, ValueError("xValues must have first dimension with size 2.")

            yValues = xValues[1, :]
            xValues = xValues[0, :]

      
        iBin = self.cellIndices(xValues, yValues, clip=True, trim=trim)

        unique, counts = np.unique(iBin, axis=1, return_counts=True)

        self._counts[unique[0], unique[1]] += counts



















