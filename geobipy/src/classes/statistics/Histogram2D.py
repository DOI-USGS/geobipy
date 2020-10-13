""" @Histogram_Class
Module describing an efficient histogram class
"""
from copy import deepcopy
from ...classes.statistics.Histogram1D import Histogram1D
from ...classes.mesh.RectilinearMesh2D import RectilinearMesh2D
from ...classes.core import StatArray
from ...base import customPlots as cP
from ...base import customFunctions as cF
from .baseDistribution import baseDistribution
from .Mixture import Mixture
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import progressbar


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

    def __init__(self, xBins=None, xBinCentres=None, yBins=None, yBinCentres=None, **kwargs):
        """ Instantiate a 2D histogram """
        if (xBins is None and xBinCentres is None):
            return

        # Instantiate the parent class
        RectilinearMesh2D.__init__(self, xCentres=xBinCentres, xEdges=xBins, yCentres=yBinCentres, yEdges=yBins, **kwargs)

        # Point counts to self.arr to make variable names more intuitive
        self._counts = StatArray.StatArray([self.y.nCells, self.x.nCells], name='Frequency', dtype=np.int64)


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

        slic0 = slic
        if isinstance(slic[0], int):
            if isinstance(slic[1].stop, int):
                s = slic[1]
                slic = (slic[0], slice(s.start, s.stop+1, s.step))
            out = Histogram1D(bins=self._x.cellEdges[slic[1]])
            out._counts += np.squeeze(self.counts[slic0])

        elif isinstance(slic[1], int):
            if isinstance(slic[0].stop, int):
                s = slic[0]
                slic = (slice(s.start, s.stop+1, s.step), slic[1])
            out = Histogram1D(bins=self._y.cellEdges[slic[0]])
            out._counts += np.squeeze(self.counts[slic0])

        else:
            if isinstance(slic[0].stop, int):
                s = slic[0]
                slic = (slice(s.start, s.stop+1, s.step), slic[1])
            if isinstance(slic[1].stop, int):
                s = slic[1]
                slic = (slic[0], slice(s.start, s.stop+1, s.step))
            # 2D Histogram
            if self.xyz:
                out = Histogram2D(xBins=self._x.cellEdges[slic[1]], yBins=self._y.cellEdges[slic[1]], zBins=self._z.cellEdges[slic[0]])
            else:
                out = Histogram2D(xBins=self._x.cellEdges[slic[1]], yBins=self._z.cellEdges[slic[0]])

            out._counts += self.counts[slic0]

        return out


    def credibleIntervals(self, percent=95.0, log=None, axis=0):
        """Gets the credible intervals for the specified axis.

        Parameters
        ----------
        percent : float
            Confidence percentage.
        log : 'e' or float, optional
            Take the log of the credible intervals to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the interval locations.

        Returns
        -------
        med : array_like
            Contains the medians along the specified axis. Has size equal to arr.shape[axis].
        low : array_like
            Contains the lower interval along the specified axis. Has size equal to arr.shape[axis].
        high : array_like
            Contains the upper interval along the specified axis. Has size equal to arr.shape[axis].

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


    def credibleRange(self, percent=95.0, log=None, axis=0):
        """ Get the range of credibility with depth

        Parameters
        ----------
        percent : float
            Percent of the credible intervals
        log : 'e' or float, optional
            If None: The range is the difference in linear space of the credible intervals
            If 'e' or float: The range is the difference in log space, or ratio in linear space.
        axis : int
            Axis along which to get the marginal histogram.


        """
        sMed, sLow, sHigh = self.credibleIntervals(percent, log=log, axis=axis)

        return sHigh - sLow



    def marginalize(self, intervals=None, index=None, log=None, axis=0):
        """Get the marginal histogram along an axis

        Parameters
        ----------
        intervals : array_like
            Array of size 2 containing lower and upper limits between which to count.
        log : 'e' or float, optional
            Entries are given in linear space, but internally bins and values are logged.
            Plotting is in log space.
        axis : int
            Axis along which to get the marginal histogram.

        Returns
        -------
        out : geobipy.Histogram1D

        """
        assert 0 <= axis <= 1, ValueError("0 <= axis <= 1")


        bins = self.x if axis == 0 else self.y

        if intervals is None and index is None:
            s = np.sum(self._counts, axis=axis)
        else:
            assert (intervals is None) or (index is None), ValueError("Cannot provide both intervals and an index")

            if not intervals is None:
                assert np.size(intervals) == 2, ValueError("intervals must have size equal to 2")
                assert intervals[1] > intervals[0], ValueError("intervals must be monotonically increasing")
                if axis == 0:
                    indices = self.y.cellCentres.searchsorted(intervals)
                else:
                    indices = self.x.cellCentres.searchsorted(intervals)
            else:
                indices = np.asarray([index, index+1])

            if axis == 0:
                s = np.sum(self._counts[indices[0]:indices[1], :], axis=axis)
            else:
                s = np.sum(self._counts[:, indices[0]:indices[1]], axis=axis)

        out = Histogram1D(bins = bins.cellEdges, log=log)
        out._counts += s
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
        return super().median(self._counts, log, axis)


    def mode(self, log=None, axis=0):
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


    def opacity(self, percent=95.0, log=None, axis=0):
        """Return an opacity between 0 and 1 based on the difference between credible invervals of the hitmap.

        Higher ranges in credibility map to less opaqueness.

        Parameters
        ----------
        percent : float, optional.
            Confidence percentage.
        log : 'e' or float, optional.
            If None: Take the difference in credible intervals.
            Else: Take the ratio of the credible intervals.
        axis : int, optional.
            Along which axis to obtain the interval locations.

        Returns
        -------
        out : array_like
            Opacity along the axis.

        """

        return 1.0 - self.transparency(percent=percent, log=log, axis=axis)


    def cdf(self, axis=0):

        total = self._counts.sum(axis=1-axis)
        cdf = np.cumsum(self._counts, axis=1-axis)

        if axis == 0:
            cdf = cdf / np.repeat(total[:, np.newaxis], self._counts.shape[1], 1)
        else:
            cdf = cdf / np.repeat(total[np.newaxis, :], self._counts.shape[0], 0)

        return cdf

    def entropy(self, axis=None):

        pdf = self.pdf(axis=axis)
        pdf = pdf[pdf > 0.0]
        return StatArray.StatArray(-(pdf * np.log(np.abs(pdf))).sum(), "Entropy")


    def pdf(self, axis=None):

        if axis is None:
            out = StatArray.StatArray(np.divide(self._counts, np.sum(self._counts)), 'Probability density')
            out[np.isnan(out)] = 0.0
            return out

        total = self._counts.sum(axis=1-axis)

        if axis == 0:
            out = StatArray.StatArray(np.divide(self._counts.T, total).T, 'Probability density')
        else:
            out = StatArray.StatArray(np.divide(self._counts, total), 'Probability density')
        out[np.isnan(out)] = 0.0
        return out


    def percentage(self, percent, log=None, axis=0):
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

        tmp = self.cdf(axis)

        p = 0.01 * percent

        ix2 = np.apply_along_axis(np.searchsorted, 1-axis, tmp, p)

        if axis == 0:
            out = self.x.cellCentres[ix2]
        else:
            out = self.y.cellCentres[ix2]

        if (not log is None):
            out, dum = cF._log(out, log=log)

        return out


    def transparency(self, percent=95.0, log=None, axis=0):
        """Return a transparency value between 0 and 1 based on the difference between credible invervals of the hitmap.

        Higher ranges in credibility are mapped to more transparency.

        Parameters
        ----------
        percent : float
            Confidence percentage.
        log : 'e' or float, optional.
            If None: Take the difference in credible intervals.
            Else: Take the ratio of the credible intervals.
        axis : int
            Along which axis to obtain the interval locations.

        Returns
        -------
        out : array_like
            Transparency along the axis.

        """

        out = self.credibleRange(percent=percent, log=log, axis=axis)
        mn = np.nanmin(out)
        mx = np.nanmax(out)
        t = mx - mn
        if t > 0.0:
            return (out - mn) / t
        else:
            return out - mn


    def comboPlot(self, **kwargs):
        """Combination plot using the 2D histogram and two axis histograms

        """

        self.gs = gridspec.GridSpec(5, 5)
        self.gs.update(wspace=0.3, hspace=0.3)
        ax = [plt.subplot(self.gs[1:, :4])]
        self.pcolor(noColorbar = True, **kwargs)

        ax.append(plt.subplot(self.gs[:1, :4]))
        h = self.marginalize(axis=0).plot()
        plt.xlabel(''); plt.ylabel('')
        plt.xticks([]); plt.yticks([])
        ax[-1].spines["left"].set_visible(False)

        ax.append(plt.subplot(self.gs[1:, 4:]))
        h = self.marginalize(axis=1).plot(rotate=True)
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


    def fit_mixture(self, intervals, axis=0, **kwargs):
        """Find peaks in the histogram along an axis.

        Parameters
        ----------
        intervals : array_like, optional
            Accumulate the histogram between these invervals before finding peaks
        axis : int, optional
            Axis along which to find peaks.

        """

        track = kwargs.pop('track', True)

        counts, intervals = super().intervalStatistic(self._counts, intervals, axis, 'sum')

        distributions = []
        active = []

        if track:
            bar = progressbar.ProgressBar()
            r = bar(range(np.size(intervals) - 1))
        else:
            r = range(np.size(intervals) - 1)

        if axis == 0:
            h = Histogram1D(bins = self.xBins)

            for i in r:
                h._counts[:] = counts[i, :]
                d, a = h.fit_mixture(**kwargs)
                distributions.append(d)
                active.append(a)

        else:
            h = Histogram1D(bins = self.yBins)
            for i in r:
                h._counts[:] = counts[:, i]
                d, a = h.fit_mixture(**kwargs)
                distributions.append(d)
                active.append(a)

        return distributions, active


    def fit_estimated_pdf(self, intervals=None, axis=0, mixture='student_t', **kwargs):
        """Find peaks in the histogram along an axis.

        Parameters
        ----------
        intervals : array_like, optional
            Accumulate the histogram between these invervals before finding peaks
        axis : int, optional
            Axis along which to find peaks.

        """

        track = kwargs.pop('track', True)
        if intervals is None:
            intervals = self.yBins if axis==0 else self.xBins
        else:
            assert np.size(intervals) >= 2, ValueError('intervals must have size >= 2')

        mixtures = []

        if track:
            Bar = progressbar.ProgressBar()
            r = Bar(range(np.size(intervals) - 1))
        else:
            r = range(np.size(intervals) - 1)

        for i in r:
            h = self.marginalize(intervals=intervals[i:i+2], axis=axis)
            ms = h.fit_estimated_pdf(mixture = mixture, **kwargs)
            mixtures.append(ms)

        return mixtures


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

        counts, intervals = super().intervalStatistic(self._counts, intervals, axis, statistic)

        if axis == 0:
            out = Histogram2D(xBins = self.x.cellEdges, yBins = StatArray.StatArray(np.asarray(intervals), name=self.y.name, units=self.y.units))
            out._counts[:] = counts
        else:
            out = Histogram2D(xBins = StatArray.StatArray(np.asarray(intervals), name=self.x.name, units=self.x.units), yBins = self.y.cellEdges)
            out._counts[:] = counts
        return out


    def marginalProbability(self, fractions, distributions, axis=0, reciprocateParameter=None, log=None, verbose=False, **kwargs):
        """Compute the marginal (joint) probability between the Histogram and a set of distributions.

        .. math::
            :label: marginal

            p(distribution_{i} | \\boldsymbol{v}) =


        """

        if not isinstance(distributions, list):
            distributions = [distributions]

        # If the distributions are univariate, and there is only one per class, its a '1D' problem
        assert isinstance(distributions[0], (baseDistribution, Mixture)), TypeError("Distributions must be geobipy distributions.")

        if isinstance(distributions[0], Mixture):
            return self._marginalProbability_1D_mixtures(mixtures=distributions, axis=axis, reciprocateParameter=reciprocateParameter, log=log, **kwargs)

        if distributions[0].multivariate:
            if reciprocateParameter is None:
                reciprocateParameter = [False, False]
            if log is None:
                log = [None, None]
            return self._marginalProbability_2D(fractions, distributions, axis=axis, reciprocateParameter=reciprocateParameter, log=log, verbose=verbose)
        else:
            if reciprocateParameter is None:
                reciprocateParameter = False
                return self._marginalProbability_1D(fractions, distributions, axis=axis, reciprocateParameter=reciprocateParameter, log=log)


    def _marginalProbability_1D(self, fractions, distributions, axis=0, reciprocateParameter=False, log=None):

        assert axis < 2, ValueError("Must have 0 <= axis < 2")
        nDistributions = np.size(distributions)
        assert np.size(fractions) == nDistributions, ValueError("Fractions must have same size as number of distributions")

        if axis == 0:
            ax = self.x.cellCentres
        else:
            ax = self.y.cellCentres

        if reciprocateParameter:
            ax = 1.0 / ax[::-1]

        ax, dum = cF._log(ax, log)

        # Compute the probabilities along the hitmap axis, using each distribution
        pdfs = np.zeros([nDistributions, ax.size])
        for i in range(nDistributions):
            pdfs[i, :] = fractions[i] * distributions[i].probability(ax, log=False)

        # Normalize by the sum of the pdfs
        normalizedPdfs = pdfs / np.sum(pdfs, axis=0)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(nDistributions):
        #     plt.subplot(nDistributions, 1, i+1)
        #     StatArray.StatArray(pdfs[i, :]).plot(x=ax)

        # Initialize the facies Model
        axisPdf = self.pdf(axis=axis)

        marginalProbability = StatArray.StatArray([nDistributions, self.shape[axis]], 'Marginal probability')
        for j in range(nDistributions):
            marginalProbability[j, :] = np.sum(axisPdf * normalizedPdfs[j, :], axis=1-axis)

        return np.squeeze(marginalProbability)


    def _marginalProbability_1D_mixtures(self, mixtures, axis=0, reciprocateParameter=False, log=None, **kwargs):

        assert axis < 2, ValueError("Must have 0 <= axis < 2")
        nMixtures = np.size(mixtures)
        # assert np.size(fractions) == nDistributions, ValueError("Fractions must have same size as number of distributions")

        maxDistributions = kwargs.pop('maxDistributions', np.max([mm.n_mixtures for mm in mixtures]))

        if axis == 0:
            ax = self.x.cellCentres
        else:
            ax = self.y.cellCentres

        if reciprocateParameter:
            ax = 1.0 / ax[::-1]

        ax, dum = cF._log(ax, log)

        # Compute the probabilities along the hitmap axis, using each distribution
        pdfs = np.zeros([maxDistributions, nMixtures, ax.size])

        if nMixtures != 1:
            for i in range(nMixtures):
                if mixtures[i].n_mixtures > 0:
                    pdfs[:mixtures[i].n_mixtures, i, :] = mixtures[i].probability(ax, log=False).T

        # Normalize by the sum of the pdfs
        normalizedPdfs = pdfs / np.sum(pdfs, axis=0)

        # Initialize the facies Model
        axisPdf = self.estimatePdf(axis=axis)

        marginalProbability = StatArray.StatArray([nMixtures, maxDistributions], 'Marginal probability')
        marginalProbability[:, :] = np.sum(axisPdf * normalizedPdfs, axis=2-axis).T

        # for in range(nMixtures):
        #     marginalProbability[j, :] = np.sum(axisPdf * normalizedPdfs[i, :, j], axis=1-axis)

        return marginalProbability


    def _marginalProbability_2D(self, fractions, distributions, axis=None, reciprocateParameter=[False, False], log=[None, None], verbose=False):

        assert np.size(reciprocateParameter) == 2, ValueError('reciprocateParameter must have 2 bools')
        assert np.size(log) == 2, ValueError('log must have size 2')
        if not axis is None:
            assert 1 <= axis <= 2, ValueError("axis must be 1 or 2")

        nDistributions = np.size(distributions)
        for d in distributions:
            assert d.multivariate and d.ndim == 2, TypeError("Each distribution must be multivariate with 2 dimensions.")

        assert np.size(fractions) == nDistributions, ValueError("Fractions must have same size as number of distributions")

        # Get the axes
        ax0 = self.y.cellCentres

        if reciprocateParameter[0]:
            ax0 = 1.0 / ax0[::-1]

        ax0, dum = cF._log(ax0, log[0])

        ax1 = self.x.cellCentres

        if reciprocateParameter[1]:
            ax1 = 1.0 / ax1[::-1]

        ax1, dum = cF._log(ax1, log[1])

        # Compute the 2D joint probability density function for each distribution
        class_xPdfs = np.zeros([nDistributions, self.shape[1]])
        class_yPdfs = np.zeros([nDistributions, self.shape[0]])
        for i, d in enumerate(distributions):
            class_yPdfs[i, :] = fractions[i] * d.probability(ax0, log=False, axis=0)
            class_xPdfs[i, :] = fractions[i] * d.probability(ax1, log=False, axis=1)


        histogram_xPdf = StatArray.StatArray(self.pdf(axis=1))
        histogram_yPdf = StatArray.StatArray(self.pdf(axis=0))

        P_class_given_v1v2 = StatArray.StatArray([nDistributions, self.shape[0] ,self.shape[1]], 'Marginal probability')
        for j in range(nDistributions):
            P_class_given_v1v2[j, :, :] = (class_xPdfs[j, :] * histogram_xPdf) * (class_yPdfs[j, :] * histogram_yPdf.T).T

        if axis is None:
            denominator = np.sum(P_class_given_v1v2, axis=0)
            marginalProbability = P_class_given_v1v2 / denominator
        else:
            p_tmp = np.sum(P_class_given_v1v2, axis=axis)
            denominator = np.sum(p_tmp, axis=0)
            marginalProbability = p_tmp / denominator

        return marginalProbability


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
            Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.

        """
        kwargs['trim'] = kwargs.pop('trim',  0.0)
        x = self.x.cellEdges
        y = self.y.cellEdges

        ax = StatArray.StatArray(self.counts).pcolor(x=x, y=y, **kwargs)
        return ax


    def plotCredibleIntervals(self, percent=95.0, log=None, axis=0, **kwargs):

        med, low, high = self.credibleIntervals(percent, log, axis)

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

        m = self.mean(log=log, axis=axis)

        if axis == 0:
            cP.plot(m, self.y.cellCentres, label='mean',  **kwargs)
        else:
            cP.plot(self.x.cellCentres, m, label='mean', **kwargs)


    def plotMedian(self, log=None, axis=0, **kwargs):

        m = self.median(log=log, axis=axis)

        if axis == 0:
            cP.plot(m, self.y.cellCentres, label='median', **kwargs)
        else:
            cP.plot(self.x.cellCentres, m, label='median', **kwargs)


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

        if np.ndim(iBin) == 1:
            unique = iBin
            counts = 1
        else:
            unique, counts = np.unique(iBin, axis=1, return_counts=True)

        self._counts[unique[0], unique[1]] += counts



















