""" @Histogram_Class
Module describing an efficient histogram class
"""
from ...classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ...classes.core import StatArray
from ...base import customFunctions as cF
from ...base import customPlots as cP
from .Distribution import Distribution
from .mixNormal import mixNormal
from .mixStudentT import mixStudentT
import numpy as np
import matplotlib.pyplot as plt
import sys

from copy import deepcopy


class Histogram1D(RectilinearMesh1D):
    """1D Histogram class that updates efficiently.

    Fast updating relies on knowing the bins ahead of time.

    Histogram1D(values, bins, binCentres, log, relativeTo)

    Parameters
    ----------
    bins : geobipy.StatArray or array_like, optional
        Specify the bin edges for the histogram. Can be regularly or irregularly increasing.
    binCentres : geobipy.StatArray or array_like, optional
        Specify the bin centres for the histogram. Can be regular or irregularly sized.
    log : 'e' or float, optional
        Entries are given in linear space, but internally bins and values are logged.
        Plotting is in log space.
    relativeTo : float, optional
        If a float is given, updates will be relative to this value.

    Returns
    -------
    out : Histogram1D
        1D histogram

    """

    def __init__(self, bins=None, binCentres=None, log=None, relativeTo=0.0, values=None):
        """ Initialize a histogram """

        # Allow an null instantiation
        if (bins is None and binCentres is None):
            self._counts = None
            return

        # Initialize the parent class
        super().__init__(cellEdges=bins, cellCentres=binCentres, log=log, relativeTo=relativeTo)

        self._counts = StatArray.StatArray(self.nCells, 'Frequency', dtype=np.int64)

        if not values is None:
            self.update(values)


    def __getitem__(self, slic):
        """Slice into the class. """

        assert np.shape(slic) == (), ValueError("slic must have one dimension.")


        bins = super().__getitem__(slic).cellEdges

        out = Histogram1D()

        out._cellEdges = bins
        out.log = self.log
        out._relativeTo = self._relativeTo

        out._counts = self.counts[slic]
        return out


    @property
    def cdf(self):
        out = np.cumsum(self.counts)
        out = out / out[-1]
        return out

    @property
    def counts(self):
        return self._counts

    @property
    def bins(self):
        return self.cellEdges

    @bins.setter
    def bins(self, values):
        self.cellEdges = values

    @property
    def binCentres(self):
        return self.cellCentres

    @binCentres.setter
    def binCentres(self, values):
        self.cellCentres = values

    @property
    def nBins(self):
        return self.nCells

    @property
    def nSamples(self):
        return self._counts.sum()


    def __deepcopy__(self, memo):
        out = type(self)()
        out._cellCentres = self._cellCentres.deepcopy()
        out._cellEdges = self._cellEdges.deepcopy()
        out.isRegular = self.isRegular
        out.dx = self.dx
        out._counts = self._counts.deepcopy()
        out.log = self.log
        out._relativeTo = self._relativeTo

        return out


    def combine(self, other):
        """Combine two histograms together.

        Using the bin centres of other, finds the corresponding bins in self.  The counts of other are then added to self.

        Parameters
        ----------
        other : geobipy.Histogram1D
            A histogram to combine.

        """

        cc = other.cellCentres

        cc = cF._power(cc, other.log)

        iBin = self.cellIndex(cc, clip=True)
        self._counts[iBin] = self._counts[iBin] + other.counts


    def credibleIntervals(self, percent=95.0, log=None):
        """Gets the credible intervals.

        Parameters
        ----------
        percent : float
            Confidence percentage.
        log : 'e' or float, optional
            Take the log of the credible intervals to a base. 'e' if log = 'e', or a number e.g. log = 10.

        Returns
        -------
        med : array_like
            Contains the median. Has size equal to arr.shape[axis].
        low : array_like
            Contains the lower interval.
        high : array_like
            Contains the upper interval.

        """

        total = self._counts.sum()
        p = 0.01 * percent
        cs = np.cumsum(self._counts / total)

        ixM = np.searchsorted(cs, 0.5)
        ix1 = np.searchsorted(cs, (1.0 - p))
        ix2 = np.searchsorted(cs, p)

        x = self.bins.internalEdges()
        med = x[ixM]
        low = x[ix1]
        high = x[ix2]

        if (not log is None):
            med, dum = cF._log(med, log=log)
            low, dum = cF._log(low, log=log)
            high, dum = cF._log(high, log=log)

        return (med, low, high)


    def credibleRange(self, percent=95.0, log=None):
        m, l, h = self.credibleIntervals(percent, log)
        return h - l


    @property
    def entropy(self):
        pdf = self.pdf[self.pdf > 0.0]
        return StatArray.StatArray(-(pdf * np.log(np.abs(pdf))).sum(), "Entropy")


    @property
    def pdf(self):
        return StatArray.StatArray(np.divide(self.counts, np.sum(self.counts)), name='Density')


    def estimateStd(self, nSamples, **kwargs):
        return np.sqrt(self.estimateVariance(nSamples, **kwargs))


    def estimateVariance(self, nSamples, **kwargs):
        X = self.sample(nSamples, **kwargs)
        return np.var(X)


    def findPeaks(self, width, **kwargs):
        """Identify peaks in the histogram.

        See Also
        --------
        scipy.spatial.find_peaks

        """

        return find_peaks(self.estimatePdf(),  width=width, **kwargs)



    def fit_mixture(self, mixture_type='gaussian', nSamples=1e5, log=None, mean_bounds=None, variance_bounds=None, k=[1, 5], tolerance=0.05):
        """Uses Gaussian or StudentT mixture models to fit the histogram.

        Starts at the minimum number of clusters and adds clusters until the BIC decreases less than the tolerance.

        Parameters
        ----------
        nSamples

        log

        mean_bounds

        variance_bounds

        k : ints
            Two ints with starting and ending # of clusters

        tolerance

        """

        X = self.sample(nSamples)[:, None]

        return X.fit_mixture(mixture_type, log, mean_bounds, variance_bounds, k, tolerance)


    def fit_estimated_pdf(self, mixture='student_t', **kwargs):
        mixture = mixture.lower()
        if mixture == 'gaussian':
            mixture = mixNormal()
        elif mixture == 'student_t':
            mixture = mixStudentT()
        else:
            assert False, ValueError("method must be either 'gaussian' or 'student_t' ")

        log = kwargs.get('log', None)
        kwargs['variance_bound'] = self.estimateVariance(10000, log=log)

        mixture.fit_to_curve(x=self.binCentres, y=self.estimatePdf(), **kwargs)
        return mixture


    def _marginal_probability_pdfs(self, pdfs):
        """

        Parameters
        ----------
        pdfs : array_like
            nPdfs x nBins.

        """

        nDistributions = pdfs.shape[0]

        # Normalize by the sum of the pdfs
        s = np.sum(pdfs, axis=0)
        i = np.where(s > 0.0)
        normalizedPdfs = pdfs
        normalizedPdfs[:, i] = normalizedPdfs[:, i] / s[i]

        # Initialize the facies Model
        axisPdf = self.estimatePdf()

        marginalProbability = StatArray.StatArray(nDistributions, 'Marginal probability')
        marginalProbability = np.sum(axisPdf * normalizedPdfs, axis=1)

        return marginalProbability


    def sample(self, nSamples, log=None):
        """Generates samples from the histogram.

        A uniform distribution is used for each bin to generate samples.
        The number of samples generated per bin is scaled by the count for that bin using the requested number of samples.

        parameters
        ----------
        nSamples : int
            Number of samples to generate.

        Returns
        -------
        out : geobipy.StatArray
            The samples.

        """
        values = np.random.rand(np.int(nSamples)) * self.cdf[-1]
        values = np.interp(values, np.hstack([0, self.cdf]), self.bins)
        values, dum = cF._log(values, log)
        return values


    def update(self, values, clip=True, trim=False):
        """Update the histogram by counting the entry of values into the bins of the histogram.

        Updates the bin counts in the histogram using fast methods.
        The values are clipped so that values outside the bins are forced inside.
        Optionally, one can trim the values so that those outside the bins are not counted.

        Parameters
        ----------
        values : array_like
            Increments the count for the bin that each value falls into.
        trim : bool
            A negative index which would normally wrap will clip to 0 and self.bins.size instead.

        """
        values = np.ravel(values[~np.isnan(values)])
        iBin = np.atleast_1d(self.cellIndex(values, clip=clip, trim=trim))
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
        kwargs['y'] = self.bins
        return super().pcolor(self._counts, **kwargs)


    def plot(self, rotate=False, flipX=False, flipY=False, trim=True, normalize=False, **kwargs):
        """ Plots the histogram """

        ax = cP.hist(self.counts, self.bins, rotate=rotate, flipX=flipX, flipY=flipY, trim=trim, normalize=normalize, **kwargs)
        return ax


    def plot_as_line(self, rotate=False, flipX=False, flipY=False, trim=True, normalize=False, **kwargs):

        x = self.binCentres
        x1 = x
        if kwargs['xscale'] == 'log':
            x1 = np.log10(x1)

        if normalize:
            cnts = self.counts / np.trapz(self.counts, x = x1)
        else:
            cnts = self.counts

        ax = cnts.plot(x=x, **kwargs)

        if normalize:
            cP.ylabel('Density')


    def hdfName(self):
        """ Reprodicibility procedure """
        return('Histogram1D()')


    def createHdf(self, parent, myName, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()

        self._counts.createHdf(grp, 'counts', nRepeats=nRepeats, fillvalue=fillvalue)

        if not self.log is None:
            grp.create_dataset('log', data = self.log)

        self._cellEdges.toHdf(grp, 'bins')
        self.relativeTo.createHdf(grp, 'relativeTo', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        self._counts.writeHdf(parent, myName+'/counts', index=index)

        self.relativeTo.writeHdf(parent, myName+'/relativeTo', index=index)
        # self._cellEdges.writeHdf(parent, myName+'/bins', index=index)


    def toHdf(self, h5obj, myName):
        """ Write the StatArray to an HDF object
        h5obj: :An HDF File or Group Object.
        """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self._cellEdges.toHdf(grp, 'bins')
        self._counts.toHdf(grp, 'counts')
        if not self.log is None:
            grp.create_dataset('log', data = self.log)
        self.relativeTo.toHdf(grp, 'relativeTo')


    def fromHdf(self, grp, index=None):
        """ Reads in the object froma HDF file """

        try:
            item = grp.get('relativeTo')
            obj = eval(cF.safeEval(item.attrs.get('repr')))
            relativeTo = obj.fromHdf(item, index=index)
        except:
            relativeTo = 0.0

        item = grp.get('bins')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        bins = obj.fromHdf(item)

        if np.ndim(bins) == 2:
            bins = bins[0, :]

        item = grp.get('counts')
        obj = eval(cF.safeEval(item.attrs.get('repr')))

        if (index is None):
            counts = obj.fromHdf(item)
        else:
            counts = obj.fromHdf(item, index=index)

        try:
            log = np.asscalar(np.asarray(grp.get('log')))
        except:
            log = None

        if bins.shape[-1] == counts.shape[-1]:
            Hist = Histogram1D(binCentres = bins)
        else:
            Hist = Histogram1D(bins = bins)

        Hist._counts = counts
        Hist.log = log
        Hist.relativeTo = relativeTo

        return Hist


    def summary(self, out=False):

        msg = ("{}\n"
              "Bins: \n{}"
              "Counts:\n{}"
              "Values are logged to base {}\n"
              "Relative to: {}").format(type(self), RectilinearMesh1D.summary(self, True), self.counts.summary(True), self.log, self.relativeTo)

        return msg if out else print(msg)

