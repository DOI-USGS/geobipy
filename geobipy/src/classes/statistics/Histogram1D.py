""" @Histogram_Class
Module describing an efficient histogram class
"""
from ...classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ...classes.core import StatArray
from ...base import utilities as cF
from ...base import plotting as cP
from .Distribution import Distribution
from .mixNormal import mixNormal
from .mixPearson import mixPearson
from .mixStudentT import mixStudentT
import numpy as np
import matplotlib.pyplot as plt
import sys
from lmfit import model

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

    def __init__(self, centres=None, edges=None, widths=None, log=None, relativeTo=None, values=None, **kwargs):
        """ Initialize a histogram """

        if 'bins' in kwargs:
            raise Exception('woops')

        # Initialize the parent class
        RectilinearMesh1D.__init__(self, centres=centres, edges=edges, widths=widths, log=log, relativeTo=relativeTo)

        self._counts = StatArray.StatArray(self.nCells.value, 'Frequency', dtype=np.int64)

        if not values is None:
            self.update(values)


    def __getitem__(self, slic):
        """Slice into the class. """

        assert np.shape(slic) == (), ValueError("slic must have one dimension.")


        bins = RectilinearMesh1D.__getitem__(self, slic).edges

        out = Histogram1D()

        out._edges = bins
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
        return self.edges

    @bins.setter
    def bins(self, values):
        self.edges = values

    @property
    def binCentres(self):
        return self.centres

    @binCentres.setter
    def binCentres(self, values):
        self.centres = values

    @property
    def nBins(self):
        return self.nCells.value

    @property
    def nSamples(self):
        return self._counts.sum()

    def __deepcopy__(self, memo={}):
        out = RectilinearMesh1D.__deepcopy__(self, memo)
        # out = type(self)()
        # out._centres = self._centres.deepcopy()
        # out._edges = self._edges.deepcopy()
        # out.isRegular = self.isRegular
        # out.dx = self.dx
        out._counts = deepcopy(self._counts)
        # out.log = self.log
        # out._relativeTo = self._relativeTo

        return out

    def combine(self, other):
        """Combine two histograms together.

        Using the bin centres of other, finds the corresponding bins in self.  The counts of other are then added to self.

        Parameters
        ----------
        other : geobipy.Histogram1D
            A histogram to combine.

        """

        cc = other.centres

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

        x, dum = cF._log(self.binCentres, self.log)
        return StatArray.StatArray(np.divide(self.counts, np.trapz(self.counts, x = x)), name='Density')


    def estimateStd(self, nSamples, **kwargs):
        return np.sqrt(self.estimateVariance(nSamples, **kwargs))


    def estimateVariance(self, nSamples, **kwargs):
        X = self.sample(nSamples, **kwargs)
        return np.var(X)


    def compute_MinsleyFoksBedrosian2020_P_lithology(self, global_mixture, local_mixture, log=None):
        """Compute the cluster probability using Minsley Foks 2020.

        Compute the probability of clusters using both a global mixture model and a local mixture model fit to the histogram.
        In MinsleyFoksBedrosian2020, the local mixture models were generated by fitting the histogram's estimated pdf while the global mixture model
        is used to label all local mixture models on a dataset scale.

        Parameters
        ----------
        global_mixture : sklearn.mixture
            Global mixture model with n components to charactize the potential labels that local mixture might belong to.
        local_mixture : geobipy.Mixture
            Mixture model with k components fit to the estimated pdf of the histogram.
        log : scalar or 'e', optional
            Take the log of the histogram bins.
            Defaults to None.

        Returns
        -------
        array_like : n x 1 array of the probability that the local mixtures belong to each global mixture component.

        """

        if local_mixture.n_components == 0:
            return np.zeros(global_mixture.n_components)

        values, dum = cF._log(self.binCentres, log)

        fit_pdfs = local_mixture.probability(values, log=False)

        b = global_mixture.weights_ * global_mixture.predict_proba(local_mixture.means.reshape(-1, 1)) / np.float64(local_mixture.n_components)
        pdfs = np.dot(fit_pdfs, b).T
        pdfs = np.sum(pdfs, axis=1)

        return pdfs / np.sum(pdfs)


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


    def fit_estimated_pdf(self, mixture_type, **kwargs):

        if np.all(self.counts == 0):
            return None

        mixture = mixture_type.lower()
        if mixture == 'gaussian':
            mix = mixNormal()
        elif mixture == 'lorentzian':
            lmfit_model = models.LorentzianModel
        elif mixture == 'splitlorentzian':
            lmfit_model = models.SplitLorentzianModel
        elif mixture == 'voigt':
            lmfit_model = models.VoigtModel
        elif mixture == 'moffat':
            lmfit_model = models.MoffatModel
        elif mixture == 'pearson':
            mix = mixPearson()
        elif mixture == 'studentst':
            mix = mixStudentT()
        # elif mixture == 'exponentialgaussian':
        #     from lmfit.models import ExponentialGaussianModel as lmfit_model
        # elif mixture == 'skewedgaussian':
        #     from lmfit.models import SkewedGaussianModel as lmfit_model
        # elif mixture == 'exponential':
        #     from lmfit.models import ExponentialModel as lmfit_model
        # elif mixture == 'powerlaw':
        #     from lmfit.models import PowerLawModel as lmfit_model

        log = kwargs.get('log', None)
        kwargs['max_variance'] = kwargs.pop('max_variance', self.estimateStd(1e5, log=log))

        return mix.fit_to_curve(x=self.binCentres, y=self.pdf, **kwargs)


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
        axisPdf = self.pdf

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
        values = np.random.rand(np.int64(nSamples)) * self.cdf[-1]
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
            Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.

        See Also
        --------
        geobipy.plotting.pcolor : For additional keywords

        """
        return RectilinearMesh1D.pcolor(self, self._counts, **kwargs)


    def plot(self, flipX=False, flipY=False, trim=True, normalize=False, line=None, **kwargs):
        """ Plots the histogram """
        bins = self.edges_absolute

        values = self.pdf if normalize else self.counts

        ax = cP.bar(values, bins, flipX=flipX, flipY=flipY, trim=trim, **kwargs)

        if not line is None:
            f = plt.axhline if kwargs.get('transpose', False) else plt.axvline
            if np.size(line) > 1:
                [f(l, color=cP.wellSeparated[3], linewidth=1) for l in line]
            else:
                f(line, color=cP.wellSeparated[3], linewidth=1)
        
        return ax


    def plot_as_line(self, transpose=False, flipX=False, flipY=False, trim=True, normalize=False, **kwargs):

        x = self.centres_absolute
        x1 = x
        if kwargs.get('xscale', '') == 'log':
            x1 = np.log10(x1)

        if normalize:
            cnts = self.counts / np.trapz(self.counts, x = x1)
        else:
            cnts = self.counts

        ax = cnts.plot(x=x, **kwargs)

        if normalize:
            cP.ylabel('Density')

    @property
    def hdf_name(self):
        """ Reprodicibility procedure """
        return('Histogram1D')


    def createHdf(self, parent, name, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = RectilinearMesh1D.createHdf(self, parent, name, withPosterior, nRepeats, fillvalue)
        # grp = self.create_hdf_group(parent, name)
        self._counts.createHdf(grp, 'counts', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        RectilinearMesh1D.writeHdf(self, parent, name, withPosterior, index=index)
        grp = parent[name]
        self._counts.writeHdf(grp, 'counts', index=index)

    @classmethod
    def fromHdf(cls, grp, index=None):
        """ Reads in the object froma HDF file """
        self = super(Histogram1D, cls).fromHdf(grp, index)
        self._counts = StatArray.StatArray.fromHdf(grp['counts'], index=index)
        return self

    @property
    def summary(self):

        msg = ("{}\n"
              "Bins: \n{}"
              "Counts:\n{}"
              "Values are logged to base {}\n"
              "Relative to: {}").format(type(self), RectilinearMesh1D.summary, self.counts.summary, self.log, self.relativeTo)

        return msg

