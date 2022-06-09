import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .mixPearson import mixPearson
from ...base import utilities
from ...base import plotting as cP
from ..model.Model import Model
from ..core.StatArray import StatArray
import progressbar

class Histogram(Model):

    def __init__(self, mesh=None, values=None):
        """ Instantiate a 2D histogram """
        super().__init__(mesh=mesh)
        
        if not values is None:
            self.update(values)

    @property
    def counts(self):
        return self._values

    @property
    def ndim(self):
        return self.mesh.ndim

    @property
    def pdf(self):
        out = Model(self.mesh)
        if self.values.max() > 0:
            out.values = StatArray(self.values / np.sum(self.mesh.area * self.values), name='Density')
        else:
            out.values = StatArray(out.mesh.shape, name='Density')
        return out

    @property
    def pmf(self):
        out = Model(self.mesh)
        if self.values.max() > 0:
            out.values = StatArray(self.values / np.sum(self.values), name='mass')
        else:
            out.values = StatArray(out.mesh.shape, name='mass')
        return out

    @Model.values.setter
    def values(self, values):
        if values is None:
            self._values = StatArray(self.shape, name='Frequency', dtype=np.int32)
            return

        # assert np.all(values.shape == self.shape), ValueError("values must have shape {}".format(self.shape))
        self._values = StatArray(values, name='Frequency')

    def __getitem__(self, slic):
        mesh = self.mesh[slic]
        out = type(self)(mesh)
        out.values = self.values[slic]
        return out

    def animate(self, axis, filename, slic=None, **kwargs):
        return self.mesh._animate(self.values, axis, filename, slic, **kwargs)

    def bar(self, **kwargs):
        if np.all(self.values == 0):
            return
        return super().bar(**kwargs)

    def cdf(self, axis=None):
    
        if axis is None:
            cdf = np.cumsum(self.values, axis=0)
            for i in range(1, self.ndim):
                cdf = np.cumsum(cdf, axis=i)
        else:
            cdf = np.cumsum(self.values, axis=axis)
        cdf = cdf / cdf.max()

        return Model(self.mesh, StatArray(cdf, name='Cumulative Density Function'))

    def compute_probability(self, distribution, log=None, log_probability=False, axis=0, **kwargs):
        return self.mesh._compute_probability(distribution, self.pdf.values, log, log_probability, axis, **kwargs)

    def credible_intervals(self, percent=90.0, axis=0):
        """Gets the median and the credible intervals for the specified axis.

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
        return self.mesh._credible_intervals(values=self.pmf.values, percent=percent, axis=axis)
    
    def credible_range(self, percent=90.0, log=None, axis=0):
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
        return self.mesh._credible_range(self.counts, percent=percent, log=log, axis=axis)

    def entropy(self, log=2, axis=None):

        assert log in [2, 10, 'e'], ValueError("log must be one of [2, 'e', 10]")
        pdf = self.pdf
        logged, _ = utilities._log(pdf.values, log=log)
        entropy = pdf.values * logged
        entropy[np.isnan(entropy)] = 0.0

        entropy = -entropy.sum(axis=axis)
        
        entropy.name = 'Entropy'

        if log == 2:
            entropy.units = 'bits'
        elif log == 10:
            entropy.units = 'bans'
        elif log == 'e':
            entropy.units = 'nats'

        return Model(mesh=self.mesh.remove_axis(axis), values=entropy)

    def estimate_std(self, n_samples, **kwargs):
        return np.sqrt(self.estimate_variance(n_samples, **kwargs))

    def estimate_variance(self, n_samples, **kwargs):
        X = self.sample(n_samples, **kwargs)
        return np.var(X)

    def fit_mixture_to_pdf(self, mixture=mixPearson, axis=0, **kwargs):
        if self.ndim == 1:
            return self.fit_mixture_to_pdf_1d(mixture, **kwargs)
        elif self.ndim == 2:
            return self.fit_mixture_to_pdf_2d(mixture, axis=axis, **kwargs)
        elif self.ndim == 3:
            return self.fit_mixture_to_pdf_3d(mixture, axis=axis, **kwargs)

    def fit_mixture_to_pdf_3d(self, mixture, axis, **kwargs):
        ax, bx = self.mesh.other_axis(axis)

        a = [x for x in (0, 1, 2) if not x == axis]
        b = [x for x in (0, 1, 2) if x == axis]

        mixtures = [None] * ax.nCells * bx.nCells
        if np.all(self.values == 0):
            return mixtures

        track = kwargs.pop('track', True)

        r = range(ax.nCells.item() * bx.nCells.item())
        if track:
            Bar = progressbar.ProgressBar()
            r = Bar(r)
            
        slic = [np.s_[:] for i in range(3)]
        mixtures = []
        for i in r:
            j = list(self.mesh.unravelIndex(i))
            j[axis] = np.s_[:]
            h = self[tuple(j)]
            mixtures.append(h.fit_mixture_to_pdf(mixture=mixture, **kwargs))
            
        return mixtures

    def fit_mixture_to_pdf_2d(self, mixture, axis, **kwargs):

        ax = self.axis(axis)
        mixtures = [None] * ax.nCells
        if np.all(self.values == 0):
            return mixtures

        track = kwargs.pop('track', True)

        r = range(ax.nCells.item())
        if track:
            Bar = progressbar.ProgressBar()
            r = Bar(r)
            
        mixtures = []
        for i in r:
            h = self.take_along_axis(i, axis=axis)

            mixtures.append(h.fit_mixture_to_pdf(mixture=mixture, **kwargs))
            
        return mixtures


    def fit_mixture_to_pdf_1d(self, mixture, **kwargs):
        """Find peaks in the histogram along an axis.

        Parameters
        ----------
        intervals : array_like, optional
            Accumulate the histogram between these invervals before finding peaks
        axis : int, optional
            Axis along which to find peaks.

        """
        if np.all(self.values == 0):
            return None

        values = self.pdf.values
        smooth = kwargs.pop('smooth', False)
        if smooth:
            values = self.pdf.values.smooth(0.5)

        return mixture().fit_to_curve(x=self.mesh.centres_absolute, y=values, **kwargs)

    def marginalize(self, axis=0):
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
        assert 0 <= axis <= self.mesh.ndim, ValueError("0 <= axis <= {}".format(self.mesh.ndim))

        mesh = self.mesh.remove_axis(axis)

        out = Histogram(mesh=mesh)
        out.values = np.sum(self.counts, axis=axis)
        return out

    def mean(self, axis=0):
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
        out = self.mesh.remove_axis(axis)
        return Model(mesh=out, values=self.mesh._mean(self.counts, axis=axis))

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
        return self.mesh._median(values=self.counts, axis=axis)

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
        out = self.transparency(percent=percent, log=log, axis=axis)
        out.values = 1.0 - out.values
        out.name = 'Opacity'
        return out

    def opacity_level(self, percent=95.0, log=None, axis=0):
        """ Get the index along axis 1 from the bottom up that corresponds to the percent opacity """

        p = 0.01 * percent
        op = self.opacity(log=log, axis=axis)

        nz = op.nCells - 1
        iC = nz
        while op.values[iC] > p and iC >= 0:
            iC -= 1
        return self.y.centres[iC]

    def percentile(self, percent, log=None, reciprocate=False, axis=0):
        """Gets the median and the credible intervals for the specified axis.

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
        return Model(self.mesh.remove_axis(axis), values=self.mesh._percentile(values=self.pmf.values, percent=percent, axis=axis))

    def pcolor(self, **kwargs):
        kwargs['cmap'] = kwargs.get('cmap', 'gray_r')
        return super().pcolor(**kwargs)

    def plot(self, line=None, **kwargs):
        """ Plots the histogram """
        kwargs['trim'] = kwargs.pop('trim', 0.0)
        
        values = self.counts
        if kwargs.pop('normalize', False):
            values = self.pdf.values

        interval_kwargs = kwargs.pop('credible_interval_kwargs', None)
        if interval_kwargs is not None:
            interval_kwargs['xscale'] = kwargs.get('xscale', 'linear')
            interval_kwargs['yscale'] = kwargs.get('yscale', 'linear')

        if self.ndim == 1:
            ax = self.mesh.bar(values=values, **kwargs)

            if line is not None:
                kwargs['color'] = kwargs.pop('linecolor', cP.wellSeparated[3])
                self.mesh.plot_line(line, **kwargs)

            if interval_kwargs is not None:
                self.plotCredibleIntervals(**interval_kwargs)
            return ax
        else:
            kwargs['cmap'] = kwargs.pop('cmap', mpl.cm.Greys)
            ax, pm, cb = self.mesh.pcolor(values=values, **kwargs)

            if interval_kwargs is not None:
                self.plotCredibleIntervals(**interval_kwargs)

            return ax, pm, cb

    def plotCredibleIntervals(self, percent=95.0, axis=0, **kwargs):
    
        med, low, high = self.credible_intervals(percent=percent, axis=axis)

        kwargs['color'] = '#5046C8'
        kwargs['linestyle'] = 'dashed'
        kwargs['linewidth'] = 2
        kwargs['alpha'] = 0.6

        p = 0.5 * np.minimum(percent, 100.0-percent)
        kwargs['label'] = '{}%'.format(p)
        self.mesh.plot_line(low, axis=axis, **kwargs)
        kwargs['label'] = '{}%'.format(100.0 - p)
        self.mesh.plot_line(high, axis=axis, **kwargs)

    def plotMean(self, log=None, axis=0, **kwargs):
    
        m = self.mean(axis=axis)
        kwargs['label'] = 'mean'
        self.mesh.plot_line(m, axis=axis, **kwargs)

    def plotMedian(self, log=None, axis=0, **kwargs):

        m = self.median(axis=axis)
        kwargs['label'] = 'median'
        self.mesh.plot_line(m, axis=axis, **kwargs)

    def sample(self, n_samples, log=None):
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
        cdf = self.cdf()
        values = np.random.rand(np.int64(n_samples))
        values = np.interp(values, np.hstack([0, cdf.values]), self.mesh.edges_absolute)
        values, dum = utilities._log(values, log)
        return values

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

        out = StatArray(self.credible_range(percent=percent, log=log, axis=axis), 'Transparency')
        mn = np.nanmin(out)
        mx = np.nanmax(out)
        t = mx - mn
        if t > 0.0:
            out = (out - mn) / t
        else:
            out -= mn

        return Model(self.mesh.remove_axis(axis), values=out)

    def update(self, *args, **kwargs):
        iBin = self.mesh.cellIndices(*args, clip=True, **kwargs)

        axis = None if iBin.size == 1 else np.ndim(iBin)-1

        unique, counts = np.unique(iBin, axis=axis, return_counts=True)
        if np.ndim(unique) > 1:
            unique = tuple(unique)
        self.values[unique] += counts

    def update_with_line(self, x, y):
        j = self.mesh.line_indices(x, y)
        self.counts[j[:, 0], j[:, 1]] += 1

    def createHdf(self, *args, **kwargs):
        return super().createHdf(*args, **kwargs)

    @classmethod
    def fromHdf(cls, grp, index=None):
        """ Reads in the object from a HDF file """
        return super(Histogram, cls).fromHdf(grp=grp, index=index, skip_posterior=True)