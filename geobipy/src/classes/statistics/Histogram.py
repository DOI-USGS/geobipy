import numpy as np
from copy import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from ...base import utilities
from ...base import plotting as cP
from ..model.Model import Model
from ..core.StatArray import StatArray

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
        return Model(self.mesh, StatArray(self.values / np.sum(self.mesh.area * self.values), name='Density'))

    @property
    def pmf(self):
        return Model(self.mesh, StatArray(self.values / np.sum(self.values), name='Mass'))

    @property
    def summary(self):
        """Summary of self """
        msg =  "Histogram\n"
        msg += "mesh:\n{}".format("|   "+(self.mesh.summary.replace("\n", "\n|   "))[:-4])
        msg += "counts:\n{}".format("|   "+(self.values.summary.replace("\n", "\n|   "))[:-4])

        return msg

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
            total = self.values.sum(axis=axis)
            cdf = np.cumsum(self.values, axis=axis)

            # s = [None if i == axis else np.s_[:] for i in range(self.ndim)]
            # cdf = cdf / total[s]

        return Model(self.mesh, StatArray(cdf, name='Cumulative Density Function'))

    def credibleIntervals(self, percent=90.0, log=None, reciprocate=False, axis=0):
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
        return self.mesh._credibleIntervals(values=self.pmf.values, percent=percent, log=log, reciprocate=reciprocate, axis=axis)
    
    def credibleRange(self, percent=90.0, log=None, axis=0):
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
        return self.mesh._credibleRange(self.counts, percent=percent, log=log, axis=axis)

    # def entropy(self, axis=None):

    #     pdf = self.pdf(axis=axis)
    #     pdf = pdf[pdf > 0.0]
    #     return StatArray(-(pdf * np.log(np.abs(pdf))).sum(), "Entropy")

    def marginalize(self, log=None, axis=0):
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
        return self.mesh._mean(self.counts, log=log, axis=axis)

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
        return self.mesh._median(values=self.counts, log=log, axis=axis)

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

    def opacity_level(self, percent=95.0, log=None, axis=0):
        """ Get the index along axis 1 from the bottom up that corresponds to the percent opacity """

        p = 0.01 * percent
        op = self.opacity(log=log, axis=axis)
        nz = op.size - 1
        iC = nz
        while op[iC] > p and iC >= 0:
            iC -= 1
        return self.y.centres[iC]

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

        if self.ndim == 1:
            ax = self.bar(**kwargs)

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

    def plotCredibleIntervals(self, percent=95.0, log=None, reciprocate=False, axis=0, **kwargs):
    
        med, low, high = self.credibleIntervals(percent=percent, log=log, reciprocate=reciprocate, axis=axis)

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
    
        m = self.mean(log=log, axis=axis)
        kwargs['label'] = 'mean'
        self.mesh.plot_line(m, axis=axis, **kwargs)

    def plotMedian(self, log=None, axis=0, **kwargs):

        m = self.median(log=log, axis=axis)
        kwargs['label'] = 'median'
        self.mesh.plot_line(m, axis=axis, **kwargs)

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

    @classmethod
    def fromHdf(cls, grp, index=None):
        """ Reads in the object from a HDF file """
        return super(Histogram, cls).fromHdf(grp=grp, index=index)