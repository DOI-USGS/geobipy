""" @Mesh_Class
Module describing a Mesh
"""
from numpy import abs, argmax, apply_along_axis, cumsum, diff, divide, expand_dims, float64
from numpy import minimum, ndim, nan
from numpy import r_, s_, searchsorted, size, squeeze, sum, take, take_along_axis, where, zeros_like
from ..core.myObject import myObject
from ...base.utilities import _log as log_
from ...base.utilities import _power as power_
from ..core.DataArray import DataArray


class Mesh(myObject):
    """Abstract Base Class

    This is an abstract base class for additional meshes

    See Also
    ----------
    geobipy.RectilinearMesh1D
    geobipy.RectilinearMesh2D
    geobipy.RectilinearMesh3D

    """
    __slots__ = ()
    def __init__(self):
        """ABC method"""
        NotImplementedError()

    def _credible_intervals(self, values, percent=90.0, axis=0):
        """Gets the median and the credible intervals for the specified axis.

        Parameters
        ----------
        values : array_like
        Values to use to compute the intervals.
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
        percent = 0.5 * minimum(percent, 100.0-percent)
        tmp = self._percentile(values=values, percent=r_[50.0, percent, 100.0-percent], axis=axis)
        return take(tmp, 0, axis), take(tmp, 1, axis), take(tmp, 2, axis)

    def _credible_range(self, values, percent=90.0, log=None, axis=0):
        """ Get the range of credibility

        Parameters
        ----------
        values : array_like
            Values to use to compute the range.
        percent : float
            Percent of the credible intervals
        log : 'e' or float, optional
            If None: The range is the difference in linear space of the credible intervals
            If 'e' or float: The range is the difference in log space, or ratio in linear space.
        axis : int
            Axis along which to get the marginal histogram.

        """
        percent = 0.5 * minimum(percent, 100.0 - percent)
        tmp = self._percentile(values, r_[percent, 100.0 - percent], axis=axis)
        if self.axis(axis).log is not None:
            tmp, _ = log_(tmp, log=self.axis(axis).log)
        return squeeze(abs(diff(tmp, axis=axis)))

    def _mean(self, counts, axis=0):
        ax = self.axis(axis)
        s = tuple([s_[:] if i == axis else None for i in range(self.ndim)])
        t = sum(ax.centres[s] * counts, axis = axis)
        N = counts.sum(axis = axis)

        if t.size == 1:
            out = t / N
        else:
            i = where(N == 0.0)
            N[i] = 1
            out = DataArray(t.shape)
            out = (t / N)

        if ax._relative_to is not None:
            re = ax.relative_to

            nd = ndim(re)
            ns = re.size
            if nd == 2:
                out += re
            elif nd == 1:
                if ns == 1:
                    out += re
                else:
                    out += re[:, None]
            else:
                out += re
        out = power_(out, ax.log)

        out.name = 'Mean ' + ax.name
        out.units = ax.units

        return out

    def _median(self, values, axis=0):
        """Gets the median for the specified axis.

        Parameters
        ----------
        values : array_like
            2D array to get the median.
        log : 'e' or float, optional
            Take the log of the median to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the median.

        Returns
        -------
        med : array_like
            Contains the medians along the specified axis. Has size equal to arr.shape[axis].

        """
        out = self._percentile(values, 50.0, axis)
        if isinstance(out, DataArray):
            out.name =  "Median " + self.axis(axis).name
        return out

    def _mode(self, values, axis=0):
        """Gets the maximum along axis.

        Parameters
        ----------
        values : array_like
            Values used to compute interval like histogram counts.
        axis : int
            Along which axis to obtain the interval locations.

        Returns
        -------
        interval : array_like
            Contains the location of the maximum. Has size equal to self.shape[axis].

        """
        # total of the counts
        i = argmax(values, axis=axis)
        # i[i == values.shape[axis]] = values.shape[axis]-1

        # Obtain the values at those locations
        if self.ndim == 1:
            out = self.centres_absolute[i]
        else:

            centres = self.axis(axis).centres_absolute
            if centres.ndim != i.ndim:
            # if centres.ndim > 1:
                i = expand_dims(i, axis)

            out = squeeze(take_along_axis(centres, i, axis=axis))

        out.name = "Mode " + self.axis(axis).name
        return out

    def _percentile(self, values, percent=95.0, axis=0):
        """Gets the percent interval along axis.

        Get the statistical interval, e.g. median is 50%.

        Parameters
        ----------
        values : array_like
            Values used to compute interval like histogram counts.
        percent : float
            Interval percentage.  0.0 < percent < 100.0
        log : 'e' or float, optional
            Take the log of the interval to a base. 'e' if log = 'e', or a number e.g. log = 10.
        axis : int
            Along which axis to obtain the interval locations.

        Returns
        -------
        interval : array_like
            Contains the interval along the specified axis. Has size equal to self.shape[axis].

        """
        percent *= 0.01

        # total of the counts
        total = values.sum(axis=axis)
        # Cumulative sum
        cs = cumsum(values, axis=axis)
        # Cumulative "probability"
        d = expand_dims(total, axis)
        tmp = zeros_like(cs, dtype=float64)
        divide(cs, d, out=tmp, where=d > 0.0)

        # Find the interval
        i = apply_along_axis(searchsorted, axis, tmp, percent)
        i[i == values.shape[axis]] = values.shape[axis]-1

        # Obtain the values at those locations
        if self.ndim == 1:
            return self.centres_absolute[i]
        else:
            if size(percent) == 1:
                i = expand_dims(i, axis)
            return squeeze(take_along_axis(self.centres(axis), i, axis=axis))

    def remove_axis(self, axis):
        tmp = [s_[:] for i in range(self.ndim)]
        tmp[axis] = 0
        tmp = tuple(tmp)
        return self[tmp]