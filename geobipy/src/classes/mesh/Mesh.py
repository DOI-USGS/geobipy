""" @Mesh_Class
Module describing a Mesh
"""
import numpy as np
from ...classes.core.myObject import myObject
from ...base.utilities import _log as log_
from ..core import StatArray


class Mesh(myObject):
    """Abstract Base Class

    This is an abstract base class for additional meshes

    See Also
    ----------
    geobipy.RectilinearMesh1D
    geobipy.RectilinearMesh2D
    geobipy.RectilinearMesh3D

    """

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
        percent = 0.5 * np.minimum(percent, 100.0-percent)
        tmp = self._percentile(values=values, percent=np.r_[50.0, percent, 100.0-percent], axis=axis)
        return np.take(tmp, 0, axis), np.take(tmp, 1, axis), np.take(tmp, 2, axis)

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
        percent = 0.5 * np.minimum(percent, 100.0 - percent)
        tmp = self._percentile(values, np.r_[percent, 100.0 - percent], axis=axis)
        tmp, _ = log_(tmp, log=log)
        return np.squeeze(np.abs(np.diff(tmp, axis=axis)))

    def _mean(self, values, axis=0):
    
        a = self.axis(axis)
        s = tuple([np.s_[:] if i == axis else None for i in range(self.ndim)])

        t = np.sum(a.centres[s] * values, axis = axis)
        s = values.sum(axis = axis)

        if t.size == 1:
            out = t / s
        else:
            i = np.where(s > 0.0)
            out = StatArray.StatArray(t.shape)
            out[i] = t[i] / s[i]

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
        return self._percentile(values, 50.0, axis)

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
        cs = np.cumsum(values, axis=axis)
        # Cumulative "probability"
        d = np.expand_dims(total, axis)
        tmp = np.zeros_like(cs, dtype=np.float64)
        np.divide(cs, d, out=tmp, where=d > 0.0)
        # Find the interval
        i = np.apply_along_axis(np.searchsorted, axis, tmp, percent)
        i[i == values.shape[axis]] = values.shape[axis]-1
        # Obtain the values at those locations
        out = self.axis(axis).centres[i]

        return out