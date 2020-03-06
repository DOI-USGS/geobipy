from .baseDistribution import baseDistribution
from ..core import StatArray
import numpy as np

class MvDistribution(baseDistribution):
    """Instantiate a multivariate statistical distribution

    Parameters
    ----------
    distributions : (list of) geobipy.baseDistribution
        Single or list of univariate distributions, one per dimension.

    Returns
    -------
    out : geobipy.MvDistribution
        Subclass of baseDistribution

    """

    def __init__(self, distributions):

        assert isinstance(distributions, list), TypeError("distributions must be a list of univariate distributions")
        assert len(distributions) > 1, ValueError("distributions must have len > 1")

        for i, d in enumerate(distributions):
            assert not d.multivariate, TypeError("distribution {} must be univariate".format(i))

        self._distributions = distributions

    @property
    def distributions(self):
        return self._distributions

    @property
    def ndim(self):
        return len(self.distributions)

    @property
    def multivariate(self):
        return True


    def probability(self, axes, log, axis=None):

        if not isinstance(axes, list):
            axes = [axes]
        assert isinstance(axes, list), TypeError("axes must be a list, each entry contains the values to evaluate each dimension at.")

        if not axis is None:
            assert axis < self.ndim, ValueError("Axis must be less than number of dimensions {}".format(self.ndim))
        else:
            assert len(axes) == self.ndim, ValueError("Number of axes {} must equal the number of dimensions {}".format(len(axes), self.ndim))

        if axis is None:
            return StatArray.StatArray(np.outer(self.distributions[0].probability(axes[0], log), self.distributions[1].probability(axes[1], log)), "Probability density")
        else:
            return StatArray.StatArray(self.distributions[axis].probability(axes[0], log), "Probability density")


    def rng(self, size = 1):
        """  """

        out = np.empty([size, self.ndim])

        for i, d, in enumerate(self.distribution):
            out[:, i] = d.rng(size=size)

        return out
