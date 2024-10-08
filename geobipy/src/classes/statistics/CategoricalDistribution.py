

""" @CategoricalDistribution
Module defining a categorical distribution with statistical procedures
"""
#from copy import deepcopy
from numpy import cumsum, empty, searchsorted, size, sum
from .baseDistribution import baseDistribution
from scipy.stats import norm
from ...base import plotting as cP

class Categorical(baseDistribution):
    """Categorical distribution

    Categorical(probabilities)

    Parameters
    ----------
    probabilities : array_like
        Probability of each category occuring.
    events : list of str
        Names of each event

    """
    def __init__(self, probabilities, events, prng=None):
        """Instantiate a Normal distribution """
        assert size(probabilities) == len(events), ValueError("Number of probabilities must equal number of events {}".format(len(events)))

        baseDistribution.__init__(self, prng)
        self._probabilities = probabilities/sum(probabilities)
        self._probabilityMassFunction = cumsum(self._probabilities)
        self._events = events


    @property
    def probabilities(self):
        return self._probabilities

    @property
    def events(self):
        return self._events

    @property
    def nEvents(self):
        return self.probabilities.size

    def probability(self, x):
        out = empty(x)
        n = size(x)
        for i in range(n):
            out[i] = self._probabilities[i]

        return out

    def __deepcopy__(self):
        """Create a deepcopy

        Returns
        -------
        out : Categorical
            Copied dsitribution.

        """
        return Categorical(self._probabilities, self._events, self.prng)


    def rng(self, size=1):
        """Randomly generate events

        Parameters
        ----------
        size : int
            Number of events to generate

        Returns
        -------
        out : ints
            Event index, 0 to nEvents - 1

        """
        r = self.prng.uniform(size=size)
        return searchsorted(self._probabilityMassFunction, r)


    def bins(self):
        raise NotImplementedError("bins is not implemented")
