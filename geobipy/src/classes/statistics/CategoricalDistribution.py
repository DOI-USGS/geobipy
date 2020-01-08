

""" @CategoricalDistribution
Module defining a categorical distribution with statistical procedures
"""
#from copy import deepcopy
import numpy as np
from ...base.logging import myLogger
from .baseDistribution import baseDistribution
from ...base.HDF.hdfWrite import writeNumpy
#from .MvNormalDistribution import MvNormal
from scipy.stats import norm
from ...base import customPlots as cP
from ..core import StatArray

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
        assert np.size(probabilities) == len(events), ValueError("Number of probabilities must equal number of events {}".format(len(events)))

        baseDistribution.__init__(self, prng)
        self._probabilities = probabilities/np.sum(probabilities)
        self._probabilityMassFunction = np.cumsum(self._probabilities)
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

    def get_event(self, index):
        """Return the event name for a given index

        Parameters
        ----------
        index : ints
            Event id from 0 to nEvents - 1

        Returns
        -------
        out : list of str
            Names of the events

        """
        return [self.events[i] for i in index]


    def probability(self, x):
        out = np.empty(x)
        n = np.size(x)
        for i in range(n):
            out[i] = self._probabilities[i]

        return out

    
    def deepcopy(self):
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
        r = self.prng.rand(size)
        return np.searchsorted(self._probabilityMassFunction, r)


    def bins(self):
        raise NotImplementedError("bins is not implemented")

