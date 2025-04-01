
import numpy as np
from copy import deepcopy
from ..core.DataArray import DataArray
from scipy.stats import (multivariate_normal, norm)
from scipy.special import beta
import matplotlib.pyplot as plt
from .mixNormal import mixNormal
from sklearn.mixture import GaussianMixture
from lmfit.models import Pearson7Model

class mixPearson(mixNormal):

    def __init__(self, amplitudes=None, means=None, sigmas=None, exponents=None, labels=None):

        super().__init__(amplitudes=amplitudes, means=means, sigmas=sigmas, labels=labels)
        self.exponents = exponents

    @property
    def moments(self):
        return [self.means, self.variances, self.exponents]

    @property
    def exponents(self):
        return DataArray(self._params[3::self.n_solvable_parameters], 'Exponent')

    @exponents.setter
    def exponents(self, values):
        assert np.size(values) == self.n_components, ValueError("Must provide {} exponents".format(self.n_components))
        self._params[3::self.n_solvable_parameters] = values

    @property
    def model(self):
        return Pearson7Model

    @property
    def mixture_model_class(self):
        return GaussianMixture

    @property
    def n_solvable_parameters(self):
        return 4

    @property
    def summary(self):
        """ """
        msg = ("Pearson7 Mixture Model: \n"
              "amplitude{}\nmean\n{}variance\n{}exponent\n{}").format(self.amplitudes.summary, self.means.summary, self.variances.summary, self.exponents.summary)
        return msg

    def squeeze(self):

        i = np.where(self.amplitudes > 0.0)[0]

        means = self.means[i]
        amplitudes = self.amplitudes[i]
        sigma = self.sigmas[i]
        exp = self.exponents[i]

        return mixPearson(amplitudes, means, sigma, exp)

    def probability(self, x, log, component=None):

        if component is None:
            out = DataArray(np.empty([np.size(x), self.n_components]), "Probability Density")
            for i in range(self.n_components):
                out[:, i] =  self._probability(x, log, self.means[i], self.variances[i], self.exponents[i])
            return out
        else:
            return self._probability(x, log, self.means[component], self.variances[component], self.exponents[component])


    def _probability(self, x, log, mean, sigma, exponent):
        """ For a realization x, compute the probability """

        p = (1.0 / (sigma * beta(exponent - 0.5, 0.5))) * (1 + ((x - mean)**2.0)/(sigma**2.0))**-exponent

        if log:
            p = np.log(p)

        return DataArray(p, "Probability Density")
