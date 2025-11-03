
import numpy as np
from ..core.DataArray import DataArray
from scipy.stats import (multivariate_normal, norm)
import matplotlib.pyplot as plt
from .Mixture import Mixture
from sklearn.mixture import GaussianMixture
from lmfit.models import GaussianModel
from copy import deepcopy

class mixNormal(Mixture):

    def __init__(self, means=None, sigmas=None, amplitudes=None, labels=None):

        if np.all([means, sigmas] is None):
            return

        self.params = np.zeros(self.n_solvable_parameters * np.size(means))

        self.amplitudes = amplitudes
        self.means = means
        self.sigmas = sigmas
        self.labels = labels


    @property
    def means(self):
        return DataArray(self._params[1::self.n_solvable_parameters], 'Mean')

    @means.setter
    def means(self, values):
        assert np.size(values) == self.n_components, ValueError("Must provide {} means".format(self.n_components))
        self._params[1::self.n_solvable_parameters] = values

    @property
    def moments(self):
        return [self.means, self.variances]

    @property
    def variances(self):
        return self.sigmas**2.0

    @property
    def sigmas(self):
        return self._params[2::self.n_solvable_parameters]

    @sigmas.setter
    def sigmas(self, values):
        assert np.size(values) == self.n_components, ValueError("Must provide {} sigmas".format(self.n_components))
        self._params[2::self.n_solvable_parameters] = values

    @property
    def model(self):
        return GaussianModel

    @property
    def mixture_model_class(self):
        return GaussianMixture

    @property
    def n_solvable_parameters(self):
        return 3

    @property
    def n_components(self):
        return np.size(self.means)

    def plot_components(self, x, log, ax=None, **kwargs):

        if not ax is None:
            plt.sca(ax)

        probability = self.amplitudes * self.probability(x, log)

        p = probability.plot(x=x, **kwargs)

        return p


    def probability(self, x, log, component=None):

        if component is None:
            out = DataArray(np.empty([np.size(x), self.n_components]), "Probability Density")
            for i in range(self.n_components):
                tmp = self.amplitudes[i] * self._probability(x, log, self.means[i], self.variances[i])
                out[:, i] = tmp
            return out
        else:
            return self.amplitudes[component] * self._probability(x, log, self.means[component], self.variances[component])


    def _probability(self, x, log, mean, variance):
        """ For a realization x, compute the probability """
        if log:
            return DataArray(norm.logpdf(x, loc = mean, scale = variance), "Probability Density")
        else:
            return DataArray(norm.pdf(x, loc = mean, scale = variance), "Probability Density")


    def sum(self, x):
        return self._sum(x, *self._params)


    def _sum(self, x, *params):

        out = np.zeros_like(x)

        nm = np.int(len(params) / 3)
        for i in range(nm):
            i1 = i*3
            amp, mean, var = params[i1:i1+3]
            out += amp * norm.pdf(x, mean, var)

        return out


    def _assign_from_mixture(self, mixture):
        self.__init__(np.squeeze(mixture.means_), np.sqrt(np.squeeze(mixture.covariances_)), amplitudes=np.squeeze(mixture.weights_))
