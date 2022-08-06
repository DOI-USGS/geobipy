
import numpy as np
from ...classes.core import StatArray
from scipy.stats import (multivariate_normal, norm)
import matplotlib.pyplot as plt
from .Mixture import Mixture
from sklearn.mixture import GaussianMixture
from lmfit.models import GaussianModel
from copy import deepcopy

class mixNormal(Mixture):

    def __init__(self, means=None, sigmas=None, amplitudes=1.0):

        if np.all([means, sigmas] is None):
            return

        self.params = np.zeros(self.n_solvable_parameters * np.size(means))

        self.amplitudes = amplitudes
        self.means = means
        self.sigmas = sigmas

    def __deepcopy__(self, memo={}):
        out = type(self)()
        out._params = deepcopy(self._params)

    @property
    def amplitudes(self):
        return self._params[0::self.n_solvable_parameters]

    @amplitudes.setter
    def amplitudes(self, values):
        assert np.size(values) == self.n_components, ValueError("Must provide {} amplitudes".format(self.n_components))
        self._params[0::self.n_solvable_parameters] = values

    # @property
    # def lmfit_model(self):
    #     return Pearson7Model

    @property
    def means(self):
        return self._params[1::self.n_solvable_parameters]

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


    def fit_to_curve(self, *args, **kwargs):
        fit = super().fit_to_curve(*args, **kwargs)
        self.params = np.asarray(list(fit.best_values.values()))
        return self


    def plot_components(self, x, log, ax=None, **kwargs):

        if not ax is None:
            plt.sca(ax)

        probability = self.amplitudes * self.probability(x, log)

        p = probability.plot(x=x, **kwargs)

        return p


    def probability(self, x, log, component=None):

        if component is None:
            out = StatArray.StatArray(np.empty([np.size(x), self.n_components]), "Probability Density")
            for i in range(self.n_components):
                tmp = self.amplitudes[i] * self._probability(x, log, self.means[i], self.variances[i])
                out[:, i] = tmp
            return out
        else:
            return self.amplitudes[component] * self._probability(x, log, self.means[component], self.variances[component])


    def _probability(self, x, log, mean, variance):
        """ For a realization x, compute the probability """
        if log:
            return StatArray.StatArray(norm.logpdf(x, loc = mean, scale = variance), "Probability Density")
        else:
            return StatArray.StatArray(norm.pdf(x, loc = mean, scale = variance), "Probability Density")


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
