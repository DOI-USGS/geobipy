
import numpy as np
from ...classes.core import StatArray
from scipy.stats import (multivariate_normal, norm)
import matplotlib.pyplot as plt
from .Mixture import Mixture
from sklearn.mixture import GaussianMixture

class mixNormal(Mixture):

    def __init__(self, means=None, variances=None, amplitudes=1.0):

        if np.all([means, variances] is None):
            return

        assert np.size(means) == np.size(variances), ValueError("means and variance must have same size")

        self._params = np.zeros(3 * np.size(means))
        self._params[0::3] = amplitudes
        self._params[1::3] = means
        self._params[2::3] = variances

    @property
    def amplitudes(self):
        return self._params[0::3]

    @amplitudes.setter
    def amplitudes(self, values):
        assert np.size(values) == self.n_mixtures, ValueError("Must provide {} amplitudes".format(self.n_mixtures))
        self._params[0::3] = values

    @property
    def means(self):
        return self._params[1::3]

    @means.setter
    def means(self, values):
        assert np.size(values) == self.n_mixtures, ValueError("Must provide {} means".format(self.n_mixtures))
        self._params[1::3] = values

    @property
    def moments(self):
        return [self.means, self.variances]

    @property
    def variances(self):
        return self._params[2::3]

    @variances.setter
    def variances(self, values):
        assert np.size(values) == self.n_mixtures, ValueError("Must provide {} variances".format(self.n_mixtures))
        self._params[2::3] = values

    @property
    def mixture_model_class(self):
        return GaussianMixture

    @property
    def n_solvable_parameters(self):
        return 3

    @property
    def n_mixtures(self):
        return self.means.size


    def plot_components(self, x, log, ax=None, **kwargs):

        if not ax is None:
            plt.sca(ax)

        probability = self.amplitudes * self.probability(x, log)

        p = probability.plot(x=x, **kwargs)

        return p


    def probability(self, x, log, component=None):

        if component is None:
            out = StatArray.StatArray(np.empty([np.size(x), self.n_mixtures]), "Probability Density")
            for i in range(self.n_mixtures):
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
        self.__init__(np.squeeze(mixture.means_), np.squeeze(mixture.covariances_), amplitudes=np.squeeze(mixture.weights_))




