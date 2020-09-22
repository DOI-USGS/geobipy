
import numpy as np
from ...classes.core import StatArray
from scipy.stats import t
import matplotlib.pyplot as plt
from .Mixture import Mixture
from smm import SMM

class mixStudentT(Mixture):

    def __init__(self, means=None, variances=None, degrees=None, amplitudes=1.0, labels=None):

        if np.all([means, variances, degrees] is None):
            return

        assert np.size(means) == np.size(variances) == np.size(degrees), ValueError("means, variances, degrees, must have same size")

        self._params = np.zeros(4 * np.size(means))
        self._params[0::4] = amplitudes
        self._params[1::4] = means
        self._params[2::4] = variances
        self._params[3::4] = degrees

        self._labels = np.zeros(self.n_mixtures)
        if not labels is None:
            self.labels = labels

    @property
    def amplitudes(self):
        return self._params[0::4]

    @amplitudes.setter
    def amplitudes(self, values):
        assert np.size(values) == self.n_mixtures, ValueError("Must provide {} amplitudes".format(self.n_mixtures))
        self._params[0::4] = values

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, values):
        assert np.size(values) == self.n_mixtures, ValueError("Labels must have size {}".format(self.n_mixtures))
        self._labels[:] = values

    @property
    def means(self):
        return self._params[1::4]

    @means.setter
    def means(self, values):
        assert np.size(values) == self.n_mixtures, ValueError("Must provide {} means".format(self.n_mixtures))
        self._params[1::4] = values

    @property
    def moments(self):
        return [self.means, self.variances, self.dfs]

    @property
    def variances(self):
        return self._params[2::4]

    @variances.setter
    def variances(self, values):
        assert np.size(values) == self.n_mixtures, ValueError("Must provide {} variances".format(self.n_mixtures))
        self._params[2::4] = values

    @property
    def degrees(self):
        return self._params[3::4]

    @degrees.setter
    def degrees(self, values):
        assert np.size(values) == self.n_mixtures, ValueError("Must provide {} degrees".format(self.n_mixtures))
        self._params[3::4] = values

    @property
    def mixture_model_class(self):
        return SMM

    @property
    def n_solvable_parameters(self):
        return 4

    @property
    def n_mixtures(self):
        return self.means.size


    def fit_to_data(self, X, mean_bounds=None, variance_bounds=None, k=[1, 5], tolerance=0.05, **kwargs):
        # kwargs['tol'] = 0.001
        return super().fit_to_data(X, mean_bounds, variance_bounds, k, tolerance, **kwargs)


    def plot_components(self, x, log, ax=None, **kwargs):

        if not ax is None:
            plt.sca(ax)

        probability = np.squeeze(self.amplitudes * self.probability(x, log))

        return probability.plot(x=x, c=self.labels, **kwargs)


    def probability(self, x, log, component=None):

        if component is None:
            out = StatArray.StatArray(np.empty([np.size(x), self.n_mixtures]), "Probability Density")
            for i in range(self.n_mixtures):
                out[:, i] = self.amplitudes[i] * self._probability(x, log, self.means[i], self.variances[i], self.degrees[i])
            return out
        else:
            return self.amplitudes[component] * self._probability(x, log, self.means[component], self.variances[component], self.degrees[component])


    def _probability(self, x, log, mean, variance, df):
        """ For a realization x, compute the probability """
        if log:
            return StatArray.StatArray(t.logpdf(x, df, loc = mean, scale = variance), "Probability Density")
        else:
            return StatArray.StatArray(t.pdf(x, df, loc = mean, scale = variance), "Probability Density")


    def sum(self, x):
        return self._sum(x, *self._params)


    def _sum(self, x, *params):

        out = np.zeros_like(x)

        nm = np.int(len(params) / 4)
        for i in range(nm):
            i1 = i*4
            amp, mean, var, df = params[i1:i1+4]
            out += amp * t.pdf(x, df, mean, var)

        return out

    def summary(self, out=False):

        h = '{}'

    def _assign_from_mixture(self, mixture):
        self.__init__(np.squeeze(mixture.means), np.squeeze(mixture.covariances), np.squeeze(mixture.degrees), amplitudes=np.squeeze(mixture.weights))





