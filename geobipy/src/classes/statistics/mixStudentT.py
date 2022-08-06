
import numpy as np
from ...classes.core import StatArray
from scipy.stats import t
from scipy.special import gamma
import matplotlib.pyplot as plt
from .Mixture import Mixture
from lmfit.models import StudentsTModel

class mixStudentT(Mixture):

    def __init__(self, means=None, sigmas=None, amplitudes=1.0, labels=None):

        if np.all([means, sigmas] is None):
            return

        self.params = np.zeros(self.n_solvable_parameters * np.size(means))

        self.amplitudes = amplitudes
        self.means = means
        self.sigmas = sigmas
        # self.degrees = degrees


    @property
    def amplitudes(self):
        return StatArray.StatArray(self._params[0::self.n_solvable_parameters], "Amplitude")

    @amplitudes.setter
    def amplitudes(self, values):
        assert np.size(values) == self.n_components, ValueError("Must provide {} amplitudes".format(self.n_components))
        self._params[0::self.n_solvable_parameters] = values

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, values):
        assert np.size(values) == self.n_components, ValueError("Labels must have size {}".format(self.n_components))
        self._labels[:] = values

    @property
    def lmfit_model(self):
        return StudentsTModel

    @property
    def means(self):
        return StatArray.StatArray(self._params[1::self.n_solvable_parameters], "means")

    @means.setter
    def means(self, values):
        assert np.size(values) == self.n_components, ValueError("Must provide {} means".format(self.n_components))
        self._params[1::self.n_solvable_parameters] = values

    @property
    def moments(self):
        return np.hstack([self.means, self.variances, self.degrees])

    @property
    def sigmas(self):
        return StatArray.StatArray(self._params[2::self.n_solvable_parameters], "standard deviation")

    @sigmas.setter
    def sigmas(self, values):
        assert np.size(values) == self.n_components, ValueError("Must provide {} sigmas".format(self.n_components))
        self._params[2::self.n_solvable_parameters] = values

    @property
    def variances(self):
        return StatArray.StatArray(self.sigmas**2.0, "variance")

    # @property
    # def degrees(self):
    #     return StatArray.StatArray(self._params[3::self.n_solvable_parameters], "degrees of freedom")

    # @degrees.setter
    # def degrees(self, values):
    #     assert np.size(values) == self.n_components, ValueError("Must provide {} degrees".format(self.n_components))
    #     self._params[3::self.n_solvable_parameters] = values

    @property
    def model(self):
        return StudentsTModel

    @property
    def n_solvable_parameters(self):
        return 3

    @property
    def n_components(self):
        return np.size(self.means)


    def fit_to_data(self, X, mean_bounds=None, variance_bounds=None, k=[1, 5], tolerance=0.05, **kwargs):
        # kwargs['tol'] = 0.001
        return super().fit_to_data(X, mean_bounds, variance_bounds, k, tolerance, **kwargs)


    def fit_to_curve(self, *args, **kwargs):
        fit = super().fit_to_curve(*args, **kwargs)
        self.params = np.asarray(list(fit.best_values.values()))
        return self


    def plot_components(self, x, log, ax=None, **kwargs):

        if not ax is None:
            plt.sca(ax)

        probability = np.squeeze(self.amplitudes * self.probability(x, log))

        return probability.plot(x=x, c=self.labels, **kwargs)


    def probability(self, x, log, component=None):

        if component is None:
            out = StatArray.StatArray(np.empty([np.size(x), self.n_components]), "Probability Density")
            for i in range(self.n_components):
                out[:, i] = self.amplitudes[i] * self._probability(x, log, self.means[i], self.sigmas[i])
            return out
        else:
            return self.amplitudes[component] * self._probability(x, log, self.means[component], self.sigmas[component])


    def _probability(self, x, log, mean, sigma):
        """ For a realization x, compute the probability """

        tmp = 0.5 * (sigma + 1.0)
        p = (gamma(tmp) / (np.sqrt(sigma * np.pi) * gamma(0.5 * sigma))) * (1.0 + ((x - mean)**2.0/sigma))**-tmp

        if log:
            return StatArray.StatArray(np.log(p), "Probability Density")
        else:
            return StatArray.StatArray(p, "Probability Density")


    def sum(self, x):
        return self._sum(x, *self._params)


    def _sum(self, x, *params):

        out = np.zeros_like(x)

        nm = np.int(len(params) / self.n_solvable_parameters)
        for i in range(nm):
            i1 = i*4
            amp, mean, var, df = params[i1:i1+4]
            out += amp * t.pdf(x, df, mean, var)/np.float(nm)

        return out

    @property
    def summary(self):

        msg = ('n_components: {}\n'
             '{}\n'
             '{}\n'
             '{}\n'
             '{}\n').format(self.n_components, self.amplitudes.summary, self.means.summary, self.variances.summary, self.degrees.summary)

        return msg

    def _assign_from_mixture(self, mixture):
        self.__init__(np.squeeze(mixture.means), np.squeeze(mixture.covariances), np.squeeze(mixture.degrees), amplitudes=np.squeeze(mixture.weights))
