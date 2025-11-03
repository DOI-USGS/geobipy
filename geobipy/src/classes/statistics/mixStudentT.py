
import numpy as np
from ..core.DataArray import DataArray
from scipy.stats import t
from scipy.special import gamma
import matplotlib.pyplot as plt
from .mixNormal import mixNormal
from lmfit.models import StudentsTModel

class mixStudentT(mixNormal):

    def __init__(self, means=None, sigmas=None, amplitudes=None, labels=None):

        if np.all([means, sigmas] is None):
            return

        super().__init__(amplitudes=amplitudes, means=means, sigmas=sigmas, labels=labels)

    @property
    def amplitudes(self):
        return DataArray(self._params[0::self.n_solvable_parameters], "Amplitude")

    @amplitudes.setter
    def amplitudes(self, values):
        if values is None:
            values = np.ones(self.n_components)
        self._params[0::self.n_solvable_parameters] = values


    @property
    def lmfit_model(self):
        return StudentsTModel

    @property
    def model(self):
        return StudentsTModel

    def fit_to_data(self, X, mean_bounds=None, variance_bounds=None, k=[1, 5], tolerance=0.05, **kwargs):
        # kwargs['tol'] = 0.001
        return super().fit_to_data(X, mean_bounds, variance_bounds, k, tolerance, **kwargs)

    def _probability(self, x, log, mean, sigma):
        """ For a realization x, compute the probability """

        tmp = 0.5 * (sigma + 1.0)
        p = (gamma(tmp) / (np.sqrt(sigma * np.pi) * gamma(0.5 * sigma))) * (1.0 + ((x - mean)**2.0/sigma))**-tmp

        if log:
            p = np.log(p)

        return DataArray(p, "Probability Density")

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
        self.__init__(np.squeeze(mixture.means), np.squeeze(mixture.covariances), amplitudes=np.squeeze(mixture.weights))
