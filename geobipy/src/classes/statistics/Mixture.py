import numpy as np
from ...classes.core import StatArray
from ...base import customFunctions as cF
from scipy.optimize import curve_fit


class Mixture(object):

    def __init__(self):
        raise NotImplementedError


    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, values):

        assert np.size(values) % self.n_solvable_parameters == 0, ValueError("Size of Params must be a multiple of {}.".format(self.n_solvable_parameters))
        self._params = np.asarray(values)


    def fit_to_curve(self, x, y, **kwargs):
        """Iteratively fits the histogram with an increasing number of distributions until the fit changes by less than a tolerance.

        """
        norm = kwargs.pop('norm', np.inf)
        epsilon = kwargs.pop('epsilon', 0.05)
        mu = kwargs.pop('mu', 0.05)
        log = kwargs.get('log', None)
        maxDistribuions = kwargs.pop('max_distributions', np.inf)

        x = StatArray.StatArray(x)
        edges = x.edges()
        centres, dum = cF._log(x, log)
        edges, dum = cF._log(edges, log)

        fit_denominator = np.linalg.norm(y, ord=norm)

        # Only fit the non-zero counts otherwise heavy tails can dominate
        iPeaks = np.argmax(y)
        keep = np.ones(1, dtype=np.bool)

        model = self._single_fit_to_curve(centres, edges, y, iPeaks, **kwargs)

        yFit = self._sum(centres, *model)

        currentMeans = model[1::self.n_solvable_parameters]
        currentStd = np.sqrt(model[2::self.n_solvable_parameters])

        residual = y - yFit
        fit = np.linalg.norm(residual, ord=norm) / fit_denominator

        for i in range(iPeaks.size):
            residual[np.abs(centres - currentMeans[i]) < 1.1*currentStd[i]] = 0.0
        residual[residual < 0.0] = 0.0

        new_peak = np.argmax(residual)
        go = (fit > epsilon) and (y[new_peak] > 0.0) and (maxDistribuions > 1)

        while go:

            iPeaks = np.hstack([iPeaks, new_peak])

            model = self._single_fit_to_curve(centres, edges, y, iPeaks, **kwargs)

            yFit = self._sum(centres, *model)
            fit0 = fit

            currentMeans = model[1::self.n_solvable_parameters]
            currentStd = np.sqrt(model[2::self.n_solvable_parameters])

            residual = y - yFit
            fit = np.linalg.norm(residual, ord=norm) / fit_denominator

            for i in range(iPeaks.size):
                residual[np.abs(centres - currentMeans[i]) < 1.1*currentStd[i]] = 0.0
            residual[residual < 0.0] = 0.0

            new_peak = np.argmax(residual)

            fit = np.linalg.norm((y - yFit), ord=norm) / fit_denominator

            go = (fit > epsilon) and (fit0 - fit > mu) and y[new_peak] > 0.0


        nG = np.int(len(model)/self.n_solvable_parameters)
        if not maxDistribuions is None:
            if nG > maxDistribuions:
                model = model[:maxDistribuions*self.n_solvable_parameters]

        self.params = model


    def _single_fit_to_curve(self, centres, edges, y, iPeaks, variance_bound, **kwargs):

        log = kwargs.pop('log', None)

        nPeaks = np.size(iPeaks)
        # Carry out the first fitting.
        guess = np.ones(nPeaks * self.n_solvable_parameters)
        lowerBounds = np.zeros(nPeaks * self.n_solvable_parameters)
        upperBounds = np.full(nPeaks * self.n_solvable_parameters, np.inf)

        # Set the mean bounds
        guess[1::self.n_solvable_parameters] = centres[iPeaks]
        lowerBounds[1::self.n_solvable_parameters] = edges[iPeaks]
        upperBounds[1::self.n_solvable_parameters] = edges[iPeaks+1]

        # Set the variance bounds
        upperBounds[2::self.n_solvable_parameters] = variance_bound

        if np.isinf(variance_bound):
            guess[2::self.n_solvable_parameters] = 1.0
        else:
            guess[2::self.n_solvable_parameters] = 0.5 * (lowerBounds[2::self.n_solvable_parameters] + upperBounds[2::self.n_solvable_parameters])

        if self.n_solvable_parameters > 3:
            dfGuess = 1e4
            # Set the degrees of freedom bounds
            guess[3::self.n_solvable_parameters] = dfGuess
            lowerBounds[3::self.n_solvable_parameters] = 2

        bounds = (lowerBounds, upperBounds)

        iWhere = np.where(y > 0.0)[0]

        model, pcov = curve_fit(self._sum, xdata=centres, ydata=y, p0=guess, bounds=bounds, ftol=1e-3, **kwargs)

        return np.asarray(model)


    def fit_to_data(self, X, mean_bounds=None, variance_bounds=None, k=[1, 5], tolerance=0.05, **kwargs):
        """Uses a mixture models to fit data.

        Starts at the minimum number of clusters and adds clusters until the BIC decreases less than the tolerance.

        Parameters
        ----------
        nSamples

        log

        mean_bounds

        variance_bounds

        k : ints
            Two ints with starting and ending # of clusters

        tolerance

        """
        # of clusters
        k_ = k[0]

        best = self.mixture_model_class(n_components=k_, **kwargs).fit(X)
        BIC0 = best.bic(X)

        k_ += 1
        go = k_ <= k[1]

        while go:
            model = self.mixture_model_class(n_components=k_, **kwargs).fit(X)
            BIC1 = model.bic(X)

            percent_reduction = np.abs((BIC1 - BIC0)/BIC0)

            go = True
            if BIC1 < BIC0 and percent_reduction > tolerance:
                best = model
                BIC0 = BIC1

            else:
                go = False

            k_ += 1
            go = go & (k_ <= k[1])


        active = np.ones(best.n_components, dtype=np.bool)

        means = np.squeeze(best.means_)
        try:
            variances = np.squeeze(best.covariances_)
        except:
            variances = np.squeeze(best.covariances)

        if not mean_bounds is None:
            active = (mean_bounds[0] <= means) & (means <= mean_bounds[1])

        if not variance_bounds is None:
            active = (variance_bounds[0] <= variances) & (variances <= variance_bounds[1]) & active

        self._assign_from_mixture(best)
        return best, np.atleast_1d(active)


    def fit_single_mixture(self , X, k, **kwargs):
        mixture = self.mixture_model_class(n_components=k, **kwargs).fit(X)
        self._assign_from_mixture(mixture)
        return mixture
