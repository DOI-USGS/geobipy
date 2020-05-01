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
        tolerance = kwargs.pop('tolerance', 0.05)
        log = kwargs.get('log', None)
        maxDistribuions = kwargs.pop('max_distributions', None)

        x = StatArray.StatArray(x)
        edges = x.edges()
        centres, dum = cF._log(x, log)
        edges, dum = cF._log(edges, log)
        bin_width = centres[1] - centres[0]

        fit_denominator = np.linalg.norm(y, ord=norm)

        # Only fit the non-zero counts otherwise heavy tails can dominate
        iPeaks = np.argmax(y)
        keep = np.ones(1, dtype=np.bool)
        nPeaks = 1

        model = self._single_fit_to_curve(centres, edges, y, iPeaks, **kwargs)

        yFit = self._sum(centres, *model)
        fit = np.linalg.norm((y - yFit), ord=norm) / fit_denominator

        new_peak = np.argmax(y - yFit)
        go = fit > tolerance
        go = go and y[new_peak] > 0.0

        while go:

            currentMeans = model[1::self.n_solvable_parameters]
            currentStd = np.sqrt(model[2::self.n_solvable_parameters])
            keep = np.hstack([keep, np.all(np.abs(currentMeans - centres[new_peak]) > currentStd)])

            iPeaks = np.hstack([iPeaks, new_peak])
            nPeaks += 1

            model = self._single_fit_to_curve(centres, edges, y, iPeaks, **kwargs)

            yFit = self._sum(centres, *model)
            fit0 = fit
            fit = np.linalg.norm((y - yFit), ord=norm) / fit_denominator

            go = fit > tolerance and (fit0 - fit > 0.05)
            go = go and y[new_peak] > 0.0

        if np.any(~keep):
            model = self._single_fit_to_curve(centres, edges, y, iPeaks[keep], **kwargs)

        nG = np.int(len(model)/self.n_solvable_parameters)
        if not maxDistribuions is None:
            if nG > maxDistribuions:
                model = model[:maxDistribuions*self.n_solvable_parameters]

        self.params = model


    def _single_fit_to_curve(self, centres, edges, y, iPeaks, constrain_loc = True, variance_bounds=[0.0, np.inf], **kwargs):

        log = kwargs.pop('log', None)

        nPeaks = np.size(iPeaks)
        # Carry out the first fitting.
        guess = np.ones(nPeaks * self.n_solvable_parameters)
        lowerBounds = np.zeros(nPeaks * self.n_solvable_parameters)
        upperBounds = np.full(nPeaks * self.n_solvable_parameters, np.inf)

        # Set the mean bounds
        guess[1::self.n_solvable_parameters] = centres[iPeaks]
        if constrain_loc:
            lowerBounds[1::self.n_solvable_parameters] = edges[iPeaks]
            upperBounds[1::self.n_solvable_parameters] = edges[iPeaks+1]
        else:
            lowerBounds[1::self.n_solvable_parameters] = -1e20
            upperBounds[1::self.n_solvable_parameters] = 1e20

        # Set the variance bounds
        lowerBounds[2::self.n_solvable_parameters] = variance_bounds[0]
        upperBounds[2::self.n_solvable_parameters] = variance_bounds[1]
        if np.isinf(variance_bounds[1]):
            guess[2::self.n_solvable_parameters] = 1.0
        else:
            guess[2::self.n_solvable_parameters] = 0.5 * (lowerBounds[2::self.n_solvable_parameters] + upperBounds[2::self.n_solvable_parameters])

        if self.n_solvable_parameters > 3:
            dfGuess = 1e4
            # Set the degrees of freedom bounds
            guess[3::self.n_solvable_parameters] = dfGuess

        bounds = (lowerBounds, upperBounds)

        iWhere = np.where(y > 0.0)[0]

        model, pcov = curve_fit(self._sum, xdata=centres, ydata=y, p0=guess, bounds=bounds, ftol=1e-3, **kwargs)
        model = np.asarray(model)

        return model


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

        return best, np.atleast_1d(active)


    def fit_single_mixture(self , X, k, **kwargs):
        return self.mixture_model_class(n_components=k, **kwargs).fit(X)
