from copy import deepcopy
import numpy as np
import h5py
from ...classes.core.myObject import myObject
from ..core.DataArray import DataArray
from ...base import utilities as cF
# from scipy.optimize import curve_fit
# from lmfit import models
# from lmfit import Parameters
# from lmfit.model import Model, ModelResult


class Mixture(myObject):

    def __deepcopy__(self, memo={}):
        out = type(self)()
        out._params = deepcopy(self._params, memo=memo)
        return out

    @property
    def amplitudes(self):
        return DataArray(self._params[0::self.n_solvable_parameters], "Amplitude")

    @amplitudes.setter
    def amplitudes(self, values):
        if values is None:
            values = np.ones(self.ndim)
        assert np.size(values) == self.n_components, ValueError("Must provide {} amplitudes".format(self.n_components))
        self._params[0::self.n_solvable_parameters] = values

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, values):
        if values is None:
            values = [str(x) for x in range(self.ndim)]
        assert np.size(values) == self.n_components, ValueError("Must provide {} labels".format(self.n_components))
        self._labels = values

    @property
    def ndim(self):
        return self.n_components

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, values):
        self._params = DataArray(values)

    def plot_components(self, x, log, **kwargs):
        probability = self.amplitudes * self.probability(x, log)
        return probability.plot(x=x, **kwargs)

    def fit_to_curve(self, x, y, n_components=None, plot=False, debug=False, verbose=False, final=False, **kwargs):
        """Iteratively fits the histogram with an increasing number of distributions until the fit changes by less than a tolerance.

        Other Parameters
        ----------------
        norm : float
            Norm to use to measure fit.  1, 2, or np.inf
        epsilon : float
            Tolerance for the fit to change.
        mu : float
            Tolerance for the gradient of the fit to change.
        log : 'e' or float
            Take the log of x before fitting
        max_distributions : int
            Maximum number of distributions to fit.
        method : str
            Method to use for fitting.  'leastsq' or 'lbfgsb'
        masking : float
            Masking factor to ignore data around peaks.
        max_variance : float
            Maximum variance for a distribution.

        """
        import warnings
        warnings.filterwarnings("error")

        if plot:
            import matplotlib.pyplot as plt
            if debug:
                plt.ion()
                fig = plt.figure()

                ax = plt.subplot(121)
                ax2 = plt.subplot(122)
            else:
                fig = plt.gcf()
                ax = plt.gca()

        norm = kwargs.pop('norm', np.inf) 
        epsilon = kwargs.pop('epsilon', 0.05)
        mu = kwargs.pop('mu', 0.1)
        log = kwargs.pop('log', None)
        maxDistributions = kwargs.pop('max_distributions', np.inf)
        kwargs['method'] = kwargs.get('method', 'leastsq')#'lbfgsb')

        masking = kwargs.pop('masking', 1.67)
        dynamic_weighting = kwargs.pop('dynamic_weighting', False)

        x = DataArray(x)

        edges = x.edges()
        centres, dum = cF._log(x, log)
        edges, dum = cF._log(edges, log)

        cdf = np.cumsum(y) / np.max(np.cumsum(y))
        i05 = cdf.searchsorted(0.1)
        i95 = cdf.searchsorted(0.90)
        kwargs['max_variance'] = kwargs.get('max_variance', np.minimum(1.0, centres[i95] - centres[i05]))

        fit_denominator = np.r_[np.linalg.norm(y, ord=np.inf), np.linalg.norm(y, ord=2.0)]

        # Get the first peak that lies between the confidence intervals
        tmp = y.copy()
        tmp[:i05] = 0.0
        tmp[i95:] = 0.0
        new_peak = np.argmax(tmp)
        x_guess = np.atleast_1d(centres[new_peak])

        go = True
        expon_guess = 20.0
        fit = None
        # while go:
            # try:
        fit, pars = self._fit_GM_to_cuve(self.model, centres, edges, y, x_guess, expon_guess=expon_guess, verbose=verbose, **kwargs)
            # except:
            #     pass
            # expon_guess *= 2.0
            # go = expon_guess < 100.0 and fit is None

        x_guess = np.asarray([fit.params['g0_center']])
        sigma_guess = [fit.params['g0_sigma']]
        # expon_guess = [fit.params['g0_expon']]

        kwargs['max_variance'] = centres[i95] - centres[i05]


        residual = y - fit.best_fit
        misfit = np.r_[np.linalg.norm(residual, ord=np.inf), np.linalg.norm(residual, ord=2.0)] / fit_denominator

        residual = np.abs(residual)

        # Get the location of the next peak
        s = fit.params['g0_fwhm'].value
        # s = fit.params['g0_sigma'].value
        residual[np.abs(centres - fit.params['g0_center'].value) < masking*s] = 0.0
        residual[:i05] = 0.0
        residual[i95:] = 0.0

        if (plot or debug) and not final:
            ax.plot(centres, y, '-.')
            ax.plot(centres, fit.best_fit)
            comps = fit.eval_components(x=centres)
            for i in range(x_guess.size):
                ax.plot(centres, comps['g{}_'.format(i)])

            if debug:
                ax2.plot(centres, residual)
                # ax2.vlines(centres[i05], ymin=0.0, ymax=y.max())
                # ax2.vlines(centres[i95], ymin=0.0, ymax=y.max())

            fig.canvas.flush_events()
            fig.canvas.draw()

        if n_components is not None:
            go = n_components > 1
        else:
            go = np.all(misfit > epsilon)
            new_peak = np.argmax(residual)
            go = go and (residual[new_peak] > 0.0) and (maxDistributions > 1)

        if verbose:
            print('first peak at {}'.format(x_guess))
            print('first params {}'.format(np.asarray(list(fit.best_values.values()))))
            print('misfit', misfit)
            print('next peak', centres[new_peak])

        if plot and verbose:
            input('\nNext\n')

        lmfit_barfed = False
        while go:

            x_guess = np.asarray([fit.params['g{}_center'.format(i)] for i in range(x_guess.size)])
            x_guess_test = np.hstack([x_guess, centres[new_peak]])
            x_guess_test = np.atleast_1d(centres[new_peak])

            # try:
            if verbose:
                print('testing peaks', x_guess_test)


            fit_test, pars_test = self._fit_GM_to_cuve(self.model, centres, edges, y, x_guess_test, expon_guess=expon_guess, previous_fit=fit, previous_pars=pars, verbose=verbose, **kwargs)
            n_tests = len(fit_test.components)

            residual = y - fit_test.best_fit
            misfit_test = np.r_[np.linalg.norm(residual, ord=np.inf), np.linalg.norm(residual, ord=2.0)] / fit_denominator
            residual = np.abs(residual)

            fwhm = [fit_test.params['g{}_fwhm'.format(i)].value for i in range(n_tests)]
            # sigma = [fit_test.params['g{}_sigma'.format(i)].value for i in range(n_tests)]
            # s = np.min(fwhm)
            for i in range(n_tests):
                residual[np.abs(centres - fit_test.params['g{}_center'.format(i)].value) < masking*fwhm[i]] = 0.0
                # residual[np.abs(centres - fit_test.params['g{}_center'.format(i)].value) < 1.67*sigma[i]] = 0.0
            residual[:i05] = 0.0
            residual[i95:] = 0.0

            # Conditions to accept new model
            misfit_decreases = (misfit_test[1] - misfit[1]) < 0.0
            gradient_substantial = np.any(np.abs(misfit_test - misfit) > mu)

            accept_model = misfit_decreases and gradient_substantial

            if verbose:
                print('test misfit', misfit_test)
                print('misfit decreases', misfit_test[1] - misfit[1])
                print('gradient substantial', mu, np.abs(misfit_test - misfit))


            # except:
            #     print('failed for blah')
            #     accept_model = False
            #     lmfit_barfed = True


            go = accept_model
            if n_components is not None:
                accept_model = True
            if accept_model:
                # Test a new peak if its possible
                new_peak = np.argmax(residual)

                # conditions to test the next model
                above_abs_threshold = np.all(misfit_test > epsilon)
                gradient_substantial = np.any(np.abs(misfit_test - misfit) > mu)
                valid_new_peak = residual[new_peak] > 0.0 and (i05 <= new_peak <= i95)

                under_limits = len(fit.components) < maxDistributions-1

                if n_components is not None:
                    go = len(fit.components) < n_components - 1
                else:
                    go = valid_new_peak and under_limits and above_abs_threshold

                if verbose:
                    print('\nmodel {}'.format(np.asarray(list(fit_test.best_values.values()))))
                    print('model accepted')

                    print('misfit', misfit_test)
                    print('misfit change should be negative', misfit_test - misfit)
                    print('next peak', centres[new_peak])

                    print('above absolute threshold?', above_abs_threshold)
                    print('valid new peak?', valid_new_peak)

                fit = fit_test
                pars = pars_test
                misfit = misfit_test
                x_guess = x_guess_test

            else:
                if verbose and not lmfit_barfed:
                    print('\nmodel {}'.format(np.asarray(list(fit_test.best_values.values()))))
                    print('test not accepted')
                    print('misfit', misfit_test)
                    print('misfit change should be negative', misfit_test - misfit)


            if (plot or debug) and not lmfit_barfed and not final:
                ax.clear()
                ax.plot(centres, y, '-.')
                comps = fit_test.eval_components(x=centres)
                for i in range(n_tests):
                    ax.plot(centres, comps['g{}_'.format(i)])

                if debug:
                    ax2.clear()
                    ax2.plot(centres, residual)
                    ax2.vlines(centres[i05], ymin=0.0, ymax=y.max())
                    ax2.vlines(centres[i95], ymin=0.0, ymax=y.max())

                fig.canvas.flush_events()
                fig.canvas.draw()
                plt.pause(1e-3)


            if plot and verbose and go:
                input('Next')

        if plot or debug:
            ax.clear()
            ax.plot(centres, y, '-.')
            comps = fit.eval_components(x=centres)
            for i in range(len(fit.components)):
                ax.plot(centres, comps['g{}_'.format(i)])

            if debug:
                ax2.clear()
                ax2.plot(centres, residual)
                ax2.vlines(centres[i05], ymin=0.0, ymax=y.max())
                ax2.vlines(centres[i95], ymin=0.0, ymax=y.max())

            ax.plot(centres, fit.best_fit)

            fig.canvas.flush_events()
            fig.canvas.draw()
            plt.pause(1e-3)

            if debug:
                plt.ioff()

        return fit, pars

    def _fit_GM_to_cuve(self, model, centres, edges, y, x_guess, sigma_guess=None, expon_guess=None, previous_fit=None, previous_pars=None, verbose=False, **kwargs):

        if previous_pars is None:
            return self.__fit_wo_previous(model, centres, edges, y, x_guess, sigma_guess=sigma_guess, expon_guess=expon_guess, verbose=verbose,**kwargs)
        else:
            return self.__fit_w_previous(model, centres, edges, y, x_guess, sigma_guess, expon_guess=expon_guess,previous_fit=previous_fit, previous_pars=previous_pars, verbose=verbose, **kwargs)

    def __fit_w_previous(self, model, centres, edges, y, x_guess, sigma_guess=None, expon_guess=None, previous_fit=None, previous_pars=None, verbose=False, window=1, **kwargs):


        mn_var = kwargs.pop('min_variance', (edges[1]-edges[0]))
        mx_var = kwargs.pop('max_variance', None)
        init = 1.0 if mx_var is None else np.min([0.5 * mx_var, 1.0])

        mod = previous_fit.components[0]
        n_previous = len(previous_fit.components)
        for i in range(1, n_previous):
            mod += previous_fit.components[i]
        pars = previous_pars

        weights = np.full(y.size, fill_value=0.1)
        for j in range(np.size(x_guess)):
            i = j + n_previous
            guess = model(prefix='g{}_'.format(i))
            pars.update(guess.make_params())
            mod += guess

            k = np.searchsorted(edges, x_guess[j])
            le = edges[np.maximum(0, k-window)]
            ue = edges[np.minimum(edges.size, k+window+1)]

            weights[k-window:k+window+1] = 1.0

            pars[f'g{i}_center'].set(value=x_guess[j], min=le, max=ue)
            pars[f'g{i}_sigma'].set(value=init, min=mn_var, max=mx_var)
            pars[f'g{i}_amplitude'].set(value=1.0, min=0.0)
            key = f'g{i}_expon'
            if key in pars:
                tmp = 10.5 #if expon_guess is None else expon_guess
                pars[key].set(value=tmp, vary=False)

        kwargs['weights'] = weights


        if verbose:
            print('fitting with', pars)

        init = mod.eval(pars, x=centres)
        out = mod.fit(y, pars, x=centres, **kwargs)

        return out, pars


    def __fit_wo_previous(self, model, centres, edges, y, x_guess, expon_guess=None, sigma_guess=None, verbose=False, window=1, **kwargs):

        n_guesses = np.size(x_guess)
        # Create the first model
        guess = model(prefix='g0_')
        pars = guess.make_params()
        mod = guess
        for i in range(1, n_guesses):
            guess = model(prefix='g{}_'.format(i))
            pars.update(guess.make_params())
            mod += guess

        mn_var = kwargs.pop('min_variance', (edges[1]-edges[0]))
        mx_var = kwargs.pop('max_variance', None)
        init = 1.0 if mx_var is None else np.min([0.5 * mx_var, 1.0])

        # Sort the guesses and sigmas
        ix = np.argsort(x_guess)

        weights = np.full(y.size, fill_value=0.1)
        for i in range(n_guesses):
            k = np.searchsorted(edges, x_guess[ix[i]])
            le = edges[np.maximum(0, k-window)]
            ue = edges[np.minimum(edges.size, k+window+1)]

            weights[k-window:k+window+1] = 1.0

            pars[f'g{i}_center'].set(value=x_guess[ix[i]], min=le, max=ue)
            pars[f'g{i}_sigma'].set(value=init, min=mn_var, max=mx_var)
            pars[f'g{i}_amplitude'].set(value=1.0, min=0.0)
            key = f'g{i}_expon'
            if key in pars:
                tmp = 10.5 if expon_guess is None else expon_guess
                pars[key].set(value=tmp, vary=False)

        kwargs['weights'] = weights

        if verbose:
            print('fitting with', pars)

        init = mod.eval(pars, x=centres)
        out = mod.fit(y, pars, x=centres, **kwargs)

        return out, pars


    def createHdf(self, parent, name, add_axis=None):
        """Create space in a HDF file for mixtures

        Parameters
        ----------
        h5obj : h5py.File or h5py.Group
            A HDF file or group object to create the contents in.
        myName : str
            The name of the group to create.
        nRepeats : int, optional
            Inserts a first dimension size nRepeats. This can be used to extend the available memory
            so that multiple MPI ranks can write to their respective parts in the extended memory.

        """
        grp = self.create_hdf_group(parent, name)
        grp = self.params.createHdf(grp, 'params', add_axis=add_axis)

        return grp


    def writeHdf(self, parent, name, index=None):
        """Write the mixture to HDF.

        Parameters
        ----------
        h5obj : h5py._hl.files.File or h5py._hl.group.Group
            A HDF file or group object to write the contents to.
        myName : str
            The name of the group to write to. The group must have been created previously.
        index : int, optional
            If the group was created using the nRepeats option, index specifies the index'th entry at which to write the data

        """
        grp = parent.get(name)
        self.params.writeHdf(grp, 'params', index=index)


    def fromHdf(self, grp, index=None):
        """Read the mixture from a HDF group

        Parameters
        ----------
        h5obj : h5py._hl.group.Group
            A HDF group object to write the contents to.
        index : slice, optional
            If the group was created using the nRepeats option, index specifies the index'th entry from which to read the data.

        """
        # assert np.size(index) == np.size(item['data'].shape) - 1, ValueError('Need to specify a {}D index'.format(np.size(item['data'].shape)-1))

        self.params = DataArray.fromHdf(grp['params'], index=index)
        return self.squeeze()

    def plot_components(self, x, log, ax=None, **kwargs):

        if not ax is None:
            plt.sca(ax)

        probability = np.squeeze(self.amplitudes * self.probability(x, log))

        return probability.plot(x=x, **kwargs)