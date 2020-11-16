import numpy as np
import h5py
from ...classes.core.myObject import myObject
from ...classes.core import StatArray
from ...base import customFunctions as cF
from scipy.optimize import curve_fit
from lmfit import models
from lmfit import Parameters
from lmfit.model import Model, ModelResult

class Mixture(myObject):

    def __init__(self, mixture_type=None):

        if mixture_type is None:
            return

        mixture = mixture_type.lower()
        if mixture == 'gaussian':
            lmfit_model = models.GaussianModel
        elif mixture == 'lorentzian':
            lmfit_model = models.LorentzianModel
        elif mixture == 'splitlorentzian':
            lmfit_model = models.SplitLorentzianModel
        elif mixture == 'voigt':
            lmfit_model = models.VoigtModel
        elif mixture == 'moffat':
            lmfit_model = models.MoffatModel
        elif mixture == 'pearson':
            lmfit_model = models.Pearson7Model
        elif mixture == 'studentst':
            lmfit_model = models.StudentsTModel
        # elif mixture == 'exponentialgaussian':
        #     from lmfit.models import ExponentialGaussianModel as lmfit_model
        # elif mixture == 'skewedgaussian':
        #     from lmfit.models import SkewedGaussianModel as lmfit_model
        # elif mixture == 'exponential':
        #     from lmfit.models import ExponentialModel as lmfit_model
        # elif mixture == 'powerlaw':
        #     from lmfit.models import PowerLawModel as lmfit_model

        else:
            raise ValueError("mixture must be one of [gaussian, lorentzian, splitlorentzian, voigt, moffat, pearson, studentst]")

        self.model = lmfit_model


    def fit_to_curve(self, x, y, plot=False, verbose=False, **kwargs):
        """Iteratively fits the histogram with an increasing number of distributions until the fit changes by less than a tolerance.

        """
        if plot:
            import matplotlib.pyplot as plt
            plt.ion()
            fig = plt.figure()
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
        norm = kwargs.pop('norm', np.inf)
        epsilon = kwargs.pop('epsilon', 0.05)
        mu = kwargs.pop('mu', 0.1)
        log = kwargs.pop('log', None)
        maxDistributions = kwargs.pop('max_distributions', np.inf)
        kwargs['method'] = kwargs.get('method', 'lbfgsb')


        x = StatArray.StatArray(x)
        edges = x.edges()
        centres, dum = cF._log(x, log)
        edges, dum = cF._log(edges, log)

        cdf = np.cumsum(y) / np.max(np.cumsum(y))
        i05 = cdf.searchsorted(0.1)
        i95 = cdf.searchsorted(0.90)
        kwargs['max_variance'] = np.minimum(1.0, centres[i95] - centres[i05])

        fit_denominator = np.r_[np.linalg.norm(y, ord=np.inf), np.linalg.norm(y, ord=2.0)]

        # Only fit the non-zero counts otherwise heavy tails can dominate
        new_peak = np.argmax(y)
        x_guess = np.atleast_1d(centres[new_peak])

        fit = self._fit_GM_to_cuve(self.model, centres, edges, y, x_guess, **kwargs)

        kwargs['max_variance'] = centres[i95] - centres[i05]


        residual = y - fit.best_fit
        misfit = np.r_[np.linalg.norm(residual, ord=np.inf), np.linalg.norm(residual, ord=2.0)] / fit_denominator

        residual = np.abs(residual)

        # Get the location of the next peak
        s = fit.params['g0_sigma'].value
        residual[np.abs(centres - fit.params['g0_center'].value) < 2.0*s] = 0.0
        residual[:i05] = 0.0
        residual[i95:] = 0.0


        if plot:
            ax1.plot(centres, y, '-.')
            ax1.plot(centres, fit.best_fit)
            comps = fit.eval_components(x=centres)
            for i in range(x_guess.size):
                ax1.plot(centres, comps['g{}_'.format(i)])

            ax2.plot(centres, residual)
            ax2.vlines(centres[i05], ymin=0.0, ymax=y.max())
            ax2.vlines(centres[i95], ymin=0.0, ymax=y.max())

            fig.canvas.flush_events()
            fig.canvas.draw()

        go = np.all(misfit > epsilon)
        new_peak = np.argmax(residual)
        go = go and (residual[new_peak] > 0.0) and (maxDistributions > 1)

        if verbose:
            print('first peak at {}'.format(x_guess))
            print('misfit', misfit)
            print('next peak', centres[new_peak])

        if plot and verbose:
            input('\nNext\n')

        while go:

            x_guess = np.asarray([fit.params['g{}_center'.format(i)] for i in range(x_guess.size)])
            x_guess_test = np.hstack([x_guess, centres[new_peak]])

            fit_test = self._fit_GM_to_cuve(self.model, centres, edges, y, x_guess_test, **kwargs)

            residual = y - fit_test.best_fit
            misfit_test = np.r_[np.linalg.norm(residual, ord=np.inf), np.linalg.norm(residual, ord=2.0)] / fit_denominator
            residual = np.abs(residual)

            fwhm = [fit_test.params['g{}_fwhm'.format(i)].value for i in range(x_guess_test.size)]
            s = np.min(fwhm)
            for i in range(x_guess_test.size):
                residual[np.abs(centres - fit_test.params['g{}_center'.format(i)].value) < s] = 0.0
            residual[:i05] = 0.0
            residual[i95:] = 0.0

            # Conditions to accept new model
            misfit_decreases = (misfit_test[1] - misfit[1]) < 0.0
            gradient_substantial = np.any(np.abs(misfit_test - misfit) > mu)

            accept_model = misfit_decreases and gradient_substantial

            if verbose:
                print('testing peaks', x_guess_test)

            go = accept_model
            if accept_model:
                # Test a new peak if its possible
                new_peak = np.argmax(residual)

                # conditions to test the next model
                above_abs_threshold = np.all(misfit_test > epsilon)
                gradient_substantial = np.any(np.abs(misfit_test - misfit) > mu)
                valid_new_peak = residual[new_peak] > 0.0 and (i05 <= new_peak <= i95)

                under_limits = x_guess.size < maxDistributions

                go = valid_new_peak and under_limits and above_abs_threshold and gradient_substantial

                if verbose:
                    print('\nmodel accepted')

                    print('misfit', misfit_test)
                    print('misfit change should be negative', misfit_test - misfit)
                    print('next peak', centres[new_peak])

                    print('above absolute threshold?', above_abs_threshold)
                    print('valid new peak?', valid_new_peak)

                fit = fit_test
                misfit = misfit_test
                x_guess = x_guess_test

            else:
                if verbose:
                    print('\ntest not accepted')
                    print('misfit', misfit_test)
                    print('misfit change should be negative', misfit_test - misfit)


            if plot:
                ax1.clear()
                ax1.plot(centres, y, '-.')
                comps = fit_test.eval_components(x=centres)
                for i in range(x_guess_test.size):
                    ax1.plot(centres, comps['g{}_'.format(i)])

                ax2.clear()
                ax2.plot(centres, residual)
                ax2.vlines(centres[i05], ymin=0.0, ymax=y.max())
                ax2.vlines(centres[i95], ymin=0.0, ymax=y.max())

                fig.canvas.flush_events()
                fig.canvas.draw()
                plt.pause(1e-3)


            if plot and verbose and go:
                input('Next')

        # if verbose:
        #     print('Final misfit with gaussians', misfit)

        #     from lmfit.models import Pearson7Model
        #     fit_test = self._fit_GM_to_cuve(Pearson7Model, centres, edges, y, iPeaks, **kwargs)
        #     residual = y - fit_test.best_fit

        #     misfit_test = np.r_[np.linalg.norm(residual, ord=np.inf), np.linalg.norm(residual, ord=2.0)] / fit_denominator

        #     print('misfit with Pearson', misfit_test)

        #     if np.any(misfit_test < misfit):
        #         fit = fit_test

        if plot:
            ax1.clear()
            ax1.plot(centres, y, '-.')
            comps = fit.eval_components(x=centres)
            for i in range(x_guess.size):
                ax1.plot(centres, comps['g{}_'.format(i)])

            ax2.clear()
            ax2.plot(centres, residual)
            ax2.vlines(centres[i05], ymin=0.0, ymax=y.max())
            ax2.vlines(centres[i95], ymin=0.0, ymax=y.max())

            ax1.plot(centres, fit.best_fit)

            fig.canvas.flush_events()
            fig.canvas.draw()
            plt.pause(1e-3)

            plt.ioff()

        self.model = fit
        return fit


    def _fit_GM_to_cuve(self, model, centres, edges, y, x_guess, **kwargs):

        guess = model(prefix='g0_')
        pars = guess.make_params()
        mod = guess
        for i in range(1, x_guess.size):
            guess = model(prefix='g{}_'.format(i))
            pars.update(guess.make_params())
            mod += guess

        mn = kwargs.pop('min_variance', 3.0*(edges[1]-edges[0]))
        mx = kwargs.pop('max_variance', None)
        if mx is None:
            init = 1.0
        else:
            init = np.min([0.5 * mx, 1.0])

        for i in range(x_guess.size):
            vary = True
            if i == 0:
                vary = False
            ix = centres.searchsorted(x_guess[i])

            pars['g{}_center'.format(i)].set(value=x_guess[i],vary=vary)#, min=edges[ix], max=edges[ix+1])
            pars['g{}_sigma'.format(i)].set(value=init, min=mn, max=mx)
            pars['g{}_amplitude'.format(i)].set(value=1.0, min=1e-3)

        init = mod.eval(pars, x=centres)
        out = mod.fit(y, pars, x=centres, **kwargs)

        return out


    # def _single_fit_to_curve(self, centres, edges, y, iPeaks, variance_bound, **kwargs):

    #     log = kwargs.pop('log', None)

    #     nPeaks = np.size(iPeaks)
    #     # Carry out the first fitting.
    #     guess = np.ones(nPeaks * self.n_solvable_parameters)
    #     lowerBounds = np.zeros(nPeaks * self.n_solvable_parameters)
    #     upperBounds = np.full(nPeaks * self.n_solvable_parameters, np.inf)

    #     # Set the mean bounds
    #     guess[1::self.n_solvable_parameters] = centres[iPeaks]
    #     lowerBounds[1::self.n_solvable_parameters] = edges[iPeaks]
    #     upperBounds[1::self.n_solvable_parameters] = edges[iPeaks+1]

    #     # Set the variance bounds
    #     upperBounds[2::self.n_solvable_parameters] = variance_bound

    #     if np.isinf(variance_bound):
    #         guess[2::self.n_solvable_parameters] = 1.0
    #     else:
    #         guess[2::self.n_solvable_parameters] = 0.5 * (lowerBounds[2::self.n_solvable_parameters] + upperBounds[2::self.n_solvable_parameters])

    #     if self.n_solvable_parameters > 3:
    #         dfGuess = 1e4
    #         # Set the degrees of freedom bounds
    #         guess[3::self.n_solvable_parameters] = dfGuess
    #         lowerBounds[3::self.n_solvable_parameters] = 2

    #     bounds = (lowerBounds, upperBounds)

    #     iWhere = np.where(y > 0.0)[0]

    #     model, pcov = curve_fit(self._sum, xdata=centres, ydata=y, p0=guess, bounds=bounds, ftol=1e-3, method='dogbox', **kwargs)

    #     return np.asarray(model)


    def createHdf(self, h5obj, myName, shape=(1, )):
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

        dt = h5py.special_dtype(vlen=str)

        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdf_name
        grp.create_dataset('data', shape, dtype=dt)

        return grp


    def writeHdf(self, h5obj, myName, index=0):
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


        s = self.model.dumps()
        h5obj[myName + '/data'][index] = s


    def fromHdf(self, h5grp, index=0):
        """Read the mixture from a HDF group

        Parameters
        ----------
        h5obj : h5py._hl.group.Group
            A HDF group object to write the contents to.
        index : slice, optional
            If the group was created using the nRepeats option, index specifies the index'th entry from which to read the data.

        """
        s = h5grp['data'][index]
        self.model = ModelResult(Model(lambda x: x, None), Parameters()).loads(s)
        return self