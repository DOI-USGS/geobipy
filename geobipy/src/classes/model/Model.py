""" @Model_Class
Module describing a Model
"""
from copy import deepcopy

from numpy import abs, any, arange, argpartition, argsort, argwhere, asarray, column_stack
from numpy import cumsum, diag, diff, dot, empty, exp, hstack, inf, isinf, maximum, mean, meshgrid
from numpy import ones, ravel_multi_index, s_, sign, size, squeeze, unique, vstack, zeros
from numpy import log as nplog
from numpy.linalg import inv
from matplotlib.pyplot import gcf
from ...base.utilities import reslice
from ...base import plotting
from ..core.myObject import myObject
from ...base.HDF import hdfRead
from ..core import StatArray
from ..mesh.Mesh import Mesh
from ..statistics.Distribution import Distribution
from ..statistics.baseDistribution import baseDistribution

class Model(myObject):
    """Generic model class with an attached mesh.

    """
    def __init__(self, mesh=None, values=None):
        """ Instantiate a 2D histogram """
        # if (mesh is None):
        #     return
        # Instantiate the parent class
        self._mesh = mesh
        self.values = values
        self._gradient = None

        self.value_bounds = None
        # self._inverse_hessian = None

    def __getitem__(self, slic):
        mesh = self.mesh[slic]
        out = type(self)(mesh, values = self.values[slic])
        return out

    def __deepcopy__(self, memo={}):
        mesh = deepcopy(self.mesh, memo=memo)

        out = type(self)(mesh=mesh)

        out._values = deepcopy(self.values, memo=memo)
        out._gradient = deepcopy(self._gradient, memo=memo)
        out.value_bounds = deepcopy(self.value_bounds)
        # out._inverse_hessian = deepcopy(self._inverse_hessian, memo=memo)

        # if len(self.__values) > 1:
        #     for k, v in zip(self.__values, self.cell_values[1:]):
        #         out.setattr(k, deepcopy(v, memo=memo))

        return out

    @property
    def addressof(self):
        msg =  '{}: {}\n'.format(type(self).__name__, hex(id(self)))
        msg += "Mesh:\n{}".format(("|   "+self.mesh.addressof.replace("\n", "\n|   "))[:-4])
        msg += "Values:\n{}".format(("|   "+self.values.addressof.replace("\n", "\n|   "))[:-4])
        return msg

    @property
    def gradient(self):
        """Compute the gradient

        Parameter gradient :math:`\\nabla_{z}\sigma` at the ith layer is computed via

        .. math::
            :label: dpdz1

            \\nabla_{z}^{i}\sigma = \\frac{\sigma_{i+1} - \\sigma_{i}}{h_{i} - h_{min}}

        where :math:`\sigma_{i+1}` and :math:`\sigma_{i}` are the log-parameters on either side of an interface, :math:`h_{i}` is the log-thickness of the ith layer, and :math:`h_{min}` is the minimum log thickness defined by

        .. math::
            :label: minThickness1

            h_{min} = \\frac{z_{max} - z_{min}}{2 k_{max}}

        where :math:`k_{max}` is a maximum number of layers, set to be far greater than the expected final solution.

        """
        # if self._gradient is None:
        #     self._gradient = StatArray.StatArray(self.mesh.nCells.item()-1, 'Derivative', r"$\frac{"+self.values.units+"}{"+self.mesh.edges.units+"}$")

        gradient = StatArray.StatArray(self.mesh.gradient(values=self.values), 'Derivative', r"$\frac{"+self.values.units+"}{"+self.mesh.edges.units+"}$")
        if self._gradient is not None:
            gradient.copyStats(self._gradient)

        self._gradient = gradient
        return self._gradient

    # @gradient.setter
    # def gradient(self, values):
    #     if values is None:
    #
    #         return

    #     self._gradient = StatArray.StatArray(values)

    # @property
    # def inverse_hessian(self):
    #     return self._inverse_hessian

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        assert isinstance(value, Mesh), TypeError('mesh must be instance of geobipy.Mesh')
        self._mesh = value

    @property
    def nCells(self):
        return self.mesh.nCells

    @property
    def shape(self):
        return self.mesh.shape

    @property
    def summary(self):
        """Summary of self """
        msg =  "{}:\n".format(type(self).__name__)
        msg += "mesh:\n{}".format("|   "+(self.mesh.summary.replace("\n", "\n|   "))[:-4])
        msg += "values:\n{}\n".format("|   "+(self.values.summary.replace("\n", "\n|   "))[:-4])
        return msg

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        if value is None:
            self._prior = None
            return
        assert isinstance(value, Model), TypeError('Model prior must have type Model')
        self._prior = value

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        if values is None:
            self._values = StatArray.StatArray(self.shape)
            return

        # assert all(values.shape == self.shape), ValueError("values must have shape {}".format(self.shape))
        self._values = StatArray.StatArray(values)

    @property
    def x(self):
        return self.mesh.x

    @property
    def y(self):
        return self.mesh.y

    @property
    def z(self):
        return self.mesh.z

    @property
    def Earth(self):
        try:
            from gatdaem1d import Earth
        except Exception as e:
            raise Exception("{}\n gatdaem1d is not installed. Please see instructions".format(e))

        return Earth(self.values[:], self.mesh.widths[:-1])


    def animate(self, axis, filename, slic=None, **kwargs):
        return self.mesh._animate(self.values, axis, filename, slic, **kwargs)

    def axis(self, *args, **kwargs):
        return self.mesh.axis(*args, **kwargs)

    def bar(self, **kwargs):
        return self.mesh.bar(self.values, **kwargs)

    def cellIndex(self, *args, **kwargs):
        return self.mesh.cellIndex(*args, **kwargs)

    def cell_indices(self, *args, **kwargs):
        return self.cellIndices(self, *args, **kwargs)

    def cellIndices(self, values, clip=True):
        return self.mesh.cellIndices(values, clip=clip)

    def compute_local_inverse_hessian(self, observation=None):
        """Generate a localized Hessian matrix using
        a dataPoint and the current realization of the Model1D.


        Parameters
        ----------
        observation : geobipy.DataPoint, geobipy.Dataset, optional
            The observed data to use when computing the local estimate of the variance.

        Returns
        -------
        out : array_like
            Hessian matrix

        """
        # Compute a new parameter variance matrix if the structure of the model changed.
        # if (self.mesh.action[0] in ['insert', 'delete']):
            # Propose new layer conductivities
        return self.local_variance(observation)
        # else:
        #     return self.values.proposal.variance

    def delete_edge(self, i):
        out, values = self.mesh.delete_edge(i, values=self.values)
        out = type(self)(mesh=out, values=values)

        # if len(values) > 1:
        #     for k, v in zip(self.__values, values[1:]):
        #         out.setattr(k, v)

        return out

    def gradient_probability(self, log=True):
        """Evaluate the prior for the gradient of the parameter with depth

        **Prior on the gradient of the physical parameter with depth**

        If instead we wish to apply a multivariate normal distribution to the gradient of the parameter with depth we get

        .. math::
            :label: gradient1

            p(\\boldsymbol{\sigma} | k, I) = \\left[(2\pi)^{k-1} |\\boldsymbol{C}_{\\nabla_{z}}|\\right]^{-\\frac{1}{2}} e ^{-\\frac{1}{2}(\\nabla_{z} \\boldsymbol{\sigma})^{T} \\boldsymbol{C}_{\\nabla_{z}}^{-1} (\\nabla_{z}\\boldsymbol{\sigma})}

        where the parameter gradient :math:`\\nabla_{z}\sigma` at the ith layer is computed via

        .. math::
            :label: dpdz1

            \\nabla_{z}^{i}\sigma = \\frac{\sigma_{i+1} - \\sigma_{i}}{h_{i} - h_{min}}

        where :math:`\sigma_{i+1}` and :math:`\sigma_{i}` are the log-parameters on either side of an interface, :math:`h_{i}` is the log-thickness of the ith layer, and :math:`h_{min}` is the minimum log thickness defined by

        .. math::
            :label: minThickness1

            h_{min} = \\frac{z_{max} - z_{min}}{2 k_{max}}

        where :math:`k_{max}` is a maximum number of layers, set to be far greater than the expected final solution.

        Parameters
        ----------
        hmin : float64
            The minimum thickness of any layer.

        Returns
        -------
        out : numpy.float64
            The probability given the prior on the gradient of the parameters with depth.

        """
        # if not self.gradient.hasPrior:
        #     return 0.0 if log else 1.0

        if self.nCells.item() == 1:
            tmp = self.insert_edge(nplog(self.mesh.min_edge) + (0.5 * (self.mesh.max_edge - self.mesh.min_edge)))
            return tmp.gradient.probability(log=log)
        else:
            return self.gradient.probability(log=log)

    def insert_edge(self, edge, value=None):

        out, values = self.mesh.insert_edge(edge, values=self.values)

        if value is not None:
            values[out.action[1]] = value
        out = type(self)(mesh=out, values=values)
        out._gradient = self.gradient.resize(out.nCells.item() - 1)

        return out

    def interpolate_centres_to_nodes(self, kind='cubic', **kwargs):
        return self.mesh.interpolate_centres_to_nodes(self.values, kind=kind, **kwargs)

    def local_precision(self, observation=None):
        """Generate a localized inverse Hessian matrix using a dataPoint and the current realization of the Model1D.

        Parameters
        ----------
        datapoint : geobipy.DataPoint, optional
            The data point to use when computing the local estimate of the variance.
            If None, only the prior derivative is used.

        Returns
        -------
        out : array_like
            Inverse Hessian matrix

        """
        assert self.values.hasPrior or self.gradient.hasPrior, Exception("Model must have either a parameter prior or gradient prior, use self.set_priors()")

        hessian = self.prior_derivative(order=2)

        if not observation is None:
            vals = observation.prior_derivative(order=2)
            hessian += vals

        return hessian

    def local_variance(self, observation=None):
        """Generate a localized inverse Hessian matrix using a dataPoint and the current realization of the Model1D.

        Parameters
        ----------
        datapoint : geobipy.DataPoint, optional
            The data point to use when computing the local estimate of the variance.
            If None, only the prior derivative is used.

        Returns
        -------
        out : array_like
            Inverse Hessian matrix

        """
        return inv(self.local_precision(observation))

    def pad(self, shape):
        """Copies the properties of a model including all priors or proposals, but pads memory to the given size

        Parameters
        ----------
        size : int, tuple
            Create memory upto this size.

        Returns
        -------
        out : geobipy.Model1D
            Padded model

        """
        mesh = self.mesh.pad(shape)
        values = self.values.pad(shape)
        out = type(self)(mesh=mesh, values=values)

        # for key in self.__values:

        #     out.setattr(key, getattr(self, key).pad(shape))


        # out._magnetic_permeability = self.magnetic_permeability.pad(size)
        # out._magnetic_susceptibility = self.magnetic_susceptibility.pad(size)
        # out._dpar = self.dpar.pad(size-1)
        # if (not self.Hitmap is None):
        #     out.Hitmap = self.Hitmap
        return out

    def map_to_pdf(self, pdf, log=False, axis=0):
        assert isinstance(self.values, baseDistribution), TypeError("values must have type geobipy.basDistribution")
        return self.mesh.map_to_pdf(distribution=self.values, pdf=pdf, log=log, axis=axis)

    def perturb(self, observation=None, low_variance=-inf, high_variance=inf):
        """Perturb a model's structure and parameter values.

        Uses a stochastic newtown approach if a datapoint is provided.
        Otherwise, uses the existing proposal distribution attached to
        self.par to generate new values.

        Parameters
        ----------
        observation : geobipy.DataPoint, optional
            The datapoint to use to perturb using a stochastic Newton approach.

        Returns
        -------
        remappedModel : geobipy.Model
            The current model remapped onto the perturbed dimension.
        perturbedModel : geobipy.Model
            The model with perturbed structure and parameter values.

        """
        return self.stochastic_newton_perturbation(observation, low_variance=low_variance, high_variance=high_variance)

    def stochastic_newton_perturbation(self, observation=None, low_variance=-inf, high_variance=inf):

        # Perturb the structure of the model
        remapped_model = self.perturb_structure()

        if observation is not None:
            if remapped_model.mesh.action[0] != 'none':
                observation.sensitivity(remapped_model)

        # Update the local Hessian around the current model.
        # inv(J'Wd'WdJ + Wm'Wm)
        inverse_hessian = remapped_model.compute_local_inverse_hessian(observation)

        if inverse_hessian.size > 1:
            ih_max = inverse_hessian.max()
            if ih_max < low_variance:
                inverse_hessian *= (low_variance / ih_max)
            elif ih_max > high_variance:
                inverse_hessian *= (high_variance / ih_max)

        # Proposing new parameter values
        # This is Wm'Wm(sigma - sigma_ref)
        # Need to have the gradient be a part of this too.
        gradient = remapped_model.prior_derivative(order=1)

        if not observation is None:
            # The gradient is now J'Wd'(dPredicted - dObserved) + Wm'Wm(sigma - sigma_ref)
            tmp = observation.prior_derivative(order=1)
            gradient += tmp

        # Compute the Model perturbation
        # This is the equivalent to the full newton gradient of the deterministic objective function.
        # delta sigma = 0.5 * inv(J'Wd'WdJ + Wm'Wm)(J'Wd'(dPredicted - dObserved) + Wm'Wm(sigma - sigma_ref))
        # This could be replaced with a CG solver for bigger problems like deterministic algorithms.
        dSigma = -0.5 * dot(inverse_hessian, gradient)
        mean = exp(nplog(remapped_model.values) + dSigma)

        perturbed_model = deepcopy(remapped_model)

        # Assign a proposal distribution for the parameter using the mean and variance.
        perturbed_model.values.proposal = Distribution('MvLogNormal',
                                                       mean=mean,
                                                       variance=inverse_hessian,
                                                       linearSpace=True,
                                                       prng=perturbed_model.values.proposal.prng)

        # Generate new conductivities
        perturbed_model.values.perturb()#imposePrior=True)

        return remapped_model, perturbed_model

    def prior_derivative(self, order):
        # Wm'Wm(m - mref) = (Wz'Wz + Ws'Ww'WwWs)(m - mref)
        operator = self.values.priorDerivative(order=2)
        if self.gradient.hasPrior:
            Wz = self.mesh.gradient_operator
            operator += dot(Wz.T, Wz)

        operator *= self.mesh.cell_weights

        return dot(operator, self.values.prior.deviation(self.values)) if order == 1 else operator

    def perturb_structure(self):

        remapped_mesh, remapped_values = self.mesh.perturb(values=self.values)
        remapped_model = type(self)(remapped_mesh, values=remapped_values)

        remapped_model._gradient = deepcopy(self._gradient)

        if self.value_bounds is not None:
            remapped_model.value_bounds = self.value_bounds
        if remapped_model.values.hasPrior:
            remapped_model.values.prior.ndim = remapped_model.nCells.item()
        if remapped_model.gradient.hasPrior:
            remapped_model.gradient.prior.ndim = maximum(1, remapped_model.nCells-1)

        return remapped_model

    def pcolor(self, **kwargs):
        return self.mesh.pcolor(values=self.values, **kwargs)

    def plot(self, **kwargs):
        ### DO NOT CHANGE THIS TO PCOLOR
        return self.mesh.plot(values=self.values, **kwargs)

    def plotGrid(self, **kwargs):
        return self.mesh.plotGrid(**kwargs)

    def _init_posterior_plots(self, gs, sharex=None, sharey=None):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        return self.mesh._init_posterior_plots(gs, values=self.values, sharex=sharex, sharey=sharey)

    def reset_posteriors(self):
        self.mesh.reset_posteriors()
        self.values.reset_posteriors()

    def plot_posteriors(self, axes=None, values_kwargs={}, axis=0, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', gcf())

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes)

        self.mesh.plot_posteriors(axes, axis=axis, values=self.values, values_kwargs=values_kwargs, **kwargs)

        self.values.posterior.mean(axis=axis).plot(xscale=values_kwargs.get('xscale', 'linear'),
                                                   flipY=False,
                                                   reciprocateX=values_kwargs.get('reciprocateX', None),
                                                   labels=False,
                                                   linewidth=1,
                                                   color='#5046C8',
                                                   ax=axes[-1])
        self.values.posterior.mode(axis=axis).plot(xscale=values_kwargs.get('xscale', 'linear'),
                                                   flipY=False,
                                                   reciprocateX=values_kwargs.get('reciprocateX', None),
                                                   labels=False,
                                                   linewidth=1,
                                                   color='#6046C8',
                                                   ax=axes[-1])

    def pcolor(self, **kwargs):
        """Plot like an image

        Other Parameters
        ----------------
        alpha : scalar or array_like, optional
            If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
            If array_like, each pixel is given an individual alpha value.
        log : 'e' or float, optional
            Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
            Values in c that are <= 0 are masked.
        equalize : bool, optional
            Equalize the histogram of the colourmap so that all colours have an equal amount.
        nbins : int, optional
            Number of bins to use for histogram equalization.
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'.
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        grid : bool, optional
            Plot the grid
        noColorbar : bool, optional
            Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.

        """
        return self.mesh.pcolor(values=self.values, **kwargs)

    def probability(self, solve_value, solve_gradient):
        """Evaluate the prior probability for the 1D Model.

        The following equation describes the components of the prior that correspond to the Model1D,

        .. math::
            p(k | I)p(\\boldsymbol{z}| k, I)p(\\boldsymbol{\sigma} | k, \\boldsymbol{z}, I),

        where :math:`k, I, \\boldsymbol{z}` and :math:`\\boldsymbol{\sigma}` are the number of layers, prior information, interface depth, and physical property, respectively.

        The multiplication here can be turned into a summation by taking the log of the components.

        **Prior on the number of layers**

        Uninformative prior using a uniform distribution.

        .. math::
            :label: layers

            p(k | I) =
            \\begin{cases}
            \\frac{1}{k_{max} - 1} & \\quad 1 \leq k \leq k_{max} \\newline
            0 & \\quad otherwise
            \\end{cases}.

        **Prior on the layer interface depths**

        We use order statistics for the prior on layer depth interfaces.

        .. math::
            :label: depth

            p(\\boldsymbol{z} | k, I) = \\frac{(k -1)!}{\prod_{i=0}^{k-1} \Delta z_{i}},

        where the numerator describes the number of ways that :math:`(k - 1)` interfaces can be ordered and
        :math:`\Delta z_{i} = (z_{max} - z_{min}) - 2 i h_{min}` describes the depth interval that is available to place a layer when there are already i interfaces in the model

        **Prior on the physical parameter**

        If we use a multivariate normal distribution to describe the joint prior pdf for log-parameter values in all layers we use,

        .. math::
            :label: parameter

            p(\\boldsymbol{\sigma} | k, I) = \\left[(2\pi)^{k} |\\boldsymbol{C}_{\\boldsymbol{\sigma}0}|\\right]^{-\\frac{1}{2}} e ^{-\\frac{1}{2}(\\boldsymbol{\sigma} - \\boldsymbol{\sigma}_{0})^{T} \\boldsymbol{C}_{\\boldsymbol{\sigma} 0}^{-1} (\\boldsymbol{\sigma} - \\boldsymbol{\sigma}_{0})}

        **Prior on the gradient of the physical parameter with depth**

        If instead we wish to apply a multivariate normal distribution to the gradient of the parameter with depth we get

        .. math::
            :label: gradient

            p(\\boldsymbol{\sigma} | k, I) = \\left[(2\pi)^{k-1} |\\boldsymbol{C}_{\\nabla_{z}}|\\right]^{-\\frac{1}{2}} e ^{-\\frac{1}{2}(\\nabla_{z} \\boldsymbol{\sigma})^{T} \\boldsymbol{C}_{\\nabla_{z}}^{-1} (\\nabla_{z}\\boldsymbol{\sigma})}

        where the parameter gradient :math:`\\nabla_{z}\sigma` at the ith layer is computed via

        .. math::
            :label: dpdz

            \\nabla_{z}^{i}\sigma = \\frac{\sigma_{i+1} - \\sigma_{i}}{h_{i} - h_{min}}

        where :math:`\sigma_{i+1}` and :math:`\sigma_{i}` are the log-parameters on either side of an interface, :math:`h_{i}` is the log-thickness of the ith layer, and :math:`h_{min}` is the minimum log thickness defined by

        .. math::
            :label: minThickness

            h_{min} = \\frac{z_{max} - z_{min}}{2 k_{max}}

        where :math:`k_{max}` is a maximum number of layers, set to be far greater than the expected final solution.

        Parameters
        ----------
        sPar : bool
            Evaluate the prior on the parameters :eq:`parameter` in the final probability
        sGradient : bool
            Evaluate the prior on the parameter gradient :eq:`gradient` in the final probability
        components : bool, optional
            Return all components used in the final probability as well as the final probability

        Returns
        -------
        probability : numpy.float64
            The probability
        components : array_like, optional
            Return the components of the probability, i.e. the individually evaluated priors as a second return argument if comonents=True on input.

        """

        # Check that the parameters are within the limits if they are bound
        if not self.value_bounds is None:
            pInBounds = self.value_bounds.probability(x=self.values, log=True)
            if any(isinf(pInBounds)):
                return -inf

        # Get the structural prior probability
        probability = self.mesh.probability

        # Evaluate the prior based on the assigned hitmap
        # if (not self.Hitmap is None):
        #     self.evaluateHitmapPrior(self.Hitmap)

        # Probability of parameter
        if solve_value: # if self.values.hasPrior
            probability += self.values.probability(log=True)

        # Probability of model gradient
        if solve_gradient: # if self.gradient.hasPrior
            probability += self.gradient_probability(log=True)

        return probability

    def proposal_probabilities(self, remappedModel, observation=None):
        """Return the forward and reverse proposal probabilities for the model

        Returns the denominator and numerator for the model's components of the proposal ratio.

        .. math::
            :label: proposal numerator

            q(k, \\boldsymbol{z} | \\boldsymbol{m}^{'})

        and

        .. math::
            :label: proposal denominator

            q(k^{'}, \\boldsymbol{z}^{'} | \\boldsymbol{m})

        Each component is dependent on the event that was chosen during perturbation.

        Parameters
        ----------
        remappedModel : geobipy.Model1D
            The current model, remapped onto the dimension of self.
        observation : geobipy.DataPoint
            The perturbed datapoint that was used to generate self.

        Returns
        -------
        forward : float
            The forward proposal probability
        reverse : float
            The reverse proposal probability

        """
        # Evaluate the Reversible Jump Step.
        # For the reversible jump, we need to compute the gradient from the perturbed parameter values
        # that were generated using pertured data, using the unperturbed data.
        # We therefore scale the sensitivity matrix by the proposed errors in the data, and our gradient uses
        # the data residual using the perturbed parameter values.

        # Compute the gradient according to the perturbed parameters and data residual
        # This is Wm'Wm(sigma - sigma_ref)
        # gradient = self.prior_derivative(order=1)
        gradient = self.prior_derivative(order=1)

        # todo:
        # replace the par.priorDerivative with appropriate gradient

        if not observation is None:
            # The prior derivative is now J'Wd'(dPredicted - dObserved) + Wm'Wm(sigma - sigma_ref)
            gradient += observation.prior_derivative(order=1)

        # Compute the stochastic newton offset.
        SN_step_from_perturbed = -0.5 * dot(self.values.proposal.variance, gradient)

        prng = self.values.proposal.prng
        # # Create a multivariate normal distribution centered on the shifted parameter values, and with variance computed from the forward step.
        # # We don't recompute the variance using the perturbed parameters, because we need to check that we could in fact step back from
        # # our perturbed parameters to the unperturbed parameters. This is the crux of the reversible jump.
        tmp = Distribution('MvLogNormal', exp(nplog(self.values) + SN_step_from_perturbed), self.values.proposal.variance, linearSpace=True, prng=prng)
        # tmp = Distribution('MvLogNormal', self.values, self.values.proposal.variance, linearSpace=True, prng=prng)
        # Probability of jumping from our perturbed parameter values to the unperturbed values.
        proposal = tmp.probability(x=remappedModel.values, log=True)

        # This is the forward proposal. Evaluate the new proposed values given a mean of the old values
        # and variance using perturbed data
        tmp = Distribution('MvLogNormal', remappedModel.values, self.values.proposal.variance, linearSpace=True, prng=prng)
        proposal1 = tmp.probability(x=self.values, log=True)

        action = self.mesh.action[0]
        if action == 'insert':
            k = self.nCells - 1

            forward = Distribution('Uniform', 0.0, self.mesh.remainingSpace(k), prng=prng)
            reverse = Distribution('Uniform', 0.0, k, prng=prng)

            proposal += reverse.probability(1, log=True)
            proposal1 += forward.probability(0.0, log=True)

        if action == 'delete':
            k = self.nCells

            forward = Distribution('Uniform', 0.0, self.mesh.remainingSpace(k), prng=prng)
            reverse = Distribution('Uniform', 0.0, k, prng=prng)

            proposal += forward.probability(0.0, log=True)
            proposal1 += reverse.probability(1, log=True)

        return proposal, proposal1

    def pyvista_mesh(self, **kwargs):
        mesh = self.mesh.pyvista_mesh(**kwargs)
        mesh.cell_data[self.values.label] = self.mesh._reorder_for_pyvista(self.values)
        return mesh

    def set_posteriors(self, values_posterior=None, **kwargs):

        from ..mesh.RectilinearMesh2D import RectilinearMesh2D
        from ..statistics.Histogram import Histogram

        self.mesh.set_posteriors(**kwargs)

        if values_posterior is None:
            relative_to = self.values.prior.mean[0]
            bins = StatArray.StatArray(self.values.prior.bins(nBins=250, nStd=4.0, axis=0), self.values.name, self.values.units)

            x_log = None
            if 'log' in type(self.values.prior).__name__.lower():
                x_log = 10
            mesh = RectilinearMesh2D(x_edges=bins, y_edges=self.mesh.edges.posterior.mesh.edges, x_relative_to=relative_to, x_log=x_log)

            # Set the posterior hitmap for conductivity vs depth
            self.values.posterior = Histogram(mesh=mesh)

    def set_priors(self, values_prior=None, gradient_prior=None, **kwargs):
        """Setup the priors of a 1D model.

        Parameters
        ----------
        halfSpaceValue : float
            Value of the parameter for the halfspace.
        min_edge : float64
            Minimum depth possible for the model
        max_edge : float64
            Maximum depth possible for the model
        max_cells : int
            Maximum number of layers allowable in the model
        parameterPrior : bool
            Sets a prior on the parameter values
        gradientPrior : bool
            Sets a prior on the gradient of the parameter values
        parameterLimits : array_like, optional
            Length 2 array with the bounds on the parameter values to impose.
        min_width : float64, optional
            Minimum thickness of any layer. If min_width = None, min_width is computed from min_edge, max_edge, and max_cells (recommended).
        factor : float, optional
            Tuning parameter used in the std of the parameter prior.
        prng : numpy.random.RandomState(), optional
            Random number generator, if none is given, will use numpy's global generator.

        See Also
        --------
        geobipy.Model1D.perturb : For a description of the perturbation cycle.

        """

        self.mesh.set_priors(**kwargs)

        self.value_bounds = None
        if kwargs.get('parameter_limits') is not None:
            self.value_bounds = Distribution('Uniform',
                                             *kwargs['parameter_limits'],
                                             log=True,
                                             prng=kwargs.get('prng'))

        if values_prior is None:
            if kwargs.get('solve_value', False):
                assert 'value_mean' in kwargs, ValueError("No value_prior given, must specify keywords 'value_mean'")
                # Assign the initial prior to the parameters
                variance = nplog(1.0 + kwargs.get('factor', 10.0))**2.0 #self.mesh.cell_weights /
                values_prior = Distribution('MvLogNormal', mean=kwargs['value_mean'],
                                                variance=variance,
                                                ndim=self.mesh.nCells,
                                                linearSpace=True,
                                                prng=kwargs.get('prng'))

        if gradient_prior is None:
            if kwargs.get('solve_gradient', False):
                gradient_prior = Distribution('MvNormal', mean=0.0,
                                            variance=kwargs.get('gradient_standard_deviation', 1.5)**2.0,
                                            ndim=maximum(1, self.mesh.nCells-1),
                                            prng=kwargs.get('prng'))

        self.set_values_prior(values_prior)
        self.set_gradient_prior(gradient_prior)

    def set_values_prior(self, prior):
        if prior is not None:
            self.values.prior = prior

    def set_gradient_prior(self, prior):
        if prior is not None:
            self.gradient.prior = prior

    def set_proposals(self, proposal=None, **kwargs):
        """Setup the proposals of a 1D model.

        Parameters
        ----------
        halfSpaceValue : float
            Value of the parameter for the halfspace.
        probabilities : array_like
            Probability of birth, death, perturb, and no change for the model
            e.g. pWheel = [0.5, 0.25, 0.15, 0.1]
        parameterProposal : geobipy.Distribution
            The proposal distribution for the parameter.
        prng : numpy.random.RandomState(), optional
            Random number generator, if none is given, will use numpy's global generator.

        See Also
        --------
        geobipy.Model1D.perturb : For a description of the perturbation cycle.

        """

        self.mesh.set_proposals(**kwargs)

        if proposal is None:
            local_variance = self.local_variance(kwargs.get('observation', None))

            # Instantiate the proposal for the parameters.
            proposal = Distribution('MvLogNormal', mean=self.values, variance=local_variance, linearSpace=True, prng=kwargs.get('prng', None))

        self.values.proposal = proposal

    def take_along_axis(self, i, axis):
        s = [s_[:] for j in range(self.ndim)]
        s[axis] = i
        return self[tuple(s)]

    def update_posteriors(self, ratio=0.5):
        """Update any attached posterior distributions.

        Parameters
        ----------
        minimumRatio : float
            Only update the depth posterior if the layer parameter ratio
            is greater than this number.

        """
        self.mesh.update_posteriors(values=self.values, ratio=ratio)
        # Update the hitmap posterior
        self.update_parameter_posterior(axis=0)

    def update_parameter_posterior(self, axis=0):
        """ Imposes a model's parameters with depth onto a 2D Hitmap.

        The cells that the parameter-depth profile passes through are accumulated by 1.

        Parameters
        ----------
        Hitmap : geobipy.Hitmap
            The hitmap to add to

        """
        histogram = self.values.posterior

        # Interpolate the cell centres and parameter values to the 'axis' of the histogram
        values = self.mesh.piecewise_constant_interpolate(self.values, histogram, axis=1-axis)

        # values is now histogram.axis(axis).nCells in length
        # interpolate those values to the opposite axis
        i0 = histogram.cellIndex(values, axis=axis, clip=True)

        ax = histogram.axis(1-axis)
        # Get the bounding indices depending on whether the mesh has open limits.
        if self.mesh.open_right:
            mx = ax.nCells
        else:
            mx = ax.cellIndex(self.mesh.edges[-1], clip=True)
        i1 = arange(mx)

        histogram.counts[i0, i1] += 1

    def resample(self, dx, dy):
        mesh, values = self.mesh.resample(dx, dy, self.values, kind='cubic')
        return type(self)(mesh, values)

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None, upcast=True):
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)

        self.mesh.createHdf(grp, 'mesh', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=upcast)
        self.values.createHdf(grp, 'values', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        return grp

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        self.mesh.writeHdf(parent, name+'/mesh', withPosterior=withPosterior, index=index)
        self.values.writeHdf(parent, name+'/values',  withPosterior=withPosterior, index=index)

    @classmethod
    def fromHdf(cls, grp, index=None, skip_posterior=False):
        """ Reads in the object from a HDF file """
        mesh = hdfRead.read_item(grp['mesh'], index=index, skip_posterior=skip_posterior)
        out = cls(mesh=mesh)
        out._values = out.mesh.fromHdf_cell_values(grp, 'values', index=index, skip_posterior=skip_posterior)
        return out