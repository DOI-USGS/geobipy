""" @Model1D_Class
Module describing a 1 Dimensional layered Model
"""
# from ...base import Error as Err
from ...classes.core.StatArray import StatArray
from ..mesh.RectilinearMesh1D import RectilinearMesh1D
from ..mesh.RectilinearMesh2D import RectilinearMesh2D
from ..statistics.Histogram import Histogram
from ..statistics.Distribution import Distribution
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from ...base import plotting as cP
from ...base import utilities as cF
from copy import deepcopy


class Model1D(RectilinearMesh1D):
    """Class extension to geobipy.Model

    Class creates a representation of the 1D layered earth.
    The class contains parameters that describe the physical property of each layer, as well as the layer
    thicknesses and interface depths. The model maintains a half space as the lowest layer with infinite extent.
    This class allows a probability wheel to be assigned such that the 1D layered earth can be peturbed. i.e.
    randomly create, delete, or perturb the layers in the model.  This is a key step in a Markov Chain for Bayesian inference.

    Model1D(nCells, top, parameters, edges, widths)

    Parameters
    ----------
    nCells : int
        Number of layers in the 1D layered earth. A single layer represents a half space.
    top : numpy.float64, optional
        Depth to the top of the model
    parameters : geobipy.StatArray, optional
        Describes the variable within each layer of the model.
    depth : geobipy.StatArray, optional
        Depths to the lower interface of each layer above the halfspace. Do not provide if thickness is given.
    thickness : geobipy.StatArray, optional
        Thickness of each layer above the halfspace. Do not provide if depths are given.

    Returns
    -------
    out : Model1D
        1D layered earth model.

    Raises
    ------
    ValueError
        If nCells is <= 0
    ValueError
        If size(parameters) != nCells
    ValueError
        If size(depth) != nCells - 1
    ValueError
        If size(thickness) != nCells - 1
    TypeError
        If both depth and thickness are provided

    """

    def __init__(self, centres=None, edges=None, widths=None, relativeTo=None, parameters=None, **kwargs):
        """Instantiate a new Model1D """

        # relativeTo = 0.0 if relativeTo is None else relativeTo

        

        super().__init__(centres=centres, edges=edges, widths=widths, relativeTo=relativeTo)

        # if (all((x is None for x in [centres, relativeTo, parameters, edges, widths]))):
        #     return
        assert (not(not widths is None and not edges is None)), TypeError('Cannot instantiate with both edges and widths values')

        self.par = parameters

        # StatArray of the change in physical parameters
        self._dpar = StatArray(self.nCells.item() - 1, 'Derivative', r"$\frac{"+self.par.units+"}{m}$")

        # StatArray of magnetic properties.
        self._magnetic_susceptibility = StatArray(self.nCells.item(), "Magnetic Susceptibility", r"$\kappa$")
        self._magnetic_permeability = StatArray(self.nCells.item(), "Magnetic Permeability", "$\frac{H}{m}$")

        self.parameterBounds = None
        self._halfSpaceParameter = None
        self.Hitmap = None
        self._inverse_hessian = None

    @property
    def dpar(self):
        return self._dpar

    def doi(self, percentage=67.0, log=None):
        if self.par.hasPosterior:
            return self.par.posterior.getOpacityLevel(percentage, log=log)

    @property
    def inverse_hessian(self):
        return self._inverse_hessian

    @property
    def magnetic_permeability(self):
        return self._magnetic_permeability

    @property
    def magnetic_susceptibility(self):
        return self._magnetic_susceptibility

    @property
    def par(self):
        return self._par

    @par.setter
    def par(self, values):

        if values is None:
            values = self.nCells.item()
        else:
            assert values.size == self.nCells, ValueError('Size of parameters {} must equal {}'.format(values.size, self.nCells.item()))

        self._par = StatArray(values)

    def __deepcopy__(self, memo={}):
        """Create a deepcopy

        Returns
        -------
        out : geobipy.Model1D
            Deepcopy of Model1D

        """
        out = super().__deepcopy__(memo)
        out._par = deepcopy(self.par)
        out._dpar = deepcopy(self.dpar)
        out._magnetic_permeability = deepcopy(self.magnetic_permeability)
        out._magnetic_susceptibility = deepcopy(self.magnetic_susceptibility)
        out.Hitmap = self.Hitmap
        out._inverse_hessian = deepcopy(self._inverse_hessian)
        out.parameterBounds = self.parameterBounds
        out._halfSpaceParameter = self._halfSpaceParameter
        return out

    def pad(self, size):
        """Copies the properties of a model including all priors or proposals, but pads memory to the given size

        Parameters
        ----------
        size : int
            Create memory upto this size.

        Returns
        -------
        out : geobipy.Model1D
            Padded model

        """
        out = super().pad(size)
        out._par = self.par.pad(size)
        out._magnetic_permeability = self.magnetic_permeability.pad(size)
        out._magnetic_susceptibility = self.magnetic_susceptibility.pad(size)
        out._dpar = self.dpar.pad(size-1)
        if (not self.Hitmap is None):
            out.Hitmap = self.Hitmap
        return out

    def localParameterVariance(self, datapoint=None):
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
        assert self.par.hasPrior or self.dpar.hasPrior, Exception("Model must have either a parameter prior or gradient prior, use self.set_priors()")

        if not datapoint is None:
            # Propose new layer conductivities
            inverse_hessian = np.linalg.inv(datapoint.prior_derivative(model=self, order=2) + self.par.priorDerivative(order=2))
        else:
            inverse_hessian = np.linalg.inv(self.par.priorDerivative(order=2))

        return inverse_hessian


    def compute_local_inverse_hessian(self, dataPoint=None):
        """Generate a localized Hessian matrix using
        a dataPoint and the current realization of the Model1D.


        Parameters
        ----------
        dataPoint : geobipy.DataPoint, optional
            The data point to use when computing the local estimate of the variance.

        Returns
        -------
        out : array_like
            Hessian matrix

        """
        # Compute a new parameter variance matrix if the structure of the model changed.

        if (self.action[0] in ['insert', 'delete'] or self.inverse_hessian is None):
            # Propose new layer conductivities
            self._inverse_hessian = self.localParameterVariance(dataPoint)

    def insert_edge(self, value, par=None, update_priors=False):
        """Insert a new edge into a model at a given location

        Parameters
        ----------
        value : numpy.float64
            Location at which to insert a new edge
        par : numpy.float64, optional
            Value of the parameter for the new layer
            If None, The value of the split layer is duplicated.
        update_priors : bool, optional
            If priors are attached, update their dimension to match

        Returns
        -------
        out : geobipy.Model1D
            Model with inserted layer.

        """
        # Insert edge into the mesh
        out = super().insert_edge(value)

        i = out.action[1] - 1

        if (par is None):
            if (i >= self.par.size):
                out._par = out.par.insert(i, self.par[i])
            else:
                out._par = out.par.insert(i, self.par[i])
        else:
            out._par = out.par.insert(i, par)

        # Reset ChiE and ChiM
        out._magnetic_permeability = StatArray(out.nCells.item(), "Electric Susceptibility", r"$\kappa$")
        out._magnetic_susceptibility = StatArray(out.nCells.item(), "Magnetic Susceptibility", r"$\frac{H}{m}$")
        # Resize the parameter gradient
        out._dpar = out.dpar.resize(out.par.size - 1)

        # Update the dimensions of any priors.
        if update_priors:
            out.par.prior.ndim = out.nCells
            out.dpar.prior.ndim = np.maximum(1, out.nCells-1)

        return out

    def delete_edge(self, i, update_priors=False):
        """Remove an edge from the model

        Parameters
        ----------
        i : int
            The edge to remove.

        Returns
        -------
        out : geobipy.Model1D
            Model with layer removed.
        update_priors : bool, optional
            If priors are attached, update their dimension to match

        """
        out = super().delete_edge(i)

        # Take the average of the deleted layer and the one below it
        out._par = out.par.delete(i)
        out._par[i-1] = 0.5 * (self.par[i-1] + self.par[i])
        # Reset ChiE and ChiM
        out._magnetic_permeability = out.magnetic_permeability.delete(i)
        out._magnetic_susceptibility = out.magnetic_susceptibility.delete(i)
        # Resize the parameter gradient
        out._dpar = out.dpar.resize(out.par.size - 1)

        if update_priors:
            out.par.prior.ndim = out.nCells
            out.dpar.prior.ndim = np.maximum(1, out.nCells-1)

        return out

    def prior_derivative(self, order):
        return self.par.priorDerivative(order=order)

    def priorProbability(self, pPrior, gPrior, log=True, verbose=False):
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
        if not self.parameterBounds is None:
            pInBounds = self.parameterBounds.probability(x=self.par, log=log)
            if np.any(np.isinf(pInBounds)):
                return -np.inf

        # Get the structural prior probability
        p = super().priorProbability(log=log)

        # Evaluate the prior based on the assigned hitmap
        if (not self.Hitmap is None):
            self.evaluateHitmapPrior(self.Hitmap)

        # Probability of parameter
        p_prior = 0.0 if log else 1.0
        if pPrior:
            p_prior = self.par.probability(log=log)

        # Probability of model gradient
        g_prior = 0.0 if log else 1.0
        if gPrior:
            g_prior = self.gradientProbability(log=log)

        return (p + p_prior + g_prior) if log else (p * p_prior * g_prior)

    def proposalProbabilities(self, remappedModel, datapoint=None):
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
        datapoint : geobipy.DataPoint
            The perturbed datapoint that was used to generate self.

        Returns
        -------
        forward : float
            The forward proposal probability
        reverse : float
            The reverse proposal probability

        """

        # Evaluate the Reversible Jump Step.
        # For the reversible jump, we need to compute the gradient from the perturbed parameter values.
        # We therefore scale the sensitivity matrix by the proposed errors in the data, and our gradient uses
        # the data residual using the perturbed parameter values.

        # Compute the gradient according to the perturbed parameters and data residual
        # This is Wm'Wm(sigma - sigma_ref)
        gradient = self.prior_derivative(order=1)

        # todo:
        # replace the par.priorDerivative with appropriate gradient

        if not datapoint is None:
            # The prior derivative is now J'Wd'(dPredicted - dObserved) + Wm'Wm(sigma - sigma_ref)
            gradient += datapoint.prior_derivative(order=1)

        # Compute the stochastic newton offset.
        # The negative sign because we want to move downhill
        SN_step_from_perturbed = 0.5 * np.dot(self.inverse_hessian, gradient)

        prng = self.par.proposal.prng

        # Create a multivariate normal distribution centered on the shifted parameter values, and with variance computed from the forward step.
        # We don't recompute the variance using the perturbed parameters, because we need to check that we could in fact step back from
        # our perturbed parameters to the unperturbed parameters. This is the crux of the reversible jump.
        tmp = Distribution('MvLogNormal', np.exp(np.log(self.par) - SN_step_from_perturbed), self.inverse_hessian, linearSpace=True, prng=prng)
        # Probability of jumping from our perturbed parameter values to the unperturbed values.
        proposal = tmp.probability(x=remappedModel.par, log=True)

        tmp = Distribution('MvLogNormal', remappedModel.par, self.inverse_hessian, linearSpace=True, prng=prng)
        proposal1 = tmp.probability(x=self.par, log=True)

        if self.action[0] == 'insert':
            k = self.nCells - 1

            forward = Distribution('Uniform', 0.0, self.remainingSpace(k))
            reverse = Distribution('Uniform', 0.0, k)

            proposal += reverse.probability(1, log=True)
            proposal1 += forward.probability(0.0, log=True)

        if self.action[0] == 'delete':
            k = self.nCells

            forward = Distribution('Uniform', 0.0, self.remainingSpace(k))
            reverse = Distribution('Uniform', 0.0, k)

            proposal += forward.probability(0.0, log=True)
            proposal1 += reverse.probability(1, log=True)

        return proposal, proposal1


    def perturb(self, datapoint=None):
        """Perturb a model's structure and parameter values.

        Uses a stochastic newtown approach if a datapoint is provided.
        Otherwise, uses the existing proposal distribution attached to
        self.par to generate new values.

        Parameters
        ----------
        dataPoint : geobipy.DataPoint, optional
            The datapoint to use to perturb using a stochastic Newton approach.

        Returns
        -------
        remappedModel : geobipy.Model1D
            The current model remapped onto the perturbed dimension.
        perturbedModel : geobipy.Model1D
            The model with perturbed structure and parameter values.

        """
        return self.stochasticNewtonPerturbation(datapoint)

    def squeeze(self, widths, parameters):

        i = np.hstack([np.where(np.diff(parameters) != 0)[0], -1])

        edges = np.hstack([0.0, np.cumsum(widths)[i]])

        self.__init__(edges=edges, parameters=parameters[i])

        return self

    def stochasticNewtonPerturbation(self, datapoint=None):

        # Perturb the structure of the model
        remappedModel = super().perturb()

        # Update the local Hessian around the current model.
        remappedModel.compute_local_inverse_hessian(datapoint)

        # Proposing new parameter values
        # This is Wm'Wm(sigma - sigma_ref)
        gradient = remappedModel.prior_derivative(order=1)

        if not datapoint is None:
            # The gradient is now J'Wd'(dPredicted-dObserved) + Wm'Wm(sigma - sigma_ref)
            gradient += datapoint.prior_derivative(order=1)

        # Compute the Model perturbation
        # This is the equivalent to the full newton gradient of the deterministic objective function.
        # delta sigma = 0.5 * inv(J'Wd'WdJ + Wm'Wm)(J'Wd'(dPredicted - dObserved) + Wm'Wm(sigma - sigma_ref))
        # This could be replaced with a CG solver for bigger problems.
        dSigma = 0.5 * np.dot(remappedModel.inverse_hessian, gradient)

        mean = np.log(remappedModel.par) - dSigma

        perturbedModel = deepcopy(remappedModel)

        # Assign a proposal distribution for the parameter using the mean and variance.
        perturbedModel.par.proposal = Distribution('MvLogNormal', mean=np.exp(mean),
                                                      variance=remappedModel.inverse_hessian,
                                                      linearSpace=True,
                                                      prng=perturbedModel.par.proposal.prng)

        # Generate new conductivities
        perturbedModel.par.perturb()

        return remappedModel, perturbedModel

    def set_posteriors(self):

        super().set_posteriors()

        if self.par.hasPrior:
            p = self.par.prior.bins(nBins=250, nStd=4.0, axis=0)
        else:
            tmp = 4.0 * np.log(11.0)
            p = np.linspace(self.halfSpaceParameter - tmp,
                            self.halfSpaceParameter + tmp, 251)

        pGrd = StatArray(p, self.par.name, self.par.units)

        mesh = RectilinearMesh2D(x_edges=pGrd, y_edges=self.edges.posterior.mesh.edges)
        # Set the posterior hitmap for conductivity vs depth
        self.par.posterior = Histogram(mesh=mesh)

    def set_priors(self, halfSpaceValue, min_edge, max_edge, max_cells, parameterPrior, gradientPrior, parameterLimits=None, min_width=None, factor=10.0, dzVariance=1.5, prng=None):
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

        super().set_priors(min_edge=min_edge, max_edge=max_edge, max_cells=max_cells, min_width=min_width, prng=prng)

        if not parameterLimits is None:
            assert np.size(parameterLimits) == 2, ValueError(
                "parameterLimits must have size 2.")
            self.parameterBounds = Distribution('Uniform', parameterLimits[0], parameterLimits[1], log=True)
        else:
            self.parameterBounds = None

        self._halfSpaceParameter = halfSpaceValue

        # if parameterPrior:
        # Assign the initial prior to the parameters
        self.par.prior = Distribution('MvLogNormal', mean=self._halfSpaceParameter,
                                         variance=np.log(1.0 + factor)**2.0,
                                         ndim=self.nCells,
                                         linearSpace=True,
                                         prng=prng)

        # if gradientPrior:
        # Assign the prior on the parameter gradient
        self.dpar.prior = Distribution('MvNormal', mean=0.0,
                                       variance=dzVariance,
                                       ndim=np.maximum(1, self.nCells-1),
                                       prng=prng)

    def setProposals(self, probabilities, parameterProposal, prng=None):
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

        super().set_proposals(probabilities, prng)
        self.par.proposal = parameterProposal

    def gradientProbability(self, log=True):
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
        assert (self.dpar.hasPrior), TypeError(
            'No prior defined on parameter gradient. Use Model1D.dpar.addPrior() to set the prior.')

        if np.int32(self.nCells) == 1:
            tmp = self.insert_edge(np.log(self.min_edge) + (0.5 * (self.max_edge - self.min_edge)))
            tmp.dpar[:] = (np.diff(np.log(tmp.par))) / (np.log(tmp.widths[:-1]) - np.log(self.min_width))
            probability = tmp.dpar.probability(log=log)

        else:
            self.dpar[:] = (np.diff(np.log(self.par))) / (np.log(self.widths[:-1]) - np.log(self.min_width))
            probability = self.dpar.probability(log=log)
        return probability

    @property
    def summary(self):
        """ Write a summary of the 1D model """

        msg = ("1D Model: \n"
               "nCells\n{}\n"
               "edges\n{}\n"
               "thickness\n{}\n"
               "parameters\n{}\n"
               ).format(self.nCells.summary,
                        self.edges.summary,
                        self.widths.summary,
                        self.par.summary)
        return msg

    def unperturb(self):
        """After a model has had its structure perturbed, remap the model back its previous state. Used for the reversible jump McMC step.

        """
        if self.action[0] == 'none':
            return deepcopy(self)

        if self.action[0] == 'perturb':
            out = deepcopy(self)
            out.edges[self.action[1]] -= self.action[2]
            return out

        if self.action[0] == 'insert':
            return self.delete_edge(self.action[1])

        if self.action[0] == 'delete':
            return self.insert_edge(self.action[2])

    def pcolor(self, *args, **kwargs):
        """Create a pseudocolour plot of the 1D Model.

        Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

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

        Returns
        -------
        ax
            matplotlib.Axes

        See Also
        --------
        matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

        """

        return super().pcolor(values=self.par, **kwargs)
    #     d = self.depth
    #     if self.hasHalfspace:
    #         if (self.max_edge is None):
    #             if (self.nCells > 1):
    #                 d[-1] = 1.1 * d[-2]
    #             else:
    #                 d[0] = 1.0
    #         else:
    #             d[-1] = 1.1 * self.max_edge

    #     ax = self.par.pcolor(*args, y = d + self.top, **kwargs)

    #     if self.hasHalfspace:
    #         h = 0.99*d[-1]
    #         if (self.nCells == 1):
    #             h = 0.99*self.max_edge
    #         plt.text(0, h, s=r'$\downarrow \infty$', fontsize=12)

    #     return ax

    # def piecewise_constant_interpolate(self, other, bound=False, axis=0):
    #     return super().piecewise_constant_interpolate(self.par, other, bound, axis)

    def plot(self, **kwargs):
        """Plots a 1D model parameters as a line against depth

        Parameters
        ----------
        reciprocateX : bool, optional
            Take the reciprocal of the x axis
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        noLabels : bool, optional
            Do not plot the labels

        """

        super().plot(values=self.par, **kwargs)

        # if self.hasHalfspace:
        #     h = 0.99*z[-1]
        #     if (self.nCells == 1):
        #         h = 0.99*self.max_edge
        #     plt.text(0, h, s=r'$\downarrow \infty$', fontsize=12)

    @property
    def n_posteriors(self):
        return super().n_posteriors + np.sum(self.par.hasPosterior)

    def init_posterior_plots(self, gs):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        splt = gs.subgridspec(2, 2, height_ratios=[1, 4])
        splt2 = splt[1, :].subgridspec(1, 2, wspace=0.2)
        ax = []
        ax.append(plt.subplot(splt[0, :]))
        ax.append(plt.subplot(splt2[0, 1]))
        ax.append(plt.subplot(splt2[0, 0]))

        for a in ax:
            cP.pretty(a)

        return ax

    def plot_posteriors(self, axes=None, parameter_kwargs={}, **kwargs):

        assert len(axes) == 3, ValueError(("Must have length 3 list of axes for the posteriors. \n"
                                          "self.init_posterior_plots() can generate them"))

        if kwargs.get('edges_kwargs', {}).get('flipY', False) and parameter_kwargs.get('flipY', False):
            parameter_kwargs['flipY'] = False

        super().plot_posteriors(axes[:2], **kwargs)
        axes[2].sharey(axes[1])
    
        self.par.plotPosteriors(ax=axes[2], **parameter_kwargs)

        best = kwargs.pop('best', None)
        if best is not None:
            best.plot(xscale=parameter_kwargs.get('xscale', None), 
                      flipY=False, 
                      reciprocateX=parameter_kwargs.get('reciprocateX', None), 
                      labels=False, 
                      linewidth=1, 
                      color=cP.wellSeparated[3])

            doi = self.par.posterior.opacity_level(log=parameter_kwargs.get('logX', None), axis=1)
            plt.axhline(doi, color = '#5046C8', linestyle = 'dashed', linewidth = 1, alpha = 0.6)
        return axes

    def evaluateHitmapPrior(self, Hitmap):
        """ Evaluates the model parameters against a hitmap.

        Given a Hitmap describing the probability of the parameters with depth, evaluate the current model using the grid cells it passes through.

        Parameters
        ----------
        Hitmap : geobipy.Hitmap
            A 2D hitmap with y axis depth, and x axis parameter to evaluate the probability of the model given the hitmap

        Returns
        -------
        out : numpy.float64
            The probability of the model given the hitmap.

        """

        iM = self.par2mesh(Hitmap)
        tmp = np.sum(Hitmap.arr[:, iM])
        return tmp / np.sum(Hitmap.arr)

    def asHistogram2D(self, variance, Hist):
        """ Creates a Hitmap from the model given the variance of each layer.

        For each depth, creates a normal distribution with a mean equal to the interpolated parameter
        at that depth and variance specified with variance.

        Parameters
        ----------
        variance : array_like
            The variance of each layer
        Hitmap : geobipy.Hitmap
            Hitmap to convert the model to.
            Must be instantiated before calling so that the model can be interpolated correctly

        """
        assert (variance.size == self.nCells), ValueError(
            'size of variance must equal number of cells')
        # Interpolate the parameter to the depths of the grid
        par = self.interp2depth(self.par, Hist)
        # Interpolate the variance to the depths of the grid
        var = self.interp2depth(variance, Hist)

        # plt.figure()
        for i in range(Hist.y.size):
            # dist = Distribution('Normal', par[i], var[i])
            # Hist.arr[i, :] = dist.probability(tmp)

            # for j in range(45, 55):
            #     print(dist.probability(tmp[j]))

            # #dist = Distribution('NormalLog', np.log(par[i]), np.log((var[i])))
            # #Hist.arr[i, :] = np.exp(dist.probability(np.log(Hist.x)))
            # plt.subplot(211)
            # plt.semilogx(tmp, Hist.arr[i, :])

            dist = Distribution('Normal', np.log(par[i]), var[i])

            Hist.arr[i, :] = np.exp(dist.probability(np.log(Hist.x), log=True))
            # print(np.max(Hist.arr[i, :]))
            # plt.subplot(212)
            # plt.plot(np.log10(tmp), Hist.arr[i, :])
            # for j in range(Hist.x.size):
            #    Hist.arr[i, j] = np.exp(dist.probability([np.log(Hist.x[j])]))
            #    Hist.sum += Hist.arr[i, j]

        Hist.sum = np.sum(Hist.arr)

    def update_parameter_posterior(self, axis=0):
        """ Imposes a model's parameters with depth onto a 2D Hitmap.

        The cells that the parameter-depth profile passes through are accumulated by 1.

        Parameters
        ----------
        Hitmap : geobipy.Hitmap
            The hitmap to add to

        """
        histogram = self.par.posterior
        # Interpolate the cell centres and parameter values to the 'axis' of the histogram
        values = self.piecewise_constant_interpolate(self.par, histogram, axis=1-axis)

        # values is now histogram.axis(axis).nCells in length
        # interpolate those values to the opposite axis
        i0 = histogram.cellIndex(values, axis=axis, clip=True)

        ax = histogram.axis(1-axis)
        # Get the bounding indices depending on whether the mesh has open limits.
        if self.open_right:
            mx = ax.nCells
        else:
            mx = ax.cellIndex(self.edges[-1], clip=True)
        i1 = np.arange(mx)

        histogram.counts[i0, i1] += 1

    # def setReferenceHitmap(self, Hitmap):
    #     """ Assigns a Hitmap as the model's prior """
    #     assert isinstance(Hitmap, Hitmap2D), "Hitmap must be a Hitmap2D class"
    #     self.Hitmap = Hitmap.deepcopy()

    def isInsideConfidence(self, histogram2d, percent=95.0, log=None):
        """ Check that the model is insde the specified confidence region of a 2D hitmap

        Parameters
        ----------
        histogram2d : geobipy.Histogram
            The hitmap to check against.
        percent : np.float, optional
            The confidence interval percentage.
        log : bool, optional
            Whether to take the parameters to a log base (log).

        Returns
        -------
        out : bool
            All parameter values are within the confidence intervals.

        """
        assert isinstance(histogram2d, Histogram2D), TypeError('histogram2d must be of type Histogram2D')

        sMed, sLow, sHigh = histogram2d.getConfidenceIntervals(percent=percent, log=log)

        par = self.piecewise_constant_interpolate(self.par, histogram2d, axis=1)

        return np.all(par > sLow) and np.all(par < sHigh)

    def update_posteriors(self, minimumRatio=0.5):
        """Update any attached posterior distributions.

        Parameters
        ----------
        minimumRatio : float
            Only update the depth posterior if the layer parameter ratio
            is greater than this number.

        """
        super().update_posteriors(values=self.par, ratio=minimumRatio)
        # Update the hitmap posterior
        self.update_parameter_posterior(axis=0)

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        """Create the Metadata for a Model1D in a HDF file

        Creates a new group in a HDF file under h5obj.
        A nested heirarchy will be created.
        This method can be used in an MPI parallel environment, if so however,
        a) the hdf file must have been opened with the mpio driver, and
        b) createHdf must be called collectively,
        i.e., called by every core in the MPI communicator that was used to open the file.
        In order to create large amounts of empty space before writing to it in parallel,
        the add_axis parameter will extend the memory in the first dimension.

        Parameters
        ----------
        h5obj : h5py._hl.files.File or h5py._hl.group.Group
            A HDF file or group object to create the contents in.
        name : str
            The name of the group to create.
        add_axis : int, optional
            Inserts a first dimension into the first dimension of each attribute of the Model1D of length add_axis.
            This can be used to extend the available memory of the Model1D so that multiple MPI ranks can write to
            their respective parts in the extended memory.
        fillvalue : number, optional
            Initializes the memory in file with the fill value

        Notes
        -----
        This method can be used in serial and MPI. As an example in MPI.
        Given 10 MPI ranks, each with a 10 length array, it is faster to create a 10x10 empty array,
        and have each rank write its row. Rather than creating 10 separate length 10 arrays because
        the overhead when creating the file metadata can become very cumbersome if done too many times.

        Example
        -------
        >>> from geobipy import Model1D
        >>> from mpi4py import MPI
        >>> import h5py

        >>> world = MPI.COMM_WORLD
        >>> # Create a holder for models in memory with more layers than you will expect.
        >>> tmp = Model1D(nCells=20)

        >>> # This is a collective open of data in the file
        >>> f = h5py.File(fName,'w', driver='mpio',comm=world)
        >>> # Collective creation of space(padded by number of mpi ranks)
        >>> tmp.createHdf(f, 'models', add_axis=world.size)

        >>> world.barrier()

        >>> # In a non collective region, we can write to different sections of x in the file
        >>> # Fake a non collective region
        >>> def noncollectivewrite(model, file, world):
        >>>     # Each rank carries out this code, but it's not collective.
        >>>     model.writeHdf(file, 'models',  index=world.rank)
        >>> noncollectivewrite(mod, f, world)

        >>> world.barrier()
        >>> f.close()

        """
        grp = super().createHdf(parent, name, withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.par.createHdf(grp, 'par', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

    def writeHdf(self, h5obj, name, withPosterior=True, index=None):
        """Create the Metadata for a Model1D in a HDF file

        Creates a new group in a HDF file under h5obj.
        A nested heirarchy will be created.
        This method can be used in an MPI parallel environment, if so however, a) the hdf file must have been opened with the mpio driver,
        and b) createHdf must be called collectively, i.e., called by every core in the MPI communicator that was used to open the file.
        In order to create large amounts of empty space before writing to it in parallel, the nRepeats parameter will extend the memory
        in the first dimension.

        Parameters
        ----------
        h5obj : h5py._hl.files.File or h5py._hl.group.Group
            A HDF file or group object to create the contents in.
        name : str
            The name of the group to create.
        fillvalue : number, optional
            Initializes the memory in file with the fill value

        Notes
        -----
        This method can be used in serial and MPI. As an example in MPI.
        Given 10 MPI ranks, each with a 10 length array, it is faster to create a 10x10 empty array, and have each rank write its row.
        Rather than creating 10 separate length 10 arrays because the overhead when creating the file metadata can become very
        cumbersome if done too many times.

        Example
        -------
        >>> from geobipy import Model1D
        >>> from mpi4py import MPI
        >>> import h5py

        >>> world = MPI.COMM_WORLD
        >>> # Create a holder for models in memory with more layers than you will expect.
        >>> tmp = Model1D(nCells=20)

        >>> # This is a collective open of data in the file
        >>> f = h5py.File(fName,'w', driver='mpio',comm=world)
        >>> # Collective creation of space(padded by number of mpi ranks)
        >>> tmp.createHdf(f, 'models', nRepeats=world.size)

        >>> world.barrier()

        >>> # In a non collective region, we can write to different sections of x in the file
        >>> # Fake a non collective region
        >>> def noncollectivewrite(model, file, world):
        >>>     # Each rank carries out this code, but it's not collective.
        >>>     model.writeHdf(file, 'models',  index=world.rank)
        >>> noncollectivewrite(mod, f, world)

        >>> world.barrier()
        >>> f.close()

        """

        super().writeHdf(h5obj, name, withPosterior, index)

        grp = h5obj.get(name)
        self.par.writeHdf(grp, 'par',  withPosterior=withPosterior, index=index)

    @classmethod
    def fromHdf(cls, grp, index=None):
        """Read the class from a HDF group

        Given the HDF group object, read the contents into an Model1D class.

        Parameters
        ----------
        h5obj : h5py._hl.group.Group
            A HDF group object to write the contents to.
        index : slice, optional
            If the group was created using the nRepeats option, index specifies the index'th entry from which to read the data.

        """
        self = super(Model1D, cls).fromHdf(grp, index)

        self._par = StatArray.fromHdf(grp['par'], index=np.s_[index, :self.nCells.item()])

        return self
