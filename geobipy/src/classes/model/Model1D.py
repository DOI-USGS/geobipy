""" @Model1D_Class
Module describing a 1 Dimensional layered Model
"""
#from ...base import Error as Err
from ...classes.core import StatArray
from .Model import Model
from ..mesh import RectilinearMesh2D
from ..statistics.Histogram1D import Histogram1D
from ..statistics.Hitmap2D import Hitmap2D
from ...base.logging import myLogger
from ..statistics.Distribution import Distribution
import numpy as np
import matplotlib.pyplot as plt
from ...base import customPlots as cP
from ...base import customFunctions as cF
from copy import deepcopy

class Model1D(Model):
    """Class extension to geobipy.Model

    Class creates a representation of the 1D layered earth.
    The class contains parameters that describe the physical property of each layer, as well as the layer
    thicknesses and interface depths. The model maintains a half space as the lowest layer with infinite extent.
    This class allows a probability wheel to be assigned such that the 1D layered earth can be peturbed. i.e.
    randomly create, delete, or perturb the layers in the model.  This is a key step in a Markov Chain for Bayesian inference.

    Model1D(nCells, top, parameters, depth, thickness)

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

    def __init__(self, nCells=None, top=None, parameters = None, depth = None, thickness = None, hasHalfspace=True):
        """Instantiate a new Model1D """

        self.hasHalfspace = hasHalfspace

        if (all((x is None for x in [nCells, top, parameters, depth, thickness]))): return
        assert (not(not thickness is None and not depth is None)), TypeError('Cannot instantiate with both depth and thickness values')

        # Number of Cells in the model
        self._nCells = StatArray.StatArray(1, '# of Cells', dtype=np.int32)
        if not nCells is None:
            assert (nCells >= 1), ValueError('nCells must >= 1')
            self._nCells[0] = nCells

        # Depth to the top of the model
        if top is None:
            top = 0.0
        self._top = StatArray.StatArray(top, "Depth to top", "m")

        if hasHalfspace:
            self._init_withHalfspace(nCells, top, parameters, depth, thickness)
        else:
            self._init_withoutHalfspace(nCells, top, parameters, depth, thickness)

        # StatArray of the change in physical parameters
        self._dpar = StatArray.StatArray(np.int(self.nCells[0]) - 1, 'Derivative', r"$\frac{"+self.par.getUnits()+"}{m}$")

        # StatArray of magnetic properties.
        self._magnetic_susceptibility = StatArray.StatArray(np.int(self.nCells[0]), "Magnetic Susceptibility", r"$\kappa$")
        self._magnetic_permeability = StatArray.StatArray(np.int(self.nCells[0]), "Magnetic Permeability", "$\frac{H}{m}$")

        # Instantiate extra parameters for Markov chain perturbations.
        # Minimum cell thickness
        self.minThickness = None
        # Minimum depth
        self.minDepth = None
        # Maximum depth
        self.maxDepth = None
        # Maximum number of layers
        self.maxLayers = None
        # Categorical distribution for choosing perturbation events
        self.eventProposal = None
        # Keep track of actions made to the Model.
        self.action = ['none', 0, 0.0]

        self.parameterBounds = None
        self._halfSpaceParameter = None
        self.Hitmap = None
        self._inverseHessian = None


    @property
    def nCells(self):
        return self._nCells

    @property
    def top(self):
        return self._top

    @property
    def dpar(self):
        return self._dpar

    def doi(self, percentage=67.0, log=None):
        if self.par.hasPosterior:
            return self.par.posterior.getOpacityLevel(percentage, log=log)

    @property
    def inverseHessian(self):
        return self._inverseHessian


    def _init_withHalfspace(self, nCells=None, top=None, parameters = None, depth = None, thickness = None):


        if (not depth is None and nCells is None):
            self._nCells[0] = depth.size + 1
        if (not thickness is None and nCells is None):
            self._nCells[0] = thickness.size + 1

        self._depth = StatArray.StatArray(np.int(self.nCells[0]), 'Depth', 'm')
        self._thk = StatArray.StatArray(np.int(self.nCells[0]), 'Thickness', 'm')
        self._par = StatArray.StatArray(np.int(self.nCells[0]))

        if (not depth is None):
            if (self.nCells > 1):
                assert depth.size == self.nCells-1, ValueError('Size of depth must equal {}'.format(np.int(self.nCells[0])-1))
                assert np.all(np.diff(depth) > 0.0), ValueError('Depths must monotonically increase')
            self._depth[:-1] = depth
            self._depth[-1] = np.inf
            self.thicknessFromDepth()


        if (not thickness is None):
            if nCells is None:
                self.nCells[0] = thickness.size + 1
            if (self.nCells > 1):
                assert thickness.size == self.nCells-1, ValueError('Size of thickness must equal {}'.format(np.int(self.nCells[0])-1))
                assert np.all(thickness > 0.0), ValueError('Thicknesses must be positive')
            self._thk[:-1] = thickness
            self._thk[-1] = np.inf
            self.depthFromThickness()

        # StatArray of the physical parameters
        if (not parameters is None):
            assert parameters.size == self.nCells, ValueError('Size of parameters must equal {}'.format(np.int(self.nCells[0])))
            self._par = StatArray.StatArray(parameters)


    def _init_withoutHalfspace(self, nCells = None, top = None, parameters = None, depth = None, thickness = None):


        if (not depth is None and nCells is None):
            self.nCells[0] = depth.size
        if (not thickness is None and nCells is None):
            self.nCells[0] = thickness.size

        self._depth = StatArray.StatArray(np.int(self.nCells[0]), 'Depth', 'm')
        self._thk = StatArray.StatArray(np.int(self.nCells[0]), 'Thickness', 'm')
        self._par = StatArray.StatArray(np.int(self.nCells[0]))

        if (not depth is None):
            if (self.nCells > 1):
                assert depth.size == self.nCells, ValueError('Size of depth must equal {}'.format(np.int(self.nCells[0])))
                assert np.all(np.diff(depth) > 0.0), ValueError('Depths must monotonically increase')
            self._depth[:] = depth
            self.thicknessFromDepth()

        if (not thickness is None):
            if nCells is None:
                self.nCells[0] = thickness.size
            if (self.nCells > 1):
                assert thickness.size == self.nCells, ValueError('Size of thickness must equal {}'.format(np.int(self.nCells[0])))
                assert np.all(thickness > 0.0), ValueError('Thicknesses must be positive')
            self._thk[:] = thickness
            self.depthFromThickness()

        # StatArray of the physical parameters
        if (not parameters is None):
            assert parameters.size == self.nCells, ValueError('Size of parameters must equal {}'.format(np.int(self.nCells[0])))
            self._par = StatArray.StatArray(parameters)

    @property
    def depth(self):
        return self._depth

    @property
    def magnetic_permeability(self):
        return self._magnetic_permeability

    @property
    def magnetic_susceptibility(self):
        return self._magnetic_susceptibility

    @property
    def thk(self):
        return self._thk

    @property
    def par(self):
        return self._par


    def deepcopy(self):
        """Create a deepcopy

        Returns
        -------
        out : geobipy.Model1D
            Deepcopy of Model1D

        """
        other = Model1D(nCells=None)
        other._nCells = self.nCells.deepcopy()
        other._top = self.top
        other._depth = self.depth.deepcopy()
        other._thk = self.thk.deepcopy()
        other.minThickness = self.minThickness
        other.minDepth = self.minDepth
        other.maxDepth = self.maxDepth
        other.maxLayers = self.maxLayers
        other._par = self.par.deepcopy()
        other._dpar = self.dpar.deepcopy()
        other._magnetic_permeability = self.magnetic_permeability.deepcopy() #StatArray(other.nCells[0], "Electric Susceptibility", r"$\kappa$")
        other._magnetic_susceptibility = self.magnetic_susceptibility.deepcopy() #StatArray(other.nCells[0], "Magnetic Susceptibility", "$\frac{H}{m}$")
        other.eventProposal = self.eventProposal
        other.action = self.action.copy()
        other.Hitmap = self.Hitmap
        other.hasHalfspace = self.hasHalfspace
        other._inverseHessian = deepcopy(self._inverseHessian)
        other.parameterBounds = self.parameterBounds
        other._halfSpaceParameter = self._halfSpaceParameter
        return other


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
        tmp = Model1D(size,self.top)
        tmp._nCells = self.nCells
        tmp._depth = self.depth.pad(size)
        tmp._thk = self.thk.pad(size)
        tmp._par = self.par.pad(size)
        tmp._magnetic_permeability = self.magnetic_permeability.pad(size)
        tmp._magnetic_susceptibility = self.magnetic_susceptibility.pad(size)
        tmp._dpar=self.dpar.pad(size-1)
        if (not self.minDepth is None): tmp.minDepth=self.minDepth
        if (not self.maxDepth is None): tmp.maxDepth=self.maxDepth
        if (not self.maxLayers is None): tmp.maxLayers=self.maxLayers
        if (not self.minThickness is None): tmp.minThickness=self.minThickness
        if (not self.Hitmap is None): tmp.Hitmap=self.Hitmap
        return tmp


    def depthFromThickness(self):
        """Given the thicknesses of each layer, create the depths to each interface. The last depth is inf for the halfspace."""
        self._depth[:] = np.cumsum(self.thk)

        if self.hasHalfspace:
            self._depth[-1] = np.infty


    def thicknessFromDepth(self):
        """Given the depths to each interface, compute the layer thicknesses. The last thickness is nan for the halfspace."""
        self._thk = self.thk.resize(np.int(self.nCells[0]))
        self._thk[0] = self.depth[0]
        for i in range(1, np.int(self.nCells[0])):
            self._thk[i] = self.depth[i] - self.depth[i - 1]

        if self.hasHalfspace:
            self._thk[-1] = np.inf


    def localParameterVariance(self, dataPoint=None):
        """Generate a localized inverse Hessian matrix using a dataPoint and the current realization of the Model1D.

        Parameters
        ----------
        dataPoint : geobipy.DataPoint, optional
            The data point to use when computing the local estimate of the variance.
            If None, only the prior derivative is used.

        Returns
        -------
        out : array_like
            Inverse Hessian matrix

        """
        assert self.par.hasPrior or self.dpar.hasPrior, Exception("Model must have either a parameter prior or gradient prior, use self.setPriors()")

        if not dataPoint is None:
            # Compute the sensitivity of the data to the perturbed model
            dataPoint.sensitivity(self)
            Wd = dataPoint.weightingMatrix(power=1.0)
            WdJ = np.dot(Wd, dataPoint.J)
            WdJTWdJ = np.dot(WdJ.T, WdJ)

            # Propose new layer conductivities
            self._inverseHessian = np.linalg.inv(WdJTWdJ + self.par.prior.derivative(x=None, order=2))
        else:
            self._inverseHessian = self.par.prior.derivative(x=None, order=2)

        return self._inverseHessian


    def updateLocalParameterVariance(self, dataPoint=None):
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

        if (self.action[0] in ['birth', 'death']):
            # Compute the sensitivity of the data to the perturbed model
            # dataPoint.updateSensitivity(self)
            # Wd = dataPoint.weightingMatrix(power=1.0)
            # WdJ = np.dot(Wd, dataPoint.J)
            # WdJTWdJ = np.dot(WdJ.T, WdJ)

            # # Propose new layer conductivities
            # self._inverseHessian = np.linalg.inv(WdJTWdJ + self.par.prior.derivative(x=None, order=2))
            self.localParameterVariance(dataPoint)

        else:  # There was no change in the model

            if self.inverseHessian is None:
                self.localParameterVariance(dataPoint)

        return self.inverseHessian


    def insertLayer(self, z, par=None):
        """Insert a new layer into a model at a given depth

        Parameters
        ----------
        z : numpy.float64
            Depth at which to insert a new interface
        par : numpy.float64, optional
            Value of the parameter for the new layer
            If None, The value of the split layer is duplicated.

        Returns
        -------
        out : geobipy.Model1D
            Model with inserted layer.

        """
        # Deepcopy the 1D Model
        tmp = self.depth[:-1]
        # Get the index to insert the new layer
        i = tmp.searchsorted(z)
        # Deepcopy the 1D Model
        other = self.deepcopy()
        # Increase the number of cells
        other._nCells += 1
        # Insert the new layer depth
        other._depth = self.depth.insert(i, z)
        if (par is None):
            if (i >= self.par.size):
                i -= 2
                other._par = other.par.insert(i, self.par[i])
                other._depth[-2] = other.depth[-1]
            else:
                other._par = other.par.insert(i, self.par[i])
        else:
            other._par = other.par.insert(i, par)
        # Get the new thicknesses
        other.thicknessFromDepth()
        # Reset ChiE and ChiM
        other._magnetic_permeability = StatArray.StatArray(np.int(other.nCells[0]), "Electric Susceptibility", r"$\kappa$")
        other._magnetic_susceptibility = StatArray.StatArray(np.int(other.nCells[0]), "Magnetic Susceptibility", r"$\frac{H}{m}$")
        # Resize the parameter gradient
        other._dpar = other.dpar.resize(other.par.size - 1)
        other.action = ['birth', np.int(i), z]
        return other


    def deleteLayer(self, i):
        """Remove a layer from the model

        Parameters
        ----------
        i : int
            The layer to remove.

        Returns
        -------
        out : geobipy.Model1D
            Model with layer removed.

        """

        if (self.nCells == 0):
            return self

        assert i < np.int(self.nCells[0]) - 1, ValueError("i must be less than the number of cells - 1{}".format(np.int(self.nCells[0])-1))

        # Deepcopy the 1D Model to ensure priors and proposals are passed
        other = self.deepcopy()
        # Decrease the number of cells
        other._nCells -= 1
        # Remove the interface depth
        other._depth = other.depth.delete(i)
        # Get the new thicknesses
        other.thicknessFromDepth()
        # Take the average of the deleted layer and the one below it
        other._par = other.par.delete(i)
        other._par[i] = 0.5 * (self.par[i] + self.par[i + 1])
        # Reset ChiE and ChiM
        other._magnetic_permeability = other.magnetic_permeability.delete(i)
        other._magnetic_susceptibility = other.magnetic_susceptibility.delete(i)
        # Resize the parameter gradient
        other._dpar = other.dpar.resize(other.par.size - 1)
        other.action = ['death', np.int(i), self.depth[i]]

        return other


    def perturbStructure(self):
        """Perturb a model

        Generates a new model by perturbing the current model based on four probabilities.
        The probabilities correspond to
        * Birth, the insertion of a new layer into the model
        * Death, the deletion of a layer from the model
        * Change, change one the existing interfaces
        * No change, do nothing and return the original

        The method self.makePerturbable must be used before calling self.perturb.

        The perturbation starts by generating a random number from a uniform distribution to determine which cycle to go through.
        If a layer is created, or an interface perturbed, any resulting layer thicknesses must be greater than the minimum thickness :math:`h_{min}`.
        If the new layer thickness test fails, the birth or perturbation tries again. If the cycle fails after 10 tries, the entire process begins again
        such that a death, or no change is possible thus preventing any neverending cycles.

        Returns
        -------
        out[0] : Model1D
            The perturbed model

        See Also
        --------
        geobipy.Model1D.makePerturbable : Must be used before calling self.perturb

        """
        assert (not self.eventProposal is None), ValueError('Please set the proposals of the model1D with model1D.addProposals()')
        prng = self.nCells.prior.prng
        # Pre-compute exponential values (Take them out of log space)
        hmin = self.minThickness
        zmin = self.minDepth
        zmax = self.maxDepth
        nTries = 10
        # This outer loop will allow the perturbation to change types. e.g. if the loop is stuck in a birthing
        # cycle, the number of tries will exceed 10, and we can try a different perturbation type.
        tryAgain = True  # Temporary to enter the loop
        while (tryAgain):
            tryAgain = False

            goodAction = False

            # Choose an action to perform, and make sure its legitimate
            #i.e. don't delete a single layer model, or add a layer to a model that is at the priors max on number of cells.
            while not goodAction:
                goodAction = True
                # Get a random probability from 0-1
                event = self.eventProposal.rng()

                if (np.int(self.nCells[0]) == 1 and (event == 1 or event == 2)):
                    goodAction = False
                elif (np.int(self.nCells[0]) == self.nCells.prior.max and event == 0):
                    goodAction = False

            # Return if no change
            if (event == 3):
                out = self.deepcopy()
                out.action = ['none', 0, 0.0]
                return out

            # Otherwise enter life-death-perturb cycle
            if (event == 0):  # Create a new layer
                newThicknessBiggerThanMinimum = False
                tries = 0
                while (not newThicknessBiggerThanMinimum):  # Continue while the new layer is smaller than the minimum
                    # Get the new depth
                    newDepth = np.exp(np.float64(prng.uniform(np.log(self.minDepth), np.log(self.maxDepth), 1)))
                    z = self.depth[:-1]
                    # Insert the new depth
                    i = z.searchsorted(newDepth)
                    z = z.insert(i, newDepth)
                    # Get the thicknesses
                    z = z.prepend(0.0)
                    h = np.min(np.diff(z[:]))
                    tries += 1
                    if (h > hmin):
                        newThicknessBiggerThanMinimum = True  # Exit if thickness is larger than minimum
                    if (tries == nTries):
                        newThicknessBiggerThanMinimum = True # just to exit.
                        tryAgain = True
                if (not tryAgain):
                    out = self.insertLayer(newDepth)
                    # Update the dimensions of any priors.
                    out.par.prior.ndim = out.nCells[0]
                    out.dpar.prior.ndim = np.maximum(1, out.nCells[0]-1)
                    return out

            if (event == 1):
                # Get the layer to remove
                iDeleted = np.int64(prng.uniform(0, self.nCells - 1, 1)[0])
                # Remove the layer and return
                out = self.deleteLayer(iDeleted)
                out.par.prior.ndim = out.nCells[0]
                out.dpar.prior.ndim = np.maximum(1, out.nCells[0]-1)
                return out

            if (event == 2):
                newThicknessBiggerThanMinimum = False
                k = np.int(self.nCells[0]) - 1
                tries = 0
                while (not newThicknessBiggerThanMinimum):  # Continue while the perturbed layer is suitable
                    z = self.depth[:-1]
                    # Get the layer to perturb
                    i = np.int64(prng.uniform(0, k, 1)[0])
                    # Get the perturbation amount
                    dz = np.sign(prng.randn()) * hmin * prng.uniform()
                    # Perturb the layer
                    z = z.prepend(0.0)
                    z[i + 1] += dz
                    # Get the minimum thickness
                    h = np.min(np.diff(z))
                    tries += 1
                    # Exit if the thickness is big enough, and we stayed within
                    # the depth bounds
                    if (h > hmin and z[1] > zmin and z[-1] < zmax):
                        newThicknessBiggerThanMinimum = True
                    if (tries == nTries):
                        newThicknessBiggerThanMinimum = True
                        tryAgain = True
                if (not tryAgain):
                    out = self.deepcopy()
                    out.depth[i] += dz  # Perturb the depth in the model
                    out.thicknessFromDepth()
                    out.action = ['perturb', np.int(i), dz]
                    return out

        assert False, Exception("Should not be here, file a bug report....")


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

        # Probability of the number of layers
        P_nCells = self.nCells.probability(log=log)

        # Probability of depth given nCells
        P_depthcells = self.depth.probability(x=self.nCells-1, log=log)

        # Evaluate the prior based on the assigned hitmap
        if (not self.Hitmap is None):
            self.evaluateHitmapPrior(self.Hitmap)

        # Probability of parameter
        # if self.par.hasPrior:
        P_parameter = 1.0 if log else 0.0
        if pPrior:
            P_parameter = self.par.probability(log=log)

        # Probability of model gradient
        # if self.dpar.hasPrior:
        P_gradient = 1.0 if log else 0.0
        if gPrior:
            P_gradient = self.gradientProbability(log=log)

        if log:
            probability = np.sum(np.r_[P_nCells, P_depthcells, P_parameter, P_gradient])
        else:
            probability = np.prod(np.r_[P_nCells, P_depthcells, P_parameter, P_gradient])

        if verbose:
            return probability, np.asarray([P_nCells, P_depthcells, P_parameter, P_gradient])
        return probability


    def remainingSpace(self, nLayers):
        return (self.maxDepth - self.minDepth) - nLayers * self.minThickness


    def proposalProbabilities(self, remappedModel, dataPoint=None):
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

        ### Evaluate the Reversible Jump Step.
        # For the reversible jump, we need to compute the gradient from the perturbed parameter values.
        # We therefore scale the sensitivity matrix by the proposed errors in the data, and our gradient uses
        # the data residual using the perturbed parameter values.

        # Compute the gradient according to the perturbed parameters and data residual
        gradient = self.par.priorDerivative(order=1)
        if not dataPoint is None:
            gradient += np.dot(dataPoint.J.T, dataPoint.predictedData.priorDerivative(order=1, i=dataPoint.active))

        # if (not burnedIn):
        #     SN_step_from_perturbed = 0.0
        # else:
        # Compute the stochastic newton offset.
        # The negative sign because we want to move downhill
        SN_step_from_perturbed = 0.5 * np.dot(self._inverseHessian, gradient)

        prng = self.par.proposal.prng

        # Create a multivariate normal distribution centered on the shifted parameter values, and with variance computed from the forward step.
        # We don't recompute the variance using the perturbed parameters, because we need to check that we could in fact step back from
        # our perturbed parameters to the unperturbed parameters. This is the crux of the reversible jump.
        tmp = Distribution('MvLogNormal', np.exp(np.log(self.par) - SN_step_from_perturbed), self.inverseHessian, linearSpace=True, prng=prng)
        # Probability of jumping from our perturbed parameter values to the unperturbed values.
        proposal = tmp.probability(x=remappedModel.par, log=True)  # CUR.prop

        tmp = Distribution('MvLogNormal', remappedModel.par, self.inverseHessian, linearSpace=True, prng=prng)
        proposal1 = tmp.probability(x=self.par, log=True)

        if self.action[0] == 'birth':
            k = self.nCells - 1

            forward = Distribution('Uniform', 0.0, self.remainingSpace(k))
            reverse = Distribution('Uniform', 0.0, k)

            proposal  += reverse.probability(1, log=True)
            proposal1 += forward.probability(0.0, log=True)

        if self.action[0] == 'death':
            k = self.nCells

            forward = Distribution('Uniform', 0.0, self.remainingSpace(k))
            reverse = Distribution('Uniform', 0.0, k)

            proposal  += forward.probability(0.0, log=True)
            proposal1 += reverse.probability(1, log=True)

        return proposal, proposal1


    # def reversibleJumpProbabilities(self):

    #     if self.action[0] in ['none', 'perturb']:
    #         return 1.0, 1.0

    #     if self.action[0] == 'birth':
    #         k = self.nCells - 1

    #         forward = Distribution('Uniform', 0.0, self.remainingSpace(k))
    #         reverse = Distribution('Uniform', 0.0, k)

    #         Pforward = forward.probability(self.maxDepth, log=True)
    #         Preverse = reverse.probability(k, log=True)

    #     if self.action[0] == 'death':
    #         k = self.nCells

    #         forward = Distribution('Uniform', 0.0, self.remainingSpace(k))
    #         reverse = Distribution('Uniform', 0.0, k)

    #         Pforward = reverse.probability(k, log=True)
    #         Preverse = forward.probability(self.maxDepth, log=True)

    #     return Pforward, Preverse


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


    def squeeze(self, thickness, parameters, hasHalfspace=False):

        i = np.hstack([np.where(np.diff(parameters) != 0)[0], -1])

        depth = np.cumsum(thickness)

        Model1D.__init__(self, depth=depth[i], parameters=parameters[i], hasHalfspace=hasHalfspace)

        return self



    def stochasticNewtonPerturbation(self, datapoint=None):

        # Perturb the structure of the model
        remappedModel = self.perturbStructure()

        # Update the local Hessian around the current model.
        inverseHessian = remappedModel.updateLocalParameterVariance(datapoint)

        ### Proposing new parameter values
        # Compute the gradient of the "deterministic" objective function using the unperturbed, remapped, parameter values.
        gradient = remappedModel.par.priorDerivative(order=1)
        if not datapoint is None:
            gradient += np.dot(datapoint.J.T, datapoint.predictedData.priorDerivative(order=1, i=datapoint.active))

        # scaling = parameterCovarianceScaling * ((2.0 * np.float64(Mod1.nCells[0])) - 1)**(-1.0 / 3.0)

        # Compute the Model perturbation
        # The negative sign because we want to move "downhill"
        # if (not burnedIn):
        #     SN_step_from_unperturbed = 0.0
        # else:
        SN_step_from_unperturbed = 0.5 * np.dot(inverseHessian, gradient)

        mean = np.log(remappedModel.par) - SN_step_from_unperturbed
        # variance = Mod1.inverseHessian

        perturbedModel = remappedModel.deepcopy()

        # Assign a proposal distribution for the parameter using the mean and variance.
        perturbedModel.par.setProposal('MvLogNormal', np.exp(mean), inverseHessian, linearSpace=True, prng=perturbedModel.par.proposal.prng)

        # Generate new conductivities
        perturbedModel.par.perturb()

        return remappedModel, perturbedModel


    def setPosteriors(self):

        assert not self.maxLayers is None, ValueError("No priors are set, user Model1D.setPriors() to do so.")

        # Initialize the posterior histogram for the number of layers
        self.nCells.setPosterior(Histogram1D(binCentres=StatArray.StatArray(np.arange(0.0, self.maxLayers + 1.0), name="# of Layers")))

        # Discretize the parameter values
        zGrd = StatArray.StatArray(np.arange(0.5 * self.minDepth, 1.1 * self.maxDepth, 0.5 * self.minThickness), self.depth.name, self.depth.units)
        # zGrd = StatArray.StatArray(np.logspace(np.log10(self.minDepth), np.log10(self.maxDepth), zGrd.size), self.depth.name, self.depth.units)

        if self.par.hasPrior:
            p = self.par.prior.bins(nBins = 250, nStd=4.0, axis=0)
        else:
            tmp = 4.0 * np.log(11.0)
            p = np.linspace(self._halfSpaceParameter - tmp, self._halfSpaceParameter + tmp, 251)

        pGrd = StatArray.StatArray(p, self.par.name, self.par.units)

        # Set the posterior hitmap for conductivity vs depth
        self.par.setPosterior(Hitmap2D(xBins = pGrd, yBinCentres = zGrd))

        # Initialize the interface Depth Histogram
        self.depth.setPosterior(Histogram1D(bins = zGrd))


    def setPriors(self, halfSpaceValue, minDepth, maxDepth, maxLayers, parameterPrior, gradientPrior, parameterLimits=None, minThickness=None, factor=10.0, dzVariance=1.5, prng=None):
        """Setup the priors of a 1D model.

        Parameters
        ----------
        halfSpaceValue : float
            Value of the parameter for the halfspace.
        minDepth : float64
            Minimum depth possible for the model
        maxDepth : float64
            Maximum depth possible for the model
        maxLayers : int
            Maximum number of layers allowable in the model
        parameterPrior : bool
            Sets a prior on the parameter values
        gradientPrior : bool
            Sets a prior on the gradient of the parameter values
        parameterLimits : array_like, optional
            Length 2 array with the bounds on the parameter values to impose.
        minThickness : float64, optional
            Minimum thickness of any layer. If minThickness = None, minThickness is computed from minDepth, maxDepth, and maxLayers (recommended).
        factor : float, optional
            Tuning parameter used in the std of the parameter prior.
        prng : numpy.random.RandomState(), optional
            Random number generator, if none is given, will use numpy's global generator.

        See Also
        --------
        geobipy.Model1D.perturb : For a description of the perturbation cycle.

        """
        assert minDepth > 0.0, ValueError("minDepth must be > 0.0")
        assert maxDepth > 0.0, ValueError("maxDepth must be > 0.0")
        assert maxLayers > 0.0, ValueError("maxLayers must be > 0.0")

        if (minThickness is None):
            # Assign a minimum possible thickness
            self.minThickness = (maxDepth - minDepth) / (2 * maxLayers)
        else:
            self.minThickness = minThickness

        self.minDepth = minDepth  # Assign the log of the min depth
        self.maxDepth = maxDepth  # Assign the log of the max depth
        self.maxLayers = np.int32(maxLayers)

        # Assign a uniform distribution to the number of layers
        self.nCells.setPrior('Uniform', 1, self.maxLayers, prng=prng)

        # Set priors on the depth interfaces, given a number of layers
        i = np.arange(self.maxLayers)
        dz = self.remainingSpace(i)

        self.depth.setPrior('Order', denominator=dz)  # priZ

        if not parameterLimits is None:
            assert np.size(parameterLimits) == 2, ValueError("parameterLimits must have size 2.")
            self.parameterBounds = Distribution('Uniform', parameterLimits[0], parameterLimits[1], log=True)
        else:
            self.parameterBounds = None

        self._halfSpaceParameter = halfSpaceValue

        # if parameterPrior:
        # Assign the initial prior to the parameters
        self.par.setPrior('MvLogNormal', self._halfSpaceParameter, np.log(1.0 + factor)**2.0, ndim=self.nCells, linearSpace=True, prng=prng)

        # if gradientPrior:
        # Assign the prior on the parameter gradient
        self.dpar.setPrior('MvNormal', 0.0, dzVariance, ndim=np.maximum(1, self.nCells-1), prng=prng)


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
            The proposal  distribution for the parameter.
        prng : numpy.random.RandomState(), optional
            Random number generator, if none is given, will use numpy's global generator.

        See Also
        --------
        geobipy.Model1D.perturb : For a description of the perturbation cycle.

        """
        assert np.size(probabilities) == 4, ValueError('pWheel must have size 4')
        # assert not self.maxLayers is None, Exception("Please set the priors on the model with setPriors()")

        self.eventProposal = Distribution('Categorical', np.asarray(probabilities), ['birth', 'death', 'perturb', 'noChange'], prng=prng)

        self.par.setProposal(parameterProposal)


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
        assert (self.dpar.hasPrior), TypeError('No prior defined on parameter gradient. Use Model1D.dpar.addPrior() to set the prior.')

        if np.int(self.nCells[0]) == 1:
            tmp = self.insertLayer(np.log(self.minDepth) + (0.5 * (self.maxDepth - self.minDepth)))
            tmp.dpar[:] = (np.diff(np.log(tmp.par))) / (np.log(tmp.thk[:-1]) - np.log(self.minThickness))
            probability = tmp.dpar.probability(log=log)

        else:
            self.dpar[:] = (np.diff(np.log(self.par))) / (np.log(self.thk[:-1]) - np.log(self.minThickness))
            probability = self.dpar.probability(log=log)
        return probability



    def summary(self, out=False):
        """ Write a summary of the 1D model """
        msg = "1D Model: \n"
        msg += self.nCells.summary(True)
        msg += 'Top of the model: ' + str(self.top) + '\n'
        msg += self.thk.summary(True)
        msg += self.par.summary(True)
        msg += self.depth.summary(True)
        if (out):
            return msg
        print(msg)


    def unperturb(self):
        """After a model has had its structure perturbed, remap the model back its previous state. Used for the reversible jump McMC step.

        """
        if self.action[0] == 'none':
            return self.deepcopy()

        if self.action[0] == 'perturb':
            other = self.deepcopy()
            other.depth[self.action[1]] -= self.action[2]
            return other

        if self.action[0] == 'birth':
            return self.deleteLayer(self.action[1])

        if self.action[0] == 'death':
            return self.insertLayer(self.action[2])


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
            Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.
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

        kwargs['flipY'] = kwargs.pop('flipY', True)

        d = self.depth.prepend(0.0)
        if self.hasHalfspace:
            if (self.maxDepth is None):
                if (self.nCells > 1):
                    d[-1] = 1.1 * d[-2]
                else:
                    d[0] = 1.0
            else:
                d[-1] = 1.1 * self.maxDepth

        ax = self.par.pcolor(*args, y = d + self.top, **kwargs)

        if self.hasHalfspace:
            h = 0.99*d[-1]
            if (self.nCells == 1):
                h = 0.99*self.maxDepth
            plt.text(0, h, s=r'$\downarrow \infty$', fontsize=12)

        return ax


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
        # Must create a new parameter, so that the last layer is plotted
        ax = plt.gca()
        cP.pretty(ax)

        reciprocateX = kwargs.pop("reciprocateX", False)
        flipY = kwargs.pop('flipY', True)
        kwargs['flipY'] = flipY
        kwargs['xscale'] = kwargs.pop('xscale', 'linear')

        # Repeat the last entry
        par = self.par.append(self.par[-1])
        if (reciprocateX):
            par = 1.0 / par

        z = self.depth.prepend(0.0)
        if self.hasHalfspace:
            if (self.maxDepth is None):
                z[-1] = 1.1 * z[-2]
            else:
                z[-1] = 1.1 * np.exp(self.maxDepth)

        cP.step(x=par, y=z, **kwargs)

        if self.hasHalfspace:
            h = 0.99*z[-1]
            if (self.nCells == 1):
                h = 0.99*self.maxDepth
            plt.text(0, h, s=r'$\downarrow \infty$', fontsize=12)


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
        assert (variance.size == self.nCells), ValueError('size of variance must equal number of cells')
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
            #for j in range(Hist.x.size):
            #    Hist.arr[i, j] = np.exp(dist.probability([np.log(Hist.x[j])]))
            #    Hist.sum += Hist.arr[i, j]

        Hist.sum = np.sum(Hist.arr)


    def addToHitMap(self, Hitmap):
        """ Imposes a model's parameters with depth onto a 2D Hitmap.

        The cells that the parameter-depth profile passes through are accumulated by 1.

        Parameters
        ----------
        Hitmap : geobipy.Hitmap
            The hitmap to add to

        """
        iM = self.getParMeshXIndex(Hitmap)
        if self.hasHalfspace:
            iz = np.arange(Hitmap.y.nCells)
        else:
            i = Hitmap.y.cellIndex(self.depth[-1], clip=True)
            iz = np.arange(i)

        Hitmap._counts[iz, iM] += 1


    # def setReferenceHitmap(self, Hitmap):
    #     """ Assigns a Hitmap as the model's prior """
    #     assert isinstance(Hitmap, Hitmap2D), "Hitmap must be a Hitmap2D class"
    #     self.Hitmap = Hitmap.deepcopy()


    def getParMeshXIndex(self, mesh):
        """ Interpolate the model parameters to a 2D rectilinear mesh.

        Uses piece wise constant interpolation of the parameter-depth profile to the y axis of the mesh.
        Then the indices into the mesh x axis for those interpolated values are returned.

        Parameters
        ----------
        mesh : geobipy.RectilinearMesh2D
            A mesh to interpolate to.

        Returns
        -------
        out : array
            The indices into mesh.x after interpolating the model parameters to the mesh.y axis.

        """
        mint = self.interpPar2Mesh(self.par, mesh)
        iM = mesh.x.cellCentres.searchsorted(mint)
        return np.minimum(iM, mesh.x.nCells - 1)


    def interpPar2Mesh(self, par, mesh, matchTop=False, bound=False):
        """ Interpolate the model parameters to a 2D rectilinear mesh.

        Uses piece wise constant interpolation of the parameter-depth profile to the y axis of the mesh.

        Parameters
        ----------
        par : geobipy.StatArray
            The values to interpolate to the mesh y axis. Must have length Model1D.nCells.
        mesh : geobipy.RectilinearMesh2D
            A mesh to interpolate to.
        matchTop : bool, optional
            Force the mesh y axis and top of the model to match.
        bound : bool, optional
            Interpolated values above the top of the model are nan.

        Returns
        -------
        out : array
            The interpolated model parameters at each y axis value of the mesh.

        """
        assert (np.size(par) == np.int(self.nCells[0])), 'par must have length nCells'
        assert (isinstance(mesh, RectilinearMesh2D.RectilinearMesh2D)), TypeError('mesh must be a RectilinearMesh2D')

        if self.hasHalfspace:
            bounds = [0.0, mesh.y.range]
        else:
            bounds = [0.0, np.minimum(self.depth[-1], mesh.y.range)]

        if self.hasHalfspace:
            depth = self.depth[:-1]
        else:
            depth = self.depth

        # Add in the top of the model
        if (matchTop):
            bounds += self.top
            depth += self.top

        if self.hasHalfspace:
            y = mesh.y.cellCentres
        else:
            i = mesh.y.cellIndex(depth[-1], clip=True)
            y = mesh.y.cellCentres[:i]

        if (np.int(self.nCells[0]) == 1):
            mint = np.interp(y, bounds, np.kron(par[:], [1, 1]))
        else:
            xp = np.kron(np.asarray(depth), [1, 1.001])
            if self.hasHalfspace:
                xp = np.insert(xp, [0, np.size(xp)], bounds)

            fp = np.kron(par[:], [1, 1])

            mint = np.interp(y, xp, fp)

        if bound:
            i = np.where((y < bounds[0]) & (y > bounds[-1]))
            mint[i] = np.nan

        return mint


    def isInsideConfidence(self, Hitmap, percent=95.0, log=None):
        """ Check that the model is insde the specified confidence region of a 2D hitmap

        Parameters
        ----------
        Hitmap : geobipy.Hitmap2D
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
        assert isinstance(Hitmap, Hitmap2D), TypeError('Hitmap must be of type Hitmap2D')

        sMed, sLow, sHigh = Hitmap.getConfidenceIntervals(percent=percent, log=log)
        print(sLow)

        par = self.interpPar2Mesh(self.par, Hitmap)

        return np.all(par > sLow) and np.all(par < sHigh)


    def updatePosteriors(self, minimumRatio=0.5):
        """Update any attached posterior distributions.

        Parameters
        ----------
        minimumRatio : float
            Only update the depth posterior if the layer parameter ratio
            is greater than this number.

        """

        # Update the number of layeres posterior
        self.nCells.updatePosterior()

        # Update the hitmap posterior
        self.addToHitMap(self.par.posterior)

        # Update the layer interface histogram
        if (self.nCells > 1):
            ratio = np.exp(np.diff(np.log(self.par)))
            m1 = ratio <= 1.0 - minimumRatio
            m2 = ratio >= 1.0 + minimumRatio
            keep = np.logical_not(np.ma.masked_invalid(ratio).mask) & np.ma.mask_or(m1,m2)
            tmp = self.depth[:-1]

            if (tmp[keep].size > 0):
                self.depth.posterior.update(tmp[keep])


    def hdfName(self):
        """Create a string that describes class instantiation

        Returns a string that should be used as an attr['repr'] in a HDF group.
        This allows reading of the attribute from the hdf file, evaluating it to return an object,
        and then reading the hdf contents via the object's methods.

        Returns
        -------
        out
            str

        """
        return('Model1D()')


    def createHdf(self, parent, myName, withPosterior=True, nRepeats=None, fillvalue=None):
        """Create the Metadata for a Model1D in a HDF file

        Creates a new group in a HDF file under h5obj.
        A nested heirarchy will be created.
        This method can be used in an MPI parallel environment, if so however,
        a) the hdf file must have been opened with the mpio driver, and
        b) createHdf must be called collectively,
        i.e., called by every core in the MPI communicator that was used to open the file.
        In order to create large amounts of empty space before writing to it in parallel,
        the nRepeats parameter will extend the memory in the first dimension.

        Parameters
        ----------
        h5obj : h5py._hl.files.File or h5py._hl.group.Group
            A HDF file or group object to create the contents in.
        myName : str
            The name of the group to create.
        nRepeats : int, optional
            Inserts a first dimension into the first dimension of each attribute of the Model1D of length nRepeats.
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

        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()

        self.nCells.createHdf(grp, 'nCells', withPosterior=withPosterior, nRepeats=nRepeats)
        self.top.createHdf(grp, 'top', nRepeats=nRepeats, fillvalue=fillvalue)
        self.depth.createHdf(grp, 'depth', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.thk.createHdf(grp, 'thk', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.par.createHdf(grp, 'par', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        #self.magnetic_permeability.createHdf(grp, 'magnetic_permeability', nRepeats=nRepeats, fillvalue=fillvalue)
        #self.magnetic_susceptibility.createHdf(grp, 'magnetic_susceptibility', nRepeats=nRepeats, fillvalue=fillvalue)

        try:
            grp.create_dataset('pWheel', data=self.pWheel)
        except:
            pass
        try:
            grp.create_dataset('zmin', data=self.minDepth)
        except:
            pass
        try:
            grp.create_dataset('zmax', data=self.maxDepth)
        except:
            pass
        try:
            grp.create_dataset('kmax', data=self.maxLayers)
        except:
            pass
        try:
            grp.create_dataset('hmin', data=self.minThickness)
        except:
            pass
        grp.create_dataset('hasHalfspace', data=self.hasHalfspace)


    def writeHdf(self, h5obj, myName, withPosterior=True, index=None):
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
        myName : str
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

        grp = h5obj.get(myName)

        self.nCells.writeHdf(grp, 'nCells',  withPosterior=withPosterior, index=index)
        self.top.writeHdf(grp, 'top', index=index)
        nCells = np.int(self.nCells[0])

        if (index is None):
            index = np.s_[:nCells]
        else:
            index = np.s_[index, :nCells]

        self.depth.writeHdf(grp, 'depth',  withPosterior=withPosterior, index=index)
        self.thk.writeHdf(grp, 'thk',  withPosterior=withPosterior, index=index)
        self.par.writeHdf(grp, 'par',  withPosterior=withPosterior, index=index)
        #self.magnetic_permeability.writeHdf(grp, 'magnetic_permeability',  withPosterior=withPosterior, index=i)
        #self.magnetic_susceptibility.writeHdf(grp, 'magnetic_susceptibility',  withPosterior=withPosterior, index=i)


    def toHdf(self, hObj, myName):
        """Write the Model1D to an HDF object

        Creates and writes a new group in a HDF file under h5obj.
        A nested heirarchy will be created.
        This function modifies the file metadata and writes the contents at the same time and
        should not be used in a parallel environment.

        Parameters
        ----------
        h5obj : h5py._hl.files.File or h5py._hl.group.Group
            A HDF file or group object to write the contents to.
        myName : str
            The name of the group to write to.

        Examples
        --------
        >>> import numpy as np
        >>> from geobipy import Model1D
        >>> import h5py
        >>> par = StatArray(np.linspace(0.01, 0.10, 10), "Test", "units")
        >>> thk = StatArray(np.ones(10) * 10.0)
        >>> mod = Model1D(nCells = 10, parameters=par, thickness=thk)

        """
        # Create a new group inside h5obj
        self.createHdf(hObj, myName, withPosterior=True)
        self.writeHdf(hObj, myName, withPosterior=True)


    def fromHdf(self, grp, index=None):
        """Read the class from a HDF group

        Given the HDF group object, read the contents into an Model1D class.

        Parameters
        ----------
        h5obj : h5py._hl.group.Group
            A HDF group object to write the contents to.
        index : slice, optional
            If the group was created using the nRepeats option, index specifies the index'th entry from which to read the data.

        """
        tmp = Model1D()

        item = grp.get('minThk')
        if (not item is None):
            tmp.minThickness = np.array(item)
        item = grp.get('hmin')
        if (not item is None):
            tmp.minThickness = np.array(item)

        item = grp.get('zmin')
        if (not item is None):
            tmp.minDepth = np.array(item)

        item = grp.get('zmax')
        if (not item is None):
            tmp.maxDepth = np.array(item)

        item = grp.get('pWheel')
        if (not item is None):
            tmp.pWheel = np.array(item)

        item = grp.get('hasHalfspace')
        if (not item is None):
            tmp.hasHalfspace = np.array(item)

        if grp['par/data'].ndim > 1:
            assert not index is None, ValueError("File was created with multiple Model1Ds, must specify an index")

        item = grp.get('nCells')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._nCells = obj.fromHdf(item, index=index)

        if grp['par/data'].ndim == 1:
            i = index
        else:
            i = np.s_[index, :np.int(tmp.nCells[0])]

        item = grp.get('top')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        tmp._top = obj.fromHdf(item, index=index)

        item = grp.get('par')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        obj = obj.resize(np.int(tmp.nCells[0]))
        tmp._par = obj.fromHdf(item, index=i)

        item = grp.get('depth')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        obj = obj.resize(np.int(tmp.nCells[0]))
        tmp._depth = obj.fromHdf(item, index=i)

        item = grp.get('thk')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        obj = obj.resize(np.int(tmp.nCells[0]))
        tmp._thk = obj.fromHdf(item, index=i)

        #item = grp.get('magnetic_permeability'); obj = eval(cF.safeEval(item.attrs.get('repr')));
        #obj = obj.resize(tmp.nCells[0]); tmp.magnetic_permeability = obj.fromHdf(item, index=i)

        #item = grp.get('magnetic_susceptibility'); obj = eval(cF.safeEval(item.attrs.get('repr')));
        #obj = obj.resize(tmp.nCells[0]); tmp.magnetic_susceptibility = obj.fromHdf(item, index=i)

        if (tmp.nCells[0] > 0):
            tmp._dpar = StatArray.StatArray(np.int(tmp.nCells[0]) - 1, 'Derivative', tmp.par.units + '/m')
        else:
            tmp._dpar = None

        return tmp
