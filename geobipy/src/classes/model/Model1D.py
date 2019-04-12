""" @Model1D_Class
Module describing a 1 Dimensional layered Model
"""
#from ...base import Error as Err
from ...classes.core.StatArray import StatArray
from .Model import Model
from ..mesh.RectilinearMesh2D import RectilinearMesh2D
from ..statistics.Hitmap2D import Hitmap2D
from ...base.logging import myLogger
from ..statistics.Distribution import Distribution
import numpy as np
import matplotlib.pyplot as plt
from ...base import customPlots as cP
from ...base.customFunctions import safeEval

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
        self.nCells = StatArray(1, '# of Cells', dtype=np.int32)
        if not nCells is None:
            assert (nCells >= 1), ValueError('nCells must >= 1')
            self.nCells[0] = nCells

        # Depth to the top of the model
        if top is None: 
            top = 0.0
        self.top = StatArray(1) + top
        

        if hasHalfspace:
            self._init_withHalfspace(nCells, top, parameters, depth, thickness)
        else:
            self._init_withoutHalfspace(nCells, top, parameters, depth, thickness)

        # StatArray of the change in physical parameters
        self.dpar = StatArray(self.nCells[0] - 1, 'Derivative', r"$\frac{"+self.par.getUnits()+"}{m}$")

        # StatArray of magnetic properties.
        self.chie = StatArray(self.nCells[0], "Magnetic Susceptibility", r"$\kappa$")
        self.chim = StatArray(self.nCells[0], "Magnetic Permeability", "$\frac{H}{m}$")

        # Initialize Minimum cell thickness
        self.minThickness = None
        # Initialize a minimum depth
        self.minDepth = None
        # Initialize a maximum depth
        self.maxDepth = None
        # Initialize a maximum number of layers
        self.maxLayers = None
        # Initialize a probability wheel
        self.pWheel = None
        # Set an index that keeps track of the last layer to be perturbed
        self.iLayer = np.int32(-1)

        self.Hitmap = None


    def _init_withHalfspace(self, nCells=None, top=None, parameters = None, depth = None, thickness = None):


        if (not depth is None and nCells is None):
            self.nCells[0] = depth.size + 1
        if (not thickness is None and nCells is None):
            self.nCells[0] = thickness.size + 1

        self.depth = StatArray(self.nCells[0], 'Depth', 'm')
        self.thk = StatArray(self.nCells[0], 'Thickness', 'm')
        self.par = StatArray(self.nCells[0])

        if (not depth is None):
            if (self.nCells > 1):
                assert depth.size == self.nCells-1, ValueError('Size of depth must equal {}'.format(self.nCells[0]-1))
                assert np.all(np.diff(depth) > 0.0), ValueError('Depths must monotonically increase')
            self.depth[:-1] = depth
            self.depth[-1] = np.inf
            self.thicknessFromDepth()
            

        if (not thickness is None):
            if nCells is None:
                self.nCells[0] = thickness.size + 1
            if (self.nCells > 1):
                assert thickness.size == self.nCells-1, ValueError('Size of thickness must equal {}'.format(self.nCells[0]-1))
                assert np.all(thickness > 0.0), ValueError('Thicknesses must be positive')
            self.thk[:-1] = thickness
            self.thk[-1] = np.inf
            self.depthFromThickness()

        # StatArray of the physical parameters
        if (not parameters is None):
            assert parameters.size == self.nCells, ValueError('Size of parameters must equal {}'.format(self.nCells[0]))
            self.par = StatArray(parameters)


    def _init_withoutHalfspace(self, nCells = None, top = None, parameters = None, depth = None, thickness = None):


        if (not depth is None and nCells is None):
            self.nCells[0] = depth.size
        if (not thickness is None and nCells is None):
            self.nCells[0] = thickness.size

        self.depth = StatArray(self.nCells[0], 'Depth', 'm')
        self.thk = StatArray(self.nCells[0], 'Thickness', 'm')
        self.par = StatArray(self.nCells[0])

        if (not depth is None):
            if (self.nCells > 1):
                assert depth.size == self.nCells, ValueError('Size of depth must equal {}'.format(self.nCells[0]))
                assert np.all(np.diff(depth) > 0.0), ValueError('Depths must monotonically increase')
            self.depth[:] = depth
            self.thicknessFromDepth()

        if (not thickness is None):
            if nCells is None:
                self.nCells[0] = thickness.size
            if (self.nCells > 1):
                assert thickness.size == self.nCells, ValueError('Size of thickness must equal {}'.format(self.nCells[0]))
                assert np.all(thickness > 0.0), ValueError('Thicknesses must be positive')
            self.thk[:] = thickness
            self.depthFromThickness()

        # StatArray of the physical parameters
        if (not parameters is None):
            assert parameters.size == self.nCells, ValueError('Size of parameters must equal {}'.format(self.nCells[0]))
            self.par = StatArray(parameters)


    def deepcopy(self):
        """Create a deepcopy
        
        Returns
        -------
        out : geobipy.Model1D
            Deepcopy of Model1D

        """
        other = Model1D(nCells=None)
        other.nCells = self.nCells.deepcopy()
        other.top = self.top
        other.depth = self.depth.deepcopy()
        other.thk = self.thk.deepcopy()
        other.minThickness = self.minThickness
        other.minDepth = self.minDepth
        other.maxDepth = self.maxDepth
        other.maxLayers = self.maxLayers
        other.par = self.par.deepcopy()
        other.dpar = self.dpar.deepcopy()
        other.chie = self.chie.deepcopy() #StatArray(other.nCells[0], "Electric Susceptibility", r"$\kappa$")
        other.chim = self.chim.deepcopy() #StatArray(other.nCells[0], "Magnetic Susceptibility", "$\frac{H}{m}$")
        other.pWheel = self.pWheel
        other.iLayer = self.iLayer
        other.Hitmap = self.Hitmap
        other.hasHalfspace = self.hasHalfspace
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
        tmp=Model1D(size,self.top)
        tmp.nCells = self.nCells
        tmp.depth = self.depth.pad(size)
        tmp.thk = self.thk.pad(size)
        tmp.par = self.par.pad(size)
        tmp.chie = self.chie.pad(size)
        tmp.chim = self.chim.pad(size)
        tmp.iLayer = self.iLayer
        tmp.dpar=self.dpar.pad(size-1)
        if (not self.minDepth is None): tmp.minDepth=self.minDepth
        if (not self.maxDepth is None): tmp.maxDepth=self.maxDepth
        if (not self.maxLayers is None): tmp.maxLayers=self.maxLayers
        if (not self.minThickness is None): tmp.minThickness=self.minThickness
        if (not self.pWheel is None): tmp.pWheel=self.pWheel
        if (not self.Hitmap is None): tmp.Hitmap=self.Hitmap
        return tmp


    def depthFromThickness(self):
        """Given the thicknesses of each layer, create the depths to each interface. The last depth is inf for the halfspace."""
        self.depth[0] = self.thk[0]
        for i in range(1, self.nCells[0]):
            self.depth[i] = self.depth[i - 1] + self.thk[i]

        if self.hasHalfspace:
            self.depth[-1] = np.infty


    def thicknessFromDepth(self):
        """Given the depths to each interface, compute the layer thicknesses. The last thickness is nan for the halfspace."""
        self.thk = self.thk.resize(self.nCells[0])
        self.thk[0] = self.depth[0]
        for i in range(1, self.nCells[0]):
            self.thk[i] = self.depth[i] - self.depth[i - 1]

        if self.hasHalfspace:
            self.thk[-1] = np.inf


    def priorProbability(self, sPar, sGradient, limits=None, components=False):
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
        limits : array_like, optional
            Bound the parameter value.  If the parameter value falls outside of the limits, -inf is returned.
        components : bool, optional
            Return all components used in the final probability as well as the final probability

        Returns
        -------
        probability : numpy.float64
            The probability
        components : array_like, optional
            Return the components of the probability, i.e. the individually evaluated priors as a second return argument if comonents=True on input.

        """

        P_parameter = np.float64(0.0)
        P_gradient = np.float64(0.0)

        probability = np.float64(0.0)

        # Check that the parameters are within the limits if they are bound
        if (not limits is None):
            if (np.min(self.par) <= limits[0]):
                probability = -np.infty
            if (np.max(self.par) >= limits[1]):
                probability = -np.infty

        # Probability of the number of layers
        P_nCells = self.nCells.probability()
        probability += P_nCells

        # Probability of depth given nCells
        P_depthcells = np.log(self.depth.probability(self.nCells))
        probability += P_depthcells

        # Evaluate the prior based on the assigned hitmap
        if (not self.Hitmap is None):
            self.evaluateHitmapPrior(self.Hitmap)

        # Probability of parameter
        if sPar:  
            P_parameter = self.par.probability(np.log(self.par))
            probability += P_parameter

        # Probability of model gradient
        if sGradient:  
            P_gradient = self.smoothModelPrior(self.minThickness)
            probability += P_gradient

        # probability=np.float64(probability)
        if components:
            return probability, np.asarray([P_nCells, P_depthcells, P_parameter, P_gradient])
        return probability


    def insertLayer(self, z, par=None):
        """Insert a new layer into a model at a given depth

        Parameters
        ----------
        z : numpy.float64
            Depth at which to insert a new interface
        par : numpy.float64, optional 
            Value of the parameter for the new layer
            
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
        other.nCells += 1
        # Insert the new layer depth
        other.depth = self.depth.insert(i, z)
        if (par is None):
            if (i >= self.par.size):
                i -= 2
                other.par = other.par.insert(i, self.par[i])
                other.depth[-2] = other.depth[-1]
            else:
                other.par = other.par.insert(i, self.par[i])
        else:
            other.par = other.par.insert(i, par)
        # Get the new thicknesses
        other.thicknessFromDepth()
        # Reset ChiE and ChiM
        other.chie = StatArray(other.nCells[0], "Electric Susceptibility", r"$\kappa$")
        other.chim = StatArray(other.nCells[0], "Magnetic Susceptibility", r"$\frac{H}{m}$")
        # Resize the parameter gradient
        other.dpar = other.dpar.resize(other.par.size - 1)
        other.iLayer = i
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

        assert i < self.nCells[0] - 1, ValueError("i must be less than the number of cells - 1{}".format(self.nCells[0]-1))
        # if (i >= self.nCells[0] - 1):
        #     return self
        # Deepcopy the 1D Model to ensure priors and proposals are passed
        other = self.deepcopy()
        # Decrease the number of cells
        other.nCells -= 1
        # Remove the interface depth
        other.depth = other.depth.delete(i)
        # Get the new thicknesses
        other.thicknessFromDepth()
        # Take the average of the deleted layer and the one below it
        other.par = other.par.delete(i)
        other.par[i] = 0.5 * (self.par[i] + self.par[i + 1])
        # Reset ChiE and ChiM
        other.chie = other.chie.delete(i)
        other.chim = other.chim.delete(i)
        # Resize the parameter gradient
        other.dpar = other.dpar.resize(other.par.size - 1)
        other.iLayer = np.int64(i)

        return other


    def smoothModelPrior(self, hmin=0.0):
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
        assert (self.dpar.hasPrior()), TypeError('No prior defined on parameter gradient. Use Model1D.dpar.setPrior() to set the prior.')
        self.dpar[:] = (np.diff(np.log(self.par))) / (np.log(self.thk[:-1]) - hmin)
        return self.dpar.probability()


    def makePerturbable(self, pWheel, minDepth, maxDepth, maxLayers, prng=None, minThickness=None):
        """Setup a model such that it can be randomly perturbed.

        Parameters
        ----------
        pWheel : array_like
            Probability of birth, death, perturb, and no change for the model
            e.g. pWheel = [0.5, 0.25, 0.15, 0.1]
        minDepth : float64
            Minimum depth possible for the model
        maxDepth : float64
            Maximum depth possible for the model
        maxLayers : int
            Maximum number of layers allowable in the model
        prng : numpy.random.RandomState(), optional
            Random number generator, if none is given, will use numpy's global generator.
        minThickness : float64, optional
            Minimum thickness of any layer. If minThickness = None, minThickness is computed from minDepth, maxDepth, and maxLayers (recommended).
               
        See Also
        --------
        geobipy.Model1D.perturb : For a description of the perturbation cycle.
        """

        assert np.size(pWheel) == 4, ValueError('pWheel must have size 4')
        assert minDepth > 0.0, ValueError("minDepth must be > 0.0")
        assert maxDepth > 0.0, ValueError("maxDepth must be > 0.0")
        assert maxLayers > 0.0, ValueError("maxLayers must be > 0.0")
        self.pWheel = np.cumsum(pWheel/np.sum(pWheel))  # Assign the probability Wheel
        if (minThickness is None):
            # Assign a minimum possible thickness
            self.minThickness = np.log((maxDepth - minDepth) / (2 * maxLayers))
        else:
            self.minThickness = minThickness
        self.minDepth = np.log(minDepth)  # Assign the log of the min depth
        self.maxDepth = np.log(maxDepth)  # Assign the log of the max depth
        self.maxLayers = np.int32(maxLayers)
        # Assign a uniform distribution to the number of layers
        self.nCells.setPrior('UniformLog', 1, maxLayers, prng=prng)


    def perturb(self):
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
        out[1] : int
            Integer for the type of perturbation, [0,1,2,3] = [birth, death, change, no change]
        out[2] : list
            Two values for the cycle that was chosen. Only matters if the cycle was a birth, death, or change.
            For a birth, contains the depth of the new layer and None [newZ, None]
            For a death, contains the index of the layer that was deleted [iDeleted, None]
            
        See Also
        --------
        geobipy.Model1D.makePerturbable : Must be used before calling self.perturb
        
        """
        other = self.deepcopy()
        assert (not other.pWheel is None), ValueError('Please assign a probability wheel to the model with model1D.setProbabilityWheel()')
        prng = self.nCells.prior.prng
        # Pre-compute exponential values (Take them out of log space)
        hmin = np.exp(other.minThickness)
        zmin = np.exp(other.minDepth)
        zmax = np.exp(other.maxDepth)
        nTries = 10
        # This outer loop will allow the perturbation to change types. e.g. if the loop is stuck in a birthing
        # cycle, the number of tries will exceed 10, and we can try a different perturbation type.
        tryAgain = True  # Temporary to enter the loop
        while (tryAgain):
            tryAgain = False
            # Get a random probability from 0-1
            lifeCycle = prng.rand(1)
            if (other.nCells == 1):
                lifeCycle = np.round(lifeCycle)
            elif (other.nCells == other.nCells.prior.max):
                lifeCycle = np.amax([lifeCycle, other.pWheel[0]])

            # Determine what evolution we will follow.
            if (lifeCycle >= other.pWheel[2]):
                option = 3 # No change to anything
            if (lifeCycle < other.pWheel[0]):
                option = 0 # Create a layer
            elif (lifeCycle < other.pWheel[1]):
                option = 1 # Delete a layer
            elif (lifeCycle < other.pWheel[2]):
                option = 2 # Perturb a layers interface

            # Return if no change
            if (option == 3):
                return other, 3, [None, None]

            # Otherwise enter life-death-perturb cycle
            if (option == 0):  # Create a new layer
                success = False
                tries = 0
                while (not success):  # Continue while the new layer is smaller than the minimum
                    # Get the new depth
                    tmp = np.float64(prng.uniform(other.minDepth, other.maxDepth, 1))
                    newDepth = np.exp(tmp)
                    z = other.depth[:-1]
                    # Insert the new depth
                    i = z.searchsorted(newDepth)
                    z = z.insert(i, newDepth)
                    # Get the thicknesses
                    z = z.prepend(0.0)
                    h = np.min(np.diff(z[:]))
                    tries += 1
                    if (h > hmin):
                        success = True  # Exit if thickness is larger than minimum
                    if (tries == nTries):
                        success = True
                        tryAgain = True
                if (not tryAgain):
                    return other.insertLayer(newDepth), 0, [newDepth, None]

            if (option == 1):
                # Get the layer to remove
                iDeleted = np.int64(prng.uniform(0, other.nCells - 1, 1)[0])
                # Remove the layer and return
                return other.deleteLayer(iDeleted), 1, [iDeleted, None]

            if (option == 2):
                success = False
                k = other.nCells[0] - 1
                tries = 0
                while (not success):  # Continue while the perturbed layer is suitable
                    z = other.depth[:-1]
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
                        success = True
                    if (tries == nTries):
                        success = True
                        tryAgain = True
                if (not tryAgain):
                    other.iLayer = i
                    other.depth[i] += dz  # Perturb the depth in the model
                    other.thicknessFromDepth()
                    return other, 2, [i, dz]


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

        kwargs['xscale'] = kwargs.pop('xscale', 'log')
        
        # Repeat the last entry
        par = self.par.append(self.par[-1])
        if (reciprocateX):
            par = 1.0 / par
            
        z = self.depth.prepend(0.0)
        if self.hasHalfspace:
            if (self.maxDepth is None):
                z[-1] = 1.1 * self.depth[-2]
            else:
                z[-1] = 1.1 * np.exp(self.maxDepth)

        cP.step(x=par, y=z, **kwargs)

        if self.hasHalfspace:
            if (self.nCells == 1):
                h = 0.99
                p = par
            else:
                h = z[-2] + 0.75 * (z[-1] - z[-2])
                p = par[-1]
            
            plt.text(p, h, s=r'$\downarrow \infty$', fontsize=12)


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
                    d[-1] = 1.1 * self.depth[-2]
                else:
                    d[0] = 1.0               
            else:
                d[-1] = 1.1 * np.exp(self.maxDepth)

        ax = self.par.pcolor(*args, y = d + self.top, **kwargs)
        
        if self.hasHalfspace:
            h = 0.99*d[-1]
            if (self.nCells == 1):
                h = 0.99
            plt.text(0, h, s=r'$\downarrow \infty$', fontsize=12)

        return ax


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

        # Hist.sum = 0.0

        # dist = Distribution('MvNormal', par, var)

        # print(dist.mean, dist.variance)
        # print(Hist.x)

        # print(var.shape)
        # print(Hist.x.shape)

        # print(np.min(np.abs(Hist.x)))

        #Hist.arr[:,:] = dist.probability(Hist.x)

        # tmp = np.logspace(-3, 0, 100)
        # print(par[0], var[0])
        # print(tmp)
        # print(Hist.x)

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

            Hist.arr[i, :] = np.exp(dist.probability(np.log(Hist.x)))
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
        assert (np.size(par) == self.nCells[0]), 'par must have length nCells'
        assert (isinstance(mesh, RectilinearMesh2D)), TypeError('mesh must be a RectilinearMesh2D')

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

        if (self.nCells[0] == 1):
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






# if __name__ == "__main__":
#     # Create an initial model for the first iteration of the inversion
#     # Initialize a 1D model
#     Mod = Model1D(2, 'Current Model')
#     # Assign a prior to the number of layers
#     # Mod.nCells.setPrior('Gamma',3.0,3.0,(layerID+1)) # this is priK in matlab
#     # Assign the depth to the interface as half the bounds
#     Mod.maketest(10)
# #  Mod.par[:]=0.0036
#     Mod.minDepth = np.log(1.0)
#     Mod.maxDepth = np.log(150.0)
#     # Compute the probability wheel for birth, death, perturbation, and No
#     # change in the number of layers and depths
#     Mod.setPerturbation([0.16666667, 0.333333, 0.5, 1.0], 1.0, 150.0, 30)

#     # Compute the mean and std for the parameter
#     # Assign a normal distribution to the conductivities
#     Mod.par.setPrior('NormalLog', np.log(0.004), np.log(11.0))

#     # Set a temporary layer indexer
#     layerID = StatArray(30, 'LayerID')
#     layerID += np.arange(30)
#     # Set priors on the depth interfaces, given a number of layers
#     Mod.depth.setPrior(
#         'Order',
#         Mod.minDepth,
#         Mod.maxDepth,
#         Mod.minThk,
#         layerID)  # priZ

#     # Assign a prior to the derivative of the model
#     Mod.dpar.setPrior('NormalLog', 0.0, np.float64(1.5))

#     # b.summary()
# #  mpl.pyplot.figure()
# #  b.plotBlocks()
#     plt.figure(1)
#     Mod.plot()
#     plt.show()

#     priMu = np.log(0.01)
#     priStd = np.log(11.0)
#     zGrd = np.arange(0.5 * np.exp(Mod.minDepth), 1.1 * np.exp(Mod.maxDepth), 0.5 * np.exp(Mod.minThk))
#     mGrd = np.logspace(np.log10(np.exp(priMu - 4.0 * priStd)), np.log10(np.exp(priMu + 4.0 * priStd)), 250)

#     aMap = Hitmap2D([zGrd.size, mGrd.size], '', '', dtype=np.int32)
#     aMap.setXaxis(mGrd, 'Resistivity', '$\Omega m$')
#     aMap.setYaxis(zGrd, Mod.depth.name, Mod.depth.units)

#     Mod.getHitMap(aMap)

#     plt.figure(2)
#     plt.clf()
#     aMap.plot(invX=True, logX=True, flipY=True)
#     plt.show()

#     # Create synthetic variances of each layers conductivity
#     var = 0.1 * (np.zeros(Mod.nCells[0]) + 1)  # *Mod.par[-1]
# #  this=Mod.interp2depth(var,aMap)
# #
#     bMap = Hitmap2D([zGrd.size, mGrd.size], '', '')
#     bMap.setXaxis(mGrd, 'Resistivity', '$\Omega m$')
#     bMap.setYaxis(zGrd, Mod.depth.name, Mod.depth.units)
#     Mod.asPrior(var, bMap)

#     plt.figure(3)
#     plt.clf()
#     bMap.plot(invX=True, logX=True, flipY=True)
#     Mod.par = 1.0 / Mod.par
#     Mod.plot()
#     Mod.par = 1.0 / Mod.par
#     plt.show()

#     print('Prior Evaluation via Image')
#     print('Probability: ' + str(Mod.evaluateGridPrior(bMap)))

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


    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
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

        self.nCells.createHdf(grp, 'nCells', nRepeats=nRepeats)
        self.top.createHdf(grp, 'top', nRepeats=nRepeats, fillvalue=fillvalue)
        self.depth.createHdf(grp, 'depth', nRepeats=nRepeats, fillvalue=fillvalue)
        self.thk.createHdf(grp, 'thk', nRepeats=nRepeats, fillvalue=fillvalue)
        self.par.createHdf(grp, 'par', nRepeats=nRepeats, fillvalue=fillvalue)
        #self.chie.createHdf(grp, 'chie', nRepeats=nRepeats, fillvalue=fillvalue)
        #self.chim.createHdf(grp, 'chim', nRepeats=nRepeats, fillvalue=fillvalue)

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


    def writeHdf(self, h5obj, myName, index=None):
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

        self.nCells.writeHdf(grp, 'nCells',  index=index)
        self.top.writeHdf(grp, 'top',  index=index)
        nCells = self.nCells[0]
        if (index is None):
            i = np.s_[0:nCells]
        else:
            i = np.s_[index, 0:nCells]

        self.depth.writeHdf(grp, 'depth',  index=i)
        self.thk.writeHdf(grp, 'thk',  index=i)
        self.par.writeHdf(grp, 'par',  index=i)
        #self.chie.writeHdf(grp, 'chie',  index=i)
        #self.chim.writeHdf(grp, 'chim',  index=i)


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
        grp = hObj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.nCells.toHdf(grp, 'nCells')
        self.top.toHdf(grp,'top')
        self.depth.toHdf(grp, 'depth')
        self.thk.toHdf(grp, 'thk')

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

        self.par.toHdf(grp, 'par')
        self.chie.toHdf(grp, 'chie')
        self.chim.toHdf(grp, 'chim')
        grp.create_dataset('iLayer', data=self.iLayer)
        grp.create_dataset('hasHalfspace', data=self.hasHalfspace)


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
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp.nCells = obj.fromHdf(item, index=index)

        if grp['par/data'].ndim == 1:
            i = np.s_[:tmp.nCells[0]]
        else:
            i = np.s_[index, :tmp.nCells[0]]

        item = grp.get('top')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp.top = obj.fromHdf(item, index=index)

        item = grp.get('par')
        obj = eval(safeEval(item.attrs.get('repr')))
        obj = obj.resize(tmp.nCells[0])
        tmp.par = obj.fromHdf(item, index=i)

        item = grp.get('depth')
        obj = eval(safeEval(item.attrs.get('repr')))
        obj = obj.resize(tmp.nCells[0])
        tmp.depth = obj.fromHdf(item, index=i)

        item = grp.get('thk')
        obj = eval(safeEval(item.attrs.get('repr')))
        obj = obj.resize(tmp.nCells[0])
        tmp.thk = obj.fromHdf(item, index=i)

        #item = grp.get('chie'); obj = eval(safeEval(item.attrs.get('repr')));
        #obj = obj.resize(tmp.nCells[0]); tmp.chie = obj.fromHdf(item, index=i)

        #item = grp.get('chim'); obj = eval(safeEval(item.attrs.get('repr')));
        #obj = obj.resize(tmp.nCells[0]); tmp.chim = obj.fromHdf(item, index=i)

        if (tmp.nCells[0] > 0):
            tmp.dpar = StatArray(tmp.nCells[0] - 1, 'Derivative', tmp.par.units + '/m')
        else:
            tmp.dpar = None

        return tmp
