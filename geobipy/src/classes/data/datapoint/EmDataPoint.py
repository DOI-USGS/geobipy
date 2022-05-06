from numpy.lib.function_base import meshgrid
from .DataPoint import DataPoint
from ....classes.core import StatArray
from ...mesh.RectilinearMesh1D import RectilinearMesh1D
from ...statistics.Histogram import Histogram
from ...model.Model import Model
# from ...statistics.Histogram2D import Histogram2D
from ...statistics.Distribution import Distribution
from ....base import utilities as cf
from ....base import plotting as cP
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class EmDataPoint(DataPoint):
    """Abstract EmDataPoint Class

    This is an abstract base class for TdemDataPoint and FdemDataPoint classes

    See Also
    ----------
    geobipy.src.classes.data.datapoint.FdemDataPoint
    geobipy.src.classes.data.datapoint.TdemDataPoint

    """

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=None,
                       components=None, channels_per_system=None,
                       data=None, std=None, predictedData=None,
                       channelNames=None,
                       lineNumber=0.0, fiducial=0.0, **kwargs):

        super().__init__(x = x, y = y, z = z, elevation = elevation,
                         data = data, std = std, predictedData = predictedData,
                         channelNames=channelNames, lineNumber=lineNumber, fiducial=fiducial, **kwargs)

        # Initialize the sensitivity matrix
        self.J = None

    @property
    def active(self):
        """Gets the indices to the observed data values that are not NaN

        Returns
        -------
        out : array of ints
            Indices into the observed data that are not NaN

        """
        d = self.data.copy()
        d[d <= 0.0] = np.nan
        return ~np.isnan(d)

    @property
    def channels_per_system(self):
        return self._channels_per_system

    @channels_per_system.setter
    def channels_per_system(self, values):
        if values is None:
            values = np.zeros(1, dtype=np.int32)
        else:
            values = np.atleast_1d(np.asarray(values, dtype=np.int32))

        self._channels_per_system = values

    @property
    def components(self):
        m = ('x', 'y', 'z')
        return [m[x] for x in self._components]

    @components.setter
    def components(self, values):

        m = {'x': 0,
             'y': 1,
             'z': 2}

        if values is None:
            values = ['z']
        else:

            if isinstance(values, str):
                values = [values]

            assert np.all([isinstance(x, str) for x in values]), TypeError('components must be list of str')

        self._components = np.asarray([m[x] for x in values], dtype=np.int32)

    @DataPoint.data.setter
    def data(self, values):

        if values is None:
            values = self.nChannels
        else:
            assert np.size(values) == self.nChannels, ValueError("data must have size {} not {}".format(self.nChannels, np.size(values)))

        self._data = StatArray.StatArray(values, "Data", self.units)

    @property
    def n_components(self):
        return np.size(self.components)

    @property
    def nChannels(self):
        return np.sum(self.channels_per_system)

    @property
    def nSystems(self):
        return np.size(self.channels_per_system)

    @DataPoint.predictedData.setter
    def predictedData(self, values):
        if values is None:
            values = self.nChannels
        else:
            if isinstance(values, list):
                assert len(values) == self.nSystems, ValueError("predictedData as a list must have {} elements".format(self.nSystems))
                values = np.hstack(values)
            assert values.size == self.nChannels, ValueError("Size of predictedData must equal total number of time channels {}".format(self.nChannels))
        self._predictedData = StatArray.StatArray(values, "Predicted Data", self.units)

    @property
    def system(self):
        return self._system

    @property
    def systemOffset(self):
        return np.hstack([0, np.cumsum(self.channels_per_system)])

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)

        # StatArray of calibration parameters
        # out.errorPosterior = self.errorPosterior
        # Initialize the sensitivity matrix
        out.J = deepcopy(self.J, memo)

        return out

    @staticmethod
    def new_model():
        mesh = RectilinearMesh1D(edges=StatArray.StatArray(np.asarray([0.0, np.inf]), 'Depth', 'm'))
        conductivity = StatArray.StatArray(mesh.nCells.item(), 'Conductivity', r'$\frac{S}{m}$')
        magnetic_susceptibility = StatArray.StatArray(mesh.nCells.item(), "Magnetic Susceptibility", r"$\kappa$")
        magnetic_permeability = StatArray.StatArray(mesh.nCells.item(), "Magnetic Permeability", "$\frac{H}{m}$")

        out = Model(mesh=mesh, values=conductivity)
        # out.setattr('magnetic_susceptibility', magnetic_susceptibility)
        # out.setattr('magnetic_permeability', magnetic_permeability)

        return out

    def find_best_halfspace(self, minConductivity=1e-4, maxConductivity=1e4, nSamples=100):
        """Computes the best value of a half space that fits the data.

        Carries out a brute force search of the halfspace conductivity that best fits the data.
        The profile of data misfit vs halfspace conductivity is not quadratic, so a bisection will not work.

        Parameters
        ----------
        minConductivity : float, optional
            The minimum conductivity to search over
        maxConductivity : float, optional
            The maximum conductivity to search over
        nSamples : int, optional
            The number of values between the min and max

        Returns
        -------
        out : np.float64
            The best fitting log10 conductivity for the half space

        """
        assert maxConductivity > minConductivity, ValueError("Maximum conductivity must be greater than the minimum")
        minConductivity = np.log10(minConductivity)
        maxConductivity = np.log10(maxConductivity)

        c = np.logspace(minConductivity, maxConductivity, nSamples)

        PhiD = np.zeros(nSamples)

        model = self.new_model()

        for i in range(nSamples):
            model.values[0] = c[i]
            self.forward(model)
            PhiD[i] = self.dataMisfit()

        i = np.argmin(PhiD)
        model.values[0] = c[i]
        return model

    def priorProbability(self, rErr, aErr, height, calibration, verbose=False):
        """Evaluate the probability for the EM data point given the specified attached priors

        Parameters
        ----------
        rEerr : bool
            Include the relative error when evaluating the prior
        aEerr : bool
            Include the additive error when evaluating the prior
        height : bool
            Include the elevation when evaluating the prior
        calibration : bool
            Include the calibration parameters when evaluating the prior
        verbose : bool
            Return the components of the probability, i.e. the individually evaluated priors

        Returns
        -------
        out : np.float64
            The evaluation of the probability using all assigned priors

        Notes
        -----
        For each boolean, the associated prior must have been set.

        Raises
        ------
        TypeError
            If a prior has not been set on a requested parameter

        """
        probability = np.float64(0.0)
        errProbability = np.float64(0.0)

        P_relative = np.float64(0.0)
        P_additive = np.float64(0.0)
        P_height = np.float64(0.0)
        P_calibration = np.float64(0.0)

        if rErr:  # Relative Errors
            P_relative = self.relErr.probability(log=True)
            errProbability += P_relative
        if aErr:  # Additive Errors
            P_additive = self.addErr.probability(log=True)
            errProbability += P_additive

        probability += errProbability
        if height:  # Elevation
            P_height = (self.z.probability(log=True))
            probability += P_height
        if calibration:  # Calibration parameters
            P_calibration = self.calibration.probability(log=True)
            probability += P_calibration

        if verbose:
            return probability, np.asarray([P_relative, P_additive, P_height, P_calibration])
        return probability

    def perturb(self):
        """Propose a new EM data point given the specified attached propsal distributions

        Parameters
        ----------
        newHeight : bool
            Propose a new observation height.
        newRelativeError : bool
            Propose a new relative error.
        newAdditiveError : bool
            Propose a new additive error.

        newCalibration : bool
            Propose new calibration parameters.

        Returns
        -------
        out : subclass of EmDataPoint
            The proposed data point

        Notes
        -----
        For each boolean, the associated proposal must have been set.

        Raises
        ------
        TypeError
            If a proposal has not been set on a requested parameter

        """
        self.perturbHeight()
        self.perturbRelativeError()
        self.perturbAdditiveError()

        # # Generate new calibration errors
        #     self.calibration[:] = self.calibration.proposal.rng(1)
        #     # Update the mean of the proposed calibration errors
        #     self.calibration.proposal.mean[:] = self.calibration

        #     self.calibrate()

    def perturbAdditiveError(self):
        if self.addErr.hasProposal:
            # Generate a new error
            self.addErr.perturb(imposePrior=True, log=True)
            # Update the mean of the proposed errors
            self.addErr.proposal.mean = self.addErr

    def perturbHeight(self):
        if self.z.hasProposal:
            # Generate a new elevation
            self.z.perturb(imposePrior=True, log=True)
            # Update the mean of the proposed elevation
            self.z.proposal.mean = self.z

    def perturbRelativeError(self):
        if self.relErr.hasProposal:
            # Generate a new error
            self.relErr.perturb(imposePrior=True, log=True)
            # Update the mean of the proposed errors
            self.relErr.proposal.mean = self.relErr

    def plotHalfSpaceResponses(self, minConductivity=-4.0, maxConductivity=2.0, nSamples=100, **kwargs):
        """Plots the reponses of different half space models.

        Parameters
        ----------
        minConductivity : float, optional
            The minimum log10 conductivity to search over
        maxConductivity : float, optional
            The maximum log10 conductivity to search over
        nInc : int, optional
            The number of increments between the min and max

        """

        # tmp = deepcopy(self)
        c = StatArray.StatArray(np.logspace(minConductivity, maxConductivity, nSamples), 'Conductivity', '$S/m$')
        PhiD = StatArray.StatArray(c.size, 'Normalized Data Misfit', '')
        
        model = self.new_model()

        for i in range(c.size):
            model.values[0] = c[i]
            self.forward(model)
            PhiD[i] = self.dataMisfit()
        plt.loglog(c, PhiD, **kwargs)
        cP.xlabel(c.getNameUnits())
        cP.ylabel('Data misfit')

    def update_posteriors(self):
        """Update any attached posteriors"""

        if self.z.hasPosterior:
            self.z.updatePosterior()

        if self.relErr.hasPosterior:
            self.relErr.updatePosterior()

        if self.addErr.hasPosterior:
            self.addErr.updatePosterior()