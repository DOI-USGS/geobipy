from .DataPoint import DataPoint
from ...model.Model1D import Model1D
from ....classes.core import StatArray
from ....base import customFunctions as cf
from ....base import customPlots as cP
import numpy as np
from ....base.logging import myLogger
import matplotlib.pyplot as plt


class EmDataPoint(DataPoint):
    """Abstract EmDataPoint Class

    This is an abstract base class for TdemDataPoint and FdemDataPoint classes

    See Also
    ----------
    geobipy.src.classes.data.datapoint.FdemDataPoint
    geobipy.src.classes.data.datapoint.TdemDataPoint

    """

    def __init__(self, nChannelsPerSystem=1, x=0.0, y=0.0, z=0.0, elevation=None, data=None, std=None, predictedData=None, dataUnits=None, channelNames=None, lineNumber=0.0, fiducial=0.0):

        DataPoint.__init__(self, nChannelsPerSystem, x, y, z, elevation, data, std, predictedData, dataUnits, channelNames)

        # StatArray of Relative Errors
        self.relErr = StatArray.StatArray(self.nSystems, '$\epsilon_{Relative}x10^{2}$', '%')
        # StatArray of Additive Errors
        self.addErr = StatArray.StatArray(self.nSystems, '$\epsilon_{Additive}$', self._data.units)
        # Initialize the sensitivity matrix
        self.J = None

        self.fiducial = fiducial
        self.lineNumber = lineNumber


    def FindBestHalfSpace(self, minConductivity=-4.0, maxConductivity=2.0, nSamples=100):
        """Computes the best value of a half space that fits the data.

        Carries out a brute force search of the halfspace conductivity that best fits the data.
        The profile of data misfit vs halfspace conductivity is not quadratic, so a bisection will not work.

        Parameters
        ----------
        minConductivity : float, optional
            The minimum log10 conductivity to search over
        maxConductivity : float, optional
            The maximum log10 conductivity to search over
        nInc : int, optional
            The number of increments between the min and max

        Returns
        -------
        out : np.float64
            The best fitting log10 conductivity for the half space

        """
        assert maxConductivity > minConductivity, ValueError("Maximum conductivity must be greater than the minimum")
        # ####lg.myLogger("Global"); ####lg.indent()
        c = np.logspace(minConductivity, maxConductivity, nSamples)
        PhiD = np.zeros(nSamples)
        Mod = Model1D(1)
        for i in range(nSamples):
            Mod.par[0] = c[i]
            self.forward(Mod)
            PhiD[i] = self.dataMisfit(squared=True)
        i = np.argmin(PhiD)
        # ####lg.dedent()
        return c[i]


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
            P_relative = self.relErr.probability()
            errProbability += P_relative
        if aErr:  # Additive Errors
            P_additive = self.addErr.probability()
            errProbability += P_additive

        probability += errProbability
        if height:  # Elevation
            P_height = (self.z.probability())
            probability += P_height
        if calibration:  # Calibration parameters
            P_calibration = self.calibration.probability()
            probability += P_calibration

        probability = np.float64(probability)
        
        if verbose:
            return probability, np.asarray([P_relative, P_additive, P_height, P_calibration])
        return probability


    def propose(self, newHeight, newRelativeError, newAdditiveError, newCalibration):
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
        other = self.deepcopy()
        if (newHeight):  # Update the candidate data elevation (if required)
            other.proposeHeight()
            
        if (newRelativeError):
            other.proposeRelativeError()
            
        if (newAdditiveError):
            other.proposeAdditiveError()

        if (newCalibration):  # Update the calibration parameters for the candidate data (if required)
            # Generate new calibration errors
            other.calibration[:] = self.calibration.proposal.rng(1)
            # Update the mean of the proposed calibration errors
            other.calibration.proposal.mean[:] = other.calibration
        return other


    def proposeAdditiveError(self):

        # Generate a new error
        self.addErr[:] = self.addErr.proposal.rng(1)
        if self.addErr.hasPrior():
            p = self.addErr.probability()
            while p == 0.0 or p == -np.inf:
                self.addErr[:] = self.addErr.proposal.rng(1)
                p = self.addErr.probability()
        # Update the mean of the proposed errors
        self.addErr.proposal.mean[:] = self.addErr

    
    def proposeHeight(self):
        # Generate a new elevation
        self.z[:] = self.z.proposal.rng(1)
        if self.z.hasPrior():
            p = self.z.probability()
            while p == 0.0 or p == -np.inf:
                self.z[:] = self.z.proposal.rng(1)
                p = self.z.probability()
        # Update the mean of the proposed elevation
        self.z.proposal.mean[:] = self.z

    
    def proposeRelativeError(self):
        # Generate a new error
        self.relErr[:] = self.relErr.proposal.rng(1)
        if self.relErr.hasPrior():
            p = self.relErr.probability()
            while p == 0.0 or p == -np.inf:
                self.relErr[:] = self.relErr.proposal.rng(1)
                p = self.relErr.probability()
        # Update the mean of the proposed errors
        self.relErr.proposal.mean[:] = self.relErr


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

        c = StatArray.StatArray(np.logspace(minConductivity, maxConductivity, nSamples), 'Conductivity', '$S/m$')
        PhiD = StatArray.StatArray(c.size, 'Data Misfit', '')
        mod = Model1D(1)
        for i in range(c.size):
            mod.par[0] = c[i]
            self.forward(mod)
            PhiD[i] = self.dataMisfit()
        plt.loglog(c, PhiD, **kwargs)
        cP.xlabel(c.getNameUnits())
        cP.ylabel('Data misfit')


    def setAdditiveErrorPrior(self, minimum, maximum, prng=None):
        minimum = np.atleast_1d(minimum)
        assert minimum.size == self.nSystems, ValueError("minimum must have {} entries".format(self.nSystems))
        assert np.all(minimum > 0.0), ValueError("minimum values must be > 0.0")
        maximum = np.atleast_1d(maximum)
        assert maximum.size == self.nSystems, ValueError("maximum must have {} entries".format(self.nSystems))
        assert np.all(maximum > 0.0), ValueError("maximum values must be > 0.0")

        self.addErr.setPrior('UniformLog', minimum, maximum, prng=prng)

    
    def setAdditiveErrorProposal(self, means, variances, prng=None):
        means = np.atleast_1d(means)
        assert means.size == self.nSystems, ValueError("means must have {} entries".format(self.nSystems))
        variances = np.atleast_1d(variances)
        assert variances.size == self.nSystems, ValueError("variances must have {} entries".format(self.nSystems))
        self.addErr.setProposal('MvNormal', means, variances, prng=prng)


    def setRelativeErrorPrior(self, minimum, maximum, prng=None):

        minimum = np.atleast_1d(minimum)
        assert minimum.size == self.nSystems, ValueError("minimum must have {} entries".format(self.nSystems))
        assert np.all(minimum > 0.0), ValueError("minimum values must be > 0.0")
        maximum = np.atleast_1d(maximum)
        assert maximum.size == self.nSystems, ValueError("maximum must have {} entries".format(self.nSystems))
        assert np.all(maximum > 0.0), ValueError("maximum values must be > 0.0")
        self.relErr.setPrior('UniformLog', minimum, maximum, prng=prng)


    def setRelativeErrorProposal(self, means, variances, prng=None):
        means = np.atleast_1d(means)
        assert means.size == self.nSystems, ValueError("means must have {} entries".format(self.nSystems))
        variances = np.atleast_1d(variances)
        assert variances.size == self.nSystems, ValueError("variances must have {} entries".format(self.nSystems))
        self.relErr.setProposal('MvNormal', means, variances, prng=prng)


    def summary(self, out=False):
        msg = ("{} \n"
                "Line number: {} \n"
                "Fiducial: {}\n").format(super().summary(True), self.lineNumber, self.fiducial)
        for s in self.system:
            msg += s.summary(True)

        return msg if out else print(msg)



