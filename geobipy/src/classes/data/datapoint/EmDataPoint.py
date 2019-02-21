from .DataPoint import DataPoint
from ...model.Model1D import Model1D
from ....classes.core.StatArray import StatArray
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
        self.relErr = StatArray(self.nSystems, '$\epsilon_{Relative}x10^{2}$', '%')
        # StatArray of Additive Errors
        self.addErr = StatArray(self.nSystems, '$\epsilon_{Additive}$', self._data.units)
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


    def propose(self, height, rErr, aErr, calibration):
        """Propose a new EM data point given the specified attached propsal distributions

        Parameters
        ----------
        rEerr : bool
            Propose a new relative error.
        aEerr : bool
            Propose a new additive error.
        height : bool
            Propose a new observation height.
        calibration : bool
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
        if (height):  # Update the candidate data elevation (if required)
            # Generate a new elevation
            other.z[:] = self.z.proposal.rng(1)
            if other.z.hasPrior():
                p = other.z.probability()
                while p == 0.0 or p == -np.inf:
                    other.z[:] = self.z.proposal.rng(1)
                    p = other.z.probability()
            # Update the mean of the proposed elevation
            other.z.proposal.mean[:] = other.z
            
        if (rErr):    
            # Generate a new error
            other.relErr[:] = self.relErr.proposal.rng(1)
            if other.relErr.hasPrior():
                p = other.relErr.probability()
                while p == 0.0 or p == -np.inf:
                    other.relErr[:] = self.relErr.proposal.rng(1)
                    p = other.relErr.probability()
            # Update the mean of the proposed errors
            other.relErr.proposal.mean[:] = other.relErr
            
        if (aErr):
            # Generate a new error
            other.addErr[:] = self.addErr.proposal.rng(1)
            if other.addErr.hasPrior():
                p = other.addErr.probability()
                while p == 0.0 or p == -np.inf:
                    other.addErr[:] = self.addErr.proposal.rng(1)
                    p = other.addErr.probability()
            # Update the mean of the proposed errors
            other.addErr.proposal.mean[:] = other.addErr

        if (calibration):  # Update the calibration parameters for the candidate data (if required)
            # Generate new calibration errors
            other.calibration[:] = self.calibration.proposal.rng(1)
            # Update the mean of the proposed calibration errors
            other.calibration.proposal.mean[:] = other.calibration
        return other


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

        c = StatArray(np.logspace(minConductivity, maxConductivity, nSamples), 'Conductivity', '$S/m$')
        PhiD = StatArray(c.size, 'Data Misfit', '')
        mod = Model1D(1)
        for i in range(c.size):
            mod.par[0] = c[i]
            self.forward(mod)
            PhiD[i] = self.dataMisfit()
        plt.loglog(c, PhiD, **kwargs)
        cP.xlabel(c.getNameUnits())
        cP.ylabel('Data misfit')


    def summary(self, out=False):
        msg = ("{} \n"
                "Line number: {} \n"
                "Fiducial: {}\n").format(super().summary(True), self.lineNumber, self.fiducial)
        for s in self.system:
            msg += s.summary(True)

        return msg if out else print(msg)



