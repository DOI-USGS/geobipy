from .DataPoint import DataPoint
from ...model.Model1D import Model1D
from ....classes.core import StatArray
from ...statistics.Histogram1D import Histogram1D
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
        self._relErr = StatArray.StatArray(self.nSystems, '$\epsilon_{Relative}x10^{2}$', '%')
        # StatArray of Additive Errors
        self._addErr = StatArray.StatArray(self.nSystems, '$\epsilon_{Additive}$', self._data.units)
        # Initialize the sensitivity matrix
        self.J = None

        self.fiducial = fiducial
        self.lineNumber = lineNumber


    @property
    def relErr(self):
        return self._relErr

    @relErr.setter
    def relErr(self, values):
        assert np.size(values) == self.nSystems, ValueError("relativeError must have length {}".format(self.nSystems))
        self._relErr[:] = values
    
    @property
    def addErr(self):
        return self._addErr

    @addErr.setter
    def addErr(self, values):
        assert np.size(values) == self.nSystems, ValueError("additiveError must have length {}".format(self.nSystems))
        self._addErr[:] = values


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


    def perturb(self, height, relativeError, additiveError, calibration):
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
        if (height):  # Update the candidate data elevation (if required)
            self.perturbHeight()
            
        if (relativeError):
            self.perturbRelativeError()
            
        if (additiveError):
            self.perturbAdditiveError()

        # Update the data errors using the updated relative errors
        if relativeError or additiveError:
            self.updateErrors(self.relErr, self.addErr)

        if (calibration):  # Update the calibration parameters for the candidate data (if required)
            # Generate new calibration errors
            self.calibration[:] = self.calibration.proposal.rng(1)
            # Update the mean of the proposed calibration errors
            self.calibration.proposal.mean[:] = self.calibration

            self.calibrate()


    def perturbAdditiveError(self):
        # Generate a new error
        self.addErr.perturb(imposePrior=True, log=True)
        # Update the mean of the proposed errors
        self.addErr.proposal.mean = self.addErr

    
    def perturbHeight(self):
        # Generate a new elevation
        self.z.perturb(imposePrior=True, log=True)
        # Update the mean of the proposed elevation
        self.z.proposal.mean = self.z

    
    def perturbRelativeError(self):
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


    def setAdditiveErrorPosterior(self):

        assert self.addErr.hasPrior, Exception("Must set a prior on the additive error")
        aBins = self.addErr.prior.bins()

        binsMidpoint = 0.5 * aBins.max(axis=-1) + aBins.min(axis=-1)
        ab = np.atleast_2d(aBins)
        binsMidpoint = np.atleast_1d(binsMidpoint)

        self.addErr.setPosterior([Histogram1D(bins = StatArray.StatArray(ab[i, :], name=self.addErr.name, units=self.data.units), relativeTo=binsMidpoint[i]) for i in range(self.nSystems)])



    def setAdditiveErrorPrior(self, minimum, maximum, prng=None):
        minimum = np.atleast_1d(minimum)
        assert minimum.size == self.nSystems, ValueError("minimum must have {} entries".format(self.nSystems))
        assert np.all(minimum > 0.0), ValueError("minimum values must be > 0.0")
        maximum = np.atleast_1d(maximum)
        assert maximum.size == self.nSystems, ValueError("maximum must have {} entries".format(self.nSystems))
        assert np.all(maximum > 0.0), ValueError("maximum values must be > 0.0")

        self.addErr.setPrior('Uniform', minimum, maximum, prng=prng)

    
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
        self.relErr.setPrior('Uniform', minimum, maximum, prng=prng)


    def setRelativeErrorProposal(self, means, variances, prng=None):
        means = np.atleast_1d(means)
        assert means.size == self.nSystems, ValueError("means must have {} entries".format(self.nSystems))
        variances = np.atleast_1d(variances)
        assert variances.size == self.nSystems, ValueError("variances must have {} entries".format(self.nSystems))
        self.relErr.setProposal('MvNormal', means, variances, prng=prng)


    def summary(self, out=False):
        msg = ("{} \n"
                "Line number: {} \n"
                "Fiducial: {}\n"
                "Relative Error {}\n"
                "Additive Error {}\n").format(super().summary(True), self.lineNumber, self.fiducial, self.relErr.summary(True), self.addErr.summary(True))
        for s in self.system:
            msg += s.summary(True)

        return msg if out else print(msg)



