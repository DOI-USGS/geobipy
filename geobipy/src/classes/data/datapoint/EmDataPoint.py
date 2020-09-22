from .DataPoint import DataPoint
from ...model.Model1D import Model1D
from ....classes.core import StatArray
from ...statistics.Histogram1D import Histogram1D
from ...statistics.Histogram2D import Histogram2D
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

        super().__init__(nChannelsPerSystem, x, y, z, elevation, data, std, predictedData, dataUnits, channelNames)

        # StatArray of Relative Errors
        self._relErr = StatArray.StatArray(self.nSystems, '$\epsilon_{Relative}x10^{2}$', '%')
        # StatArray of Additive Errors
        self._addErr = StatArray.StatArray(self.nSystems, '$\epsilon_{Additive}$', self._data.units)
        # Initialize the sensitivity matrix
        self.J = None

        self.fiducial = fiducial
        self.lineNumber = lineNumber
        self.errorPosterior = None


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


    def FindBestHalfSpace(self, minConductivity=1e-4, maxConductivity=1e4, nSamples=100):
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
        p = StatArray.StatArray(1, 'Conductivity', r'$\frac{S}{m}$')
        model = Model1D(1, parameters=p)
        for i in range(nSamples):
            model._par[0] = c[i]
            self.forward(model)
            PhiD[i] = self.dataMisfit(squared=True)
        i = np.argmin(PhiD)
        model._par[0] = c[i]
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
        PhiD = StatArray.StatArray(c.size, 'Normalized Data Misfit', '')
        mod = Model1D(1)
        for i in range(c.size):
            mod.par[0] = c[i]
            self.forward(mod)
            PhiD[i] = self.dataMisfit()
        plt.loglog(c, PhiD/self.nActiveChannels, **kwargs)
        cP.xlabel(c.getNameUnits())
        cP.ylabel('Data misfit')


    def setPriors(self, heightPrior=None, relativeErrorPrior=None, additiveErrorPrior=None):
        """Set the priors on the datapoint's perturbable parameters

        Parameters
        ----------
        heightPrior : geobipy.baseDistribution, optional
            The prior to attach to the height. Must be univariate
        relativeErrorPrior : geobipy.baseDistribution, optional
            The prior to attach to the relative error.
            If the datapoint has only one system, relativeErrorPrior is univariate.
            If there are more than one system, relativeErrorPrior is multivariate.
        additiveErrorPrior : geobipy.baseDistribution, optional
            The prior to attach to the relative error.
            If the datapoint has only one system, additiveErrorPrior is univariate.
            If there are more than one system, additiveErrorPrior is multivariate.

        """

        if not heightPrior is None:
            self.z.setPrior(heightPrior)

        if not relativeErrorPrior is None:
            self.relErr.setPrior(relativeErrorPrior)

        if not additiveErrorPrior is None:
            self.addErr.setPrior(additiveErrorPrior)


    def setProposals(self, heightProposal=None, relativeErrorProposal=None, additiveErrorProposal=None):
        """Set the proposals on the datapoint's perturbable parameters

        Parameters
        ----------
        heightProposal : geobipy.baseDistribution, optional
            The proposal to attach to the height. Must be univariate
        relativeErrorProposal : geobipy.baseDistribution, optional
            The proposal to attach to the relative error.
            If the datapoint has only one system, relativeErrorProposal is univariate.
            If there are more than one system, relativeErrorProposal is multivariate.
        additiveErrorProposal : geobipy.baseDistribution, optional
            The proposal to attach to the relative error.
            If the datapoint has only one system, additiveErrorProposal is univariate.
            If there are more than one system, additiveErrorProposal is multivariate.

        """

        if not heightProposal is None:
            self.z.setProposal(heightProposal)

        if not relativeErrorProposal is None:
            self.relErr.setProposal(relativeErrorProposal)

        if not additiveErrorProposal is None:
            self.addErr.setProposal(additiveErrorProposal)


    def setPosteriors(self, log=None):
        """ Set the posteriors based on the attached priors

        Parameters
        ----------
        log :

        """
        # Create a histogram to set the height posterior.
        self.setHeightPosterior()
        # # Initialize the histograms for the relative errors
        # self.setRelativeErrorPosterior()
        # # Set the posterior for the data point.
        # self.setAdditiveErrorPosterior(log=log)
        self.setErrorPosterior(log=log)


    def setHeightPosterior(self):
        """

        """
        if self.z.hasPrior:
            H = Histogram1D(bins = StatArray.StatArray(self.z.prior.bins(), name=self.z.name, units=self.z.units), relativeTo=self.z)
            self.z.setPosterior(H)


    def setErrorPosterior(self, log=None):
        """

        """
        if (not self.relErr.hasPrior) and (not self.addErr.hasPrior):
            return

        if self.relErr.hasPrior:
            rb = StatArray.StatArray(np.atleast_2d(self.relErr.prior.bins()), name=self.relErr.name, units=self.relErr.units)
            if not self.addErr.hasPrior:
                self.relErr.setPosterior([Histogram1D(bins = rb[i, :]) for i in range(self.nSystems)])
                return

        if self.addErr.hasPrior:
            ab = StatArray.StatArray(np.atleast_2d(self.addErr.prior.bins()), name=self.addErr.name, units=self.data.units)
            if not self.relErr.hasPrior:
                self.addErr.setPosterior([Histogram1D(bins = ab[i, :], log=log) for i in range(self.nSystems)])
                return

        self.errorPosterior = [Histogram2D(xBins=ab[i, :], yBins=rb[i, :], xlog=log) for i in range(self.nSystems)]

        # self.relErr.setPosterior([self.errorPosterior[i].marginalize(axis=1) for i in range(self.nSystems)])
        # self.addErr.setPosterior([self.errorPosterior[i].marginalize(axis=0) for i in range(self.nSystems)])


    # def setAdditiveErrorPosterior(self, log=None):

    #     assert self.addErr.hasPrior, Exception("Must set a prior on the additive error")

    #     aBins = self.addErr.prior.bins()
    #     binsMidpoint = 0.5 * aBins.max(axis=-1) + aBins.min(axis=-1)
    #     ab = np.atleast_2d(aBins)
    #     binsMidpoint = np.atleast_1d(binsMidpoint)

    #     self.addErr.setPosterior([Histogram1D(bins = StatArray.StatArray(ab[i, :], name=self.addErr.name, units=self.data.units), log=log, relativeTo=binsMidpoint[i]) for i in range(self.nSystems)])


    # def setRelativeErrorPosterior(self):

    #     rBins = self.relErr.prior.bins()
    #     if self.nSystems > 1:
    #         self.relErr.setPosterior([Histogram1D(bins = StatArray.StatArray(rBins[i, :], name='$\epsilon_{Relative}x10^{2}$', units='%')) for i in range(self.nSystems)])
    #     else:
    #         self.relErr.setPosterior(Histogram1D(bins = StatArray.StatArray(rBins, name='$\epsilon_{Relative}x10^{2}$', units='%')))


    def summary(self, out=False):
        msg = ("{} \n"
                "Line number: {} \n"
                "Fiducial: {}\n"
                "Relative Error {}\n"
                "Additive Error {}\n").format(super().summary(True), self.lineNumber, self.fiducial, self.relErr.summary(True), self.addErr.summary(True))
        for s in self.system:
            msg += s.summary(True)

        return msg if out else print(msg)


    def updatePosteriors(self):
        """Update any attached posteriors"""

        if self.z.hasPosterior:
            self.z.updatePosterior()

        if not self.errorPosterior is None:
            for i in range(self.nSystems):
                self.errorPosterior[i].update(xValues=self.addErr[i], yValues=self.relErr[i])
        else:
            if self.relErr.hasPosterior:
                self.relErr.updatePosterior()

            if self.addErr.hasPosterior:
                self.addErr.updatePosterior()



