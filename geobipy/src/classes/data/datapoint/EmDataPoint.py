from .DataPoint import DataPoint
from ...model.Model1D import Model1D
from ....classes.core import StatArray
from ...statistics.Histogram1D import Histogram1D
from ...statistics.Histogram2D import Histogram2D
from ....base import utilities as cf
from ....base import plotting as cP
from copy import deepcopy
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

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=None, channels_per_system=1, components_per_channel=None, data=None, std=None, predictedData=None, channelNames=None, lineNumber=0.0, fiducial=0.0):

        super().__init__(x = x, y = y, z = z, elevation = elevation,
                         channels_per_system = channels_per_system,
                         components_per_channel = components_per_channel,
                         data = data, std = std, predictedData = predictedData,
                         channelNames=channelNames, lineNumber=lineNumber, fiducial=fiducial)

        # Initialize the sensitivity matrix
        self.J = None

    @property
    def nSystems(self):
        return np.size(self.channels_per_system)

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)

        # StatArray of calibration parameters
        # out.errorPosterior = self.errorPosterior
        # Initialize the sensitivity matrix
        out.J = deepcopy(self.J, memo)

        return out

    @property
    def active(self):
        """Gets the indices to the observed data values that are not NaN

        Returns
        -------
        out : array of ints
            Indices into the observed data that are not NaN

        """
        d = np.asarray(self.data)
        d[d <= 0.0] = np.nan
        return ~np.isnan(d)

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

        e = StatArray.StatArray(np.asarray([0.0, np.inf]), 'Depth', 'm')
        p = StatArray.StatArray(1, 'Conductivity', r'$\frac{S}{m}$')
        model = Model1D(edges=e, parameters=p)

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
        mod = Model1D(1, edges=np.asarray([0.0, np.inf]))
        for i in range(c.size):
            mod.par[0] = c[i]
            self.forward(mod)
            PhiD[i] = self.dataMisfit()
        plt.loglog(c, PhiD, **kwargs)
        cP.xlabel(c.getNameUnits())
        cP.ylabel('Data misfit')


    def set_priors(self, height_prior=None, relative_error_prior=None, additive_error_prior=None):
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

        if not height_prior is None:
            self.z.set_prior(height_prior)

        if not relative_error_prior is None:
            assert relative_error_prior.ndim == self.nSystems, ValueError("relative_error_prior must have {} dimensions".format(self.nSystems))
            self.relErr.set_prior(relative_error_prior)

        if not additive_error_prior is None:
            assert additive_error_prior.ndim == self.nSystems, ValueError("additive_error_prior must have {} dimensions".format(self.nSystems))
            self.addErr.set_prior(additive_error_prior)

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

        self.z.setProposal(heightProposal)
        self.relErr.setProposal(relativeErrorProposal)
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
        # self.set_predicted_data_posterior()
        # # Set the posterior for the data point.
        # self.setAdditiveErrorPosterior(log=log)
        self.setErrorPosterior(log=log)

        # self.set_predicted_data_posterior()

    def setHeightPosterior(self):
        """

        """
        if self.z.hasPrior:
            H = Histogram1D(edges = StatArray.StatArray(self.z.prior.bins(), name=self.z.name, units=self.z.units), relativeTo=self.z)
            self.z.setPosterior(H)

    def setErrorPosterior(self, log=None):
        """

        """
        if (not self.relErr.hasPrior) and (not self.addErr.hasPrior):
            return

        if self.relErr.hasPrior:
            rb = StatArray.StatArray(np.atleast_2d(self.relErr.prior.bins()), name=self.relErr.name, units=self.relErr.units)
            self.relErr.setPosterior([Histogram1D(edges = rb[i, :]) for i in range(self.nSystems)])

        if self.addErr.hasPrior:
            ab = StatArray.StatArray(np.atleast_2d(self.addErr.prior.bins()), name=self.addErr.name, units=self.data.units)
            self.addErr.setPosterior([Histogram1D(edges = ab[i, :], log=log) for i in range(self.nSystems)])

    @property
    def summary(self):
        msg = ("{} \n"
                "Line number: {} \n"
                "Fiducial: {}\n"
                "Relative Error {}\n"
                "Additive Error {}\n").format(super().summary, self.lineNumber, self.fiducial, self.relErr.summary, self.addErr.summary)
        for s in self.system:
            msg += s.summary

        return msg


    def updatePosteriors(self):
        """Update any attached posteriors"""

        if self.z.hasPosterior:
            self.z.updatePosterior()

        if self.relErr.hasPosterior:
            self.relErr.updatePosterior()

        if self.addErr.hasPosterior:
            self.addErr.updatePosterior()