from ....classes.core.myObject import myObject
from ...model.Model1D import Model1D
from ....classes.core.StatArray import StatArray
from ....base import customFunctions as cf
from ....base import customPlots as cP
import numpy as np
from ....base.logging import myLogger
import matplotlib.pyplot as plt


class EmDataPoint(myObject):
    """Abstract EmDataPoint Class

    This is an abstract base class for TdemDataPoint and FdemDataPoint classes

    See Also
    ----------
    geobipy.src.classes.data.datapoint.FdemDataPoint
    geobipy.src.classes.data.datapoint.TdemDataPoint

    """

    def __init__(self):
        """Abstract base class """
        raise NotImplementedError("Cannot instantiate this class, use a subclass")


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


    def getActiveData(self):
        """Gets the indices to the observed data values that are not NaN

        Returns
        -------
        out : array of ints
            Indices into the observed data that are not NaN

        """
        return cf.findNotNans(self.d)


    def likelihood(self):
        """Compute the likelihood of the current predicted data given the observed data and assigned errors

        Returns
        -------
        out : np.float64
            Likelihood of the data point

        """
        return self.p.probability(i=self.iActive)


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
            PhiD[i] = self.dataMisfit(squared=True)
        plt.loglog(c, PhiD, **kwargs)
        cP.xlabel(c.getNameUnits())
        cP.ylabel('Data misfit')


    def deltaD(self):
        """Get the difference between the predicted and observed data,

        .. math::
            \delta \mathbf{d} = \mathbf{d}^{obs} - \mathbf{d}^{pre}.

        Returns
        -------
        out : StatArray
            The residual between the active observed and predicted data 
            with size equal to the number of active channels.

        """
        return self.p[self.iActive] - self.d[self.iActive]


    def dataMisfit(self, squared=False):
        """Compute the :math:`L_{2}` norm squared misfit between the observed and predicted data

        .. math::
            \| \mathbf{W}_{d} (\mathbf{d}^{obs}-\mathbf{d}^{pre})\|_{2}^{2},
        where :math:`\mathbf{W}_{d}` are the reciprocal data errors.

        Parameters
        ----------
        squared : bool
            Return the squared misfit.

        Returns
        -------
        out : np.float64
            The misfit value.

        """
        assert not any(self.s[self.iActive] == 0.0), ValueError('Cannot compute the misfit when the data standard deviations are zero.')
        tmp2 = self.s[self.iActive]**-1.0
        PhiD = np.float64(np.sum((cf.Ax(tmp2, self.deltaD()))**2.0, dtype=np.float64))
        return PhiD if squared else np.sqrt(PhiD)

    
    def scaleJ(self, Jin, power=1.0):
        """ Scales a matrix by the errors in the given data

        Useful if a sensitivity matrix is generated using one data point, but must be scaled by the errors in another.

        Parameters
        ----------
        Jin : array_like
            2D array representing a sensitivity matrix
        power : float
            Power to raise the error level too. Default is simply reciprocal

        Returns
        -------
        Jout : array_like
            Sensitivity matrix divided by the data errors

        Raises
        ------
        ValueError
            If the number of rows in Jin do not match the number of active channels in the datapoint

        """
        assert Jin.shape[0] == self.iActive.size, ValueError("Number of rows of Jin must match the number of active channels in the datapoint {}".format(self.iActive.size))

        Jout = np.zeros(Jin.shape)
        Jout[:, :] = Jin * (np.repeat(self.s[self.iActive, np.newaxis] ** -power, Jout.shape[1], 1))
        return Jout


    def summary(self, out=False):
        """ Print a summary of the EMdataPoint """
        msg = 'EM Data Point: \n'
        msg += 'x: :' + str(self.x) + '\n'
        msg += 'y: :' + str(self.y) + '\n'
        msg += 'z: :' + str(self.z) + '\n'
        msg += 'e: :' + str(self.e) + '\n'
        msg += self.d.summary(True)
        msg += self.p.summary(True)
        msg += self.s.summary(True)
        if (out):
            return msg
        print(msg)


    def updateErrors(self, relativeErr, additiveErr):
        """Updates the data errors

        Updates the standard deviation of the data errors using the following model

        .. math::
            \sqrt{(\mathbf{\epsilon}_{rel} \mathbf{d}^{obs})^{2} + \mathbf{\epsilon}^{2}_{add}},
        where :math:`\mathbf{\epsilon}_{rel}` is the relative error, a percentage fraction and :math:`\mathbf{\epsilon}_{add}` is the additive error.

        If the predicted data have been assigned a multivariate normal distribution, the variance of that distribution is also updated as the squared standard deviations.

        Parameters
        ----------
        relativeErr : float or array_like
            A fraction percentage that is multiplied by the observed data.
        additiveErr : float or array_like
            An absolute value of additive error.

        Raises
        ------
        ValueError
            If relativeError is <= 0.0
        ValueError
            If additiveError is <= 0.0

        """
        assert relativeErr > 0.0, ValueError("relativeErr must be > 0.0")
        assert additiveErr > 0.0, ValueError("additiveErr must be > 0.0")

        tmp = (relativeErr * self.d)**2.0 + additiveErr**2.0

        if self.p.hasPrior():
            self.p.prior.variance[:] = tmp[self.iActive]

        self.s[:] = np.sqrt(tmp)