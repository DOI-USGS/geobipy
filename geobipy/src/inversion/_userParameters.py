""" @_userParameters_Class
Parent handler for user defined parameters. Checks that the input user parameters match for a given data point.
This provides a bit more robust checking when the user is new to the codes, and must modify an input parameter class file.
"""
#from ..base import Error as Err
from ..classes.core.myObject import myObject
from ..classes.core import StatArray
from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.statistics.Hitmap2D import Hitmap2D
import numpy as np
from ..base.customFunctions import isInt


class _userParameters(myObject):
    """ Handler class to user defined parameters. Allows us to check a users input parameters in the backend """

    def __init__(self, Datapoint):

        if isinstance(self.dataFilename , str):
            self.dataFilename = [self.dataFilename]
        if isinstance(self.systemFilename , str):
            self.systemFilename = [self.systemFilename]


        self.nMarkovChains = np.int32(self.nMarkovChains)
        self.maximumNumberofLayers = np.int32(self.maximumNumberofLayers)
        self.minimumDepth = np.float64(self.minimumDepth)
        self.maximumDepth = np.float64(self.maximumDepth)
        if not self.minimumThickness is None:
            self.minimumThickness = np.float64(self.minimumThickness)

        if not self.parameterLimits is None:
            assert np.size(self.parameterLimits) == 2, ValueError("parameterLimits must have 2 entries")
            assert self.parameterLimits[0] < self.parameterLimits[1], ValueError("parameterLimits must be in increasing order.")


        self.initialRelativeError = StatArray.StatArray(Datapoint.nSystems, 'Relative Error') + self.initialRelativeError
        self.minimumRelativeError = StatArray.StatArray(Datapoint.nSystems, 'Minimum Relative Error') + self.minimumRelativeError
        self.maximumRelativeError = StatArray.StatArray(Datapoint.nSystems, 'Maximum Relative Error') + self.maximumRelativeError

        assert np.all((self.minimumRelativeError <= self.initialRelativeError) & (self.initialRelativeError <= self.maximumRelativeError) ), ValueError("Initial relative error must be within the min max")

        self.initialAdditiveError = StatArray.StatArray(Datapoint.nSystems, 'Additive Error') + self.initialAdditiveError
        self.minimumAdditiveError = StatArray.StatArray(Datapoint.nSystems, 'Minimum Additive Error') + self.minimumAdditiveError
        self.maximumAdditiveError = StatArray.StatArray(Datapoint.nSystems, 'Maximum Additive Error') + self.maximumAdditiveError

        assert np.all((self.minimumAdditiveError <= self.initialAdditiveError) & (self.initialAdditiveError <= self.maximumAdditiveError)), ValueError("Initial additive error must be within the min max")

        self.maximumElevationChange = np.float64(self.maximumElevationChange)

        self.relativeErrorProposalVariance = StatArray.StatArray(Datapoint.nSystems, 'Proposal relative error') + self.relativeErrorProposalVariance
        # Proposal variance for the additive error
        self.additiveErrorProposalVariance = StatArray.StatArray(Datapoint.nSystems, 'Proposal additive error') + self.additiveErrorProposalVariance
        # Proposal variance of the elevation
        self.elevationProposalVariance = np.float64(self.elevationProposalVariance)

        self.pBirth = np.float64(self.pBirth)
        # Probablitiy that a layer is removed from the model.
        self.pDeath = np.float64(self.pDeath)
        # Probability that an interface in the model is perturbed.
        self.pPerturb = np.float64(self.pPerturb)
        # Probability of no change occuring to the layers of the model.
        self.pNochange = np.float64(self.pNochange)


        # Typically defaulted parameters
        # Standard Deviation of log(rho) = log(1 + factor)
        self.factor = np.float64(10.0) if self.factor is None else np.float64(self.factor)
        # Standard Deviation for the difference in layer resistivity
        self.gradientStd = np.float64(1.5) if self.gradientStd is None else np.float64(self.gradientStd)
        self.errorModel = 2


        try:
            self.parameterCovarianceScaling = np.float64(1.65) if self.parameterCovarianceScaling is None else  np.float64(self.parameterCovarianceScaling)
        except:
            self.parameterCovarianceScaling = np.float64(1.65)


        # Scaling factor for data misfit
        self.multiplier = np.float64(1.02) if self.multiplier is None else np.float64(self.multiplier)

        # Clipping Ratio for interface contrasts
        self.clipRatio = np.float64(0.5) if self.clipRatio is None else np.float64(self.clipRatio)

        # Use another data points posterior parameter distribution as a prior reference distribution?
        # Not implemented yet!
        self.refLine = None
        self.refID = None
        self.referenceHitmap = None

        # ## Prior Means for gain, phase, InPhase Bias and Quadrature Bias
        # if (self.solveCalibration):
        #     tmp = StatArray.StatArray(4)
        #     tmp += [1.0,0.0,0.0,0.0]
        #     self.calMean = StatArray.StatArray([N,4],'prior calibration means')
        #     for i in range(N):
        #         self.calMean[i,:] = tmp
        #     ## Prior variance for gain, phase, InPhase Bias and Quadrature Bias
        #     tmp = StatArray.StatArray(4)
        #     tmp += [.0225,.0019,19025.0,19025.0]
        #     self.calVar = StatArray.StatArray([N,4],'prior calibration variance')
        #     for i in range(N):
        #         self.calVar[i,:] = tmp

        try:
            self.ignoreLikelihood = False if self.ignoreLikelihood is None else self.ignoreLikelihood
        except:
            self.ignoreLikelihood = False

        self.check(Datapoint)


    def check(self, DataPoint):
        """ Check that the specified input parameters match the type of data point"""

        assert isinstance(DataPoint, (FdemDataPoint, TdemDataPoint)), TypeError('Invalid DataPoint type used')

        # Check the number of Markov chains
        self.nMarkovChains = np.int(self.nMarkovChains)
        assert isInt(self.nMarkovChains), TypeError('nMC must be a numpy integer')
        assert self.nMarkovChains >= 1000, ValueError('Number of Markov Chain iterations nMC must be >= 1000')

        # Check the minumum layer depth
        assert isinstance(self.minimumDepth, float), TypeError('minimumDepth must be a float (preferably np.float64)')

        # Check the maximum layer depth
        assert isinstance(self.maximumDepth, float), TypeError('maximumDepth must be a float (preferably np.float64)')

        # Check the minimum layer thickness
        if (not self.minimumThickness is None):
            assert isinstance(self.minimumThickness, float), TypeError('minimumThickness must be a float (preferably np.float64)')

        # Check the maximum number of layers
        assert isInt(self.maximumNumberofLayers), ValueError('maximumNumberofLayers must be an int')

        # Check the standard deviation of log(rhoa)=log(1+fac)
        assert isinstance(self.factor, float), TypeError('factor must be a float (preferably np.float64)')

        # Check the standard deviation
        assert isinstance(self.gradientStd, float), TypeError('gradientStd must be a float (preferably np.float64)')

        # Check the reference hitmap if given
        if (not self.referenceHitmap is None):
            assert isinstance(self.referenceHitmap, Hitmap2D), TypeError('referenceHitmap must be of type geobipy.Hitmap2D')

        # Check the relative Error
        assert self.initialRelativeError.size == DataPoint.nSystems, ValueError('Initial relative error must have size {}'.format(DataPoint.nSystems))

        # Check the minimum relative error
        assert self.minimumRelativeError.size == DataPoint.nSystems, ValueError('Minimum relative error must be size {}'.format(DataPoint.nSystems))

        # Check the maximum relative error
        assert self.maximumRelativeError.size == DataPoint.nSystems, ValueError('Maximum Relative error must be size {}'.format(DataPoint.nSystems))

        # Check the error floor
        assert self.initialAdditiveError.size == DataPoint.nSystems, ValueError('Initial additive error must have size {}'.format(DataPoint.nSystems))

        # Check the minimum relative error
        assert self.minimumAdditiveError.size == DataPoint.nSystems, ValueError('Minimum additive error must be size {}'.format(DataPoint.nSystems))

        # Check the maximum relative error
        assert self.maximumAdditiveError.size == DataPoint.nSystems, ValueError('Maximum additive error must be size {}'.format(DataPoint.nSystems))

        # Check the range allowed on the data point elevation
        assert isinstance(self.maximumElevationChange, float), TypeError('Elevation range must be a float (preferably np.float64)')

        # # Check the calibration errors if they are used
        # if (self.solveCalibration):
        #     assert self.calMean.shape == [N1, nCalibration], ValueError('Calibration mean must have shape {}'.format([N1, nCalibration]))

        #     assert self.calVar.shape == [N1, nCalibration], ValueError('Calibration variance must have shape {}'.format([N1, nCalibration]))

        # Checking Proposal Variables
        # Check the elevation proposal variance
        assert isinstance(self.elevationProposalVariance, float), TypeError('Proposal elevation variance must be a float (preferably np.float64)')

        # Check the relative error proposal variance
        assert self.relativeErrorProposalVariance.size == DataPoint.nSystems, ValueError('Proposal additive error variance must be size {}'.format(DataPoint.nSystems))

        # Check the relative error proposal variance
        assert self.additiveErrorProposalVariance.size == DataPoint.nSystems, ValueError('Proposal additive error variance must be size {}'.format(DataPoint.nSystems))

        # Check the calibration proposal variance if they are used
        # if (self.solveCalibration):
        #     assert self.propCal.shape == [N1, nCalibration], ValueError('Proposal Calibration variance must have shape {}'.forma([N1, nCalibration]))

        # Check the covariance scaling parameter
        assert isinstance(self.parameterCovarianceScaling, float), TypeError('Covariance scaling must be a float (preferably np.float64)')

        # Check the data misfit multiplier factor
        assert isinstance(self.multiplier, float), TypeError('Data misfit multiplier must be a float (preferably np.float64)')

        # Checking the Probability Wheel
        assert isinstance(self.pBirth, float), TypeError('Probability of birth must be a float (preferably np.float64)')
        assert isinstance(self.pDeath, float), TypeError('Probability of death must be a float (preferably np.float64)')
        assert isinstance(self.pPerturb, float), TypeError('Probability of perturb must be a float (preferably np.float64)')
        assert isinstance(self.pNochange, float), TypeError('Probability of no change must be a float (preferably np.float64)')

        if self.ignoreLikelihood:
            self.stochasticNewton = False

