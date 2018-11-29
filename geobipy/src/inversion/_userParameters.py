""" @_userParameters_Class
Parent handler for user defined parameters. Checks that the input user parameters match for a given data point.
This provides a bit more robust checking when the user is new to the codes, and must modify an input parameter class file.
"""
#from ..base import Error as Err
from ..classes.core.myObject import myObject
from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.statistics.Hitmap2D import Hitmap2D
import numpy as np
from ..base.customFunctions import isInt


class _userParameters(myObject):
    """ Handler class to user defined parameters. Allows us to check a users input parameters in the backend """

    def __init__(self):
        raise NotImplementedError()

    def check(self, D):
        """ Check that the specified input parameters match the type of data point"""

        if (isinstance(D, FdemDataPoint)):
            N1 = D.sys.nFreq
#            N2 = 2 * N1
            nCalibration = 4
        elif (isinstance(D, TdemDataPoint)):
            N1 = D.nSystems
#            N2 = N1
            nCalibration = 0
        else:
            assert False, TypeError('Invalid DataPoint type used')

        # Check the number of Markov chains
        self.nMC = np.int(self.nMC)
        assert isInt(self.nMC), 'nMC must be a numpy integer'
        assert self.nMC > 1000, 'Number of Markov Chain iterations nMC must be > 1000'

        # Check the minumum layer depth
        assert isinstance(self.minDepth, float), 'minDepth must be a float (preferably np.float64)'

        # Check the maximum layer depth
        assert isinstance(self.maxDepth, float), 'maxDepth must be a float (preferably np.float64)'

        # Check the minimum layer thickness
        if (not self.minThickness is None):
            assert isinstance(self.minThickness, float), 'minThickness must be a float (preferably np.float64)'

        # Check the maximum number of layers
        assert isInt(self.maxLayers), 'maxLayers must be an int'

        # Check the standard deviation of log(rhoa)=log(1+fac)
        assert isinstance(self.factor, float), 'factor must be a float (preferably np.float64)'

        # Check the standard deviation
        assert isinstance(self.GradientStd, float), 'GradientStd must be a float (preferably np.float64)'

        # Check the reference hitmap if given
        if (not self.referenceHitmap is None):
            assert isinstance(self.referenceHitmap, Hitmap2D), 'referenceHitmap must be of type Hitmap2D'

        # Check the relative Error
        assert self.relErr.size == D.nSystems, 'Relative Error must have size ' + str(D.nSystems)

        # Check the error floor
        assert self.addErr.size == D.nSystems, 'Additive Error must have size ' + str(D.nSystems)

        # Check the minimum relative error
        assert self.rErrMinimum.size == D.nSystems, 'Relative Error minimum must be size ' + str(D.nSystems)

        # Check the maximum relative error
        assert self.rErrMaximum.size == D.nSystems, 'Relative Error maximum must be size ' + str(D.nSystems)

        # Check the range allowed on the data point elevation
        assert isinstance(self.zRange, float), 'Elevation range must be a float (preferably np.float64)'

        # Check the calibration errors if they are used
        if (self.solveCalibration):
            assert self.calMean.shape == [N1, nCalibration], 'Calibration mean must have shape '+str([N1, nCalibration])

            assert self.calVar.shape == [N1, nCalibration], 'Calibration variance must have shape '+str([N1, nCalibration])

        # Checking Proposal Variables
        # Check the elevation proposal variance
        assert isinstance(self.propEl, float), 'Proposal elevation variance must be a float (preferably np.float64)'

        # Check the relative error proposal variance
        assert self.propRerr.size == D.nSystems, 'Proposal Relative Error variance must be size ' + str(D.nSystems)

        # Check the calibration proposal variance if they are used
        if (self.solveCalibration):
            assert self.propCal.shape == [N1, nCalibration], 'Proposal Calibration variance must have shape '+str([N1, nCalibration])

        # Check the covariance scaling parameter
        assert isinstance(self.covScaling, float), 'Covariance scaling must be a float (preferably np.float64)'

        # Check the data misfit multiplier factor
        assert isinstance(self.multiplier, float), 'Data misfit multiplier must be a float (preferably np.float64)'

        # Checking the Probability Wheel
        assert isinstance(self.pBirth, float), 'Probability of birth must be a float (preferably np.float64)'
        assert isinstance(self.pDeath, float), 'Probability of death must be a float (preferably np.float64)'
        assert isinstance(self.pPerturb, float), 'Probability of perturb must be a float (preferably np.float64)'
        assert isinstance(self.pNochange, float), 'Probability of no change must be a float (preferably np.float64)'

