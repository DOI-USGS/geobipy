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
            N1 = D.system[0].nFrequencies
#            N2 = 2 * N1
            nCalibration = 4
        elif (isinstance(D, TdemDataPoint)):
            N1 = D.nSystems
#            N2 = N1
            nCalibration = 0
        else:
            assert False, TypeError('Invalid DataPoint type used')

        # Check the number of Markov chains
        self.nMarkovChains = np.int(self.nMarkovChains)
        assert isInt(self.nMarkovChains), TypeError('nMC must be a numpy integer')
        assert self.nMarkovChains >= 1000, ValueError('Number of Markov Chain iterations nMC must be >= 1000')

        # Check the minumum layer depth
        assert isinstance(self.minDepth, float), TypeError('minDepth must be a float (preferably np.float64)')

        # Check the maximum layer depth
        assert isinstance(self.maxDepth, float), TypeError('maxDepth must be a float (preferably np.float64)')

        # Check the minimum layer thickness
        if (not self.minThickness is None):
            assert isinstance(self.minThickness, float), TypeError('minThickness must be a float (preferably np.float64)')

        # Check the maximum number of layers
        assert isInt(self.maxLayers), ValueError('maxLayers must be an int')

        # Check the standard deviation of log(rhoa)=log(1+fac)
        assert isinstance(self.factor, float), TypeError('factor must be a float (preferably np.float64)')

        # Check the standard deviation
        assert isinstance(self.GradientStd, float), TypeError('GradientStd must be a float (preferably np.float64)')

        # Check the reference hitmap if given
        if (not self.referenceHitmap is None):
            assert isinstance(self.referenceHitmap, Hitmap2D), TypeError('referenceHitmap must be of type geobipy.Hitmap2D')

        # Check the relative Error
        assert self.relErr.size == D.nSystems, ValueError('Relative Error must have size {}'.format(D.nSystems))

        # Check the error floor
        assert self.addErr.size == D.nSystems, ValueError('Additive Error must have size {}'.format(D.nSystems))

        # Check the minimum relative error
        assert self.rErrMinimum.size == D.nSystems, ValueError('Relative Error minimum must be size {}'.format(D.nSystems))

        # Check the maximum relative error
        assert self.rErrMaximum.size == D.nSystems, ValueError('Relative Error maximum must be size {}'.format(D.nSystems))

        # Check the range allowed on the data point elevation
        assert isinstance(self.zRange, float), TypeError('Elevation range must be a float (preferably np.float64)')

        # Check the calibration errors if they are used
        if (self.solveCalibration):
            assert self.calMean.shape == [N1, nCalibration], ValueError('Calibration mean must have shape {}'.format([N1, nCalibration]))

            assert self.calVar.shape == [N1, nCalibration], ValueError('Calibration variance must have shape {}'.format([N1, nCalibration]))

        # Checking Proposal Variables
        # Check the elevation proposal variance
        assert isinstance(self.propEl, float), TypeError('Proposal elevation variance must be a float (preferably np.float64)')

        # Check the relative error proposal variance
        assert self.propRerr.size == D.nSystems, ValueError('Proposal Relative Error variance must be size {}'.format(D.nSystems))

        # Check the calibration proposal variance if they are used
        if (self.solveCalibration):
            assert self.propCal.shape == [N1, nCalibration], ValueError('Proposal Calibration variance must have shape {}'.forma([N1, nCalibration]))

        # Check the covariance scaling parameter
        assert isinstance(self.covScaling, float), TypeError('Covariance scaling must be a float (preferably np.float64)')

        # Check the data misfit multiplier factor
        assert isinstance(self.multiplier, float), TypeError('Data misfit multiplier must be a float (preferably np.float64)')

        # Checking the Probability Wheel
        assert isinstance(self.pBirth, float), TypeError('Probability of birth must be a float (preferably np.float64)')
        assert isinstance(self.pDeath, float), TypeError('Probability of death must be a float (preferably np.float64)')
        assert isinstance(self.pPerturb, float), TypeError('Probability of perturb must be a float (preferably np.float64)')
        assert isinstance(self.pNochange, float), TypeError('Probability of no change must be a float (preferably np.float64)')

