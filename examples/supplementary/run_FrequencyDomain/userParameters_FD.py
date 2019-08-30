""" @userParameters
Module describing the user input for GeoBIPy
"""
from geobipy.src.classes.data.datapoint.FdemDataPoint import FdemDataPoint
from geobipy.src.classes.data.datapoint.TdemDataPoint import TdemDataPoint
from geobipy.src.inversion._userParameters import _userParameters
from geobipy.src.classes.core.StatArray import StatArray
import numpy as np
from geobipy.src.base import Error as Err
from os.path import join

# Define whether this parameter file uses time domain or frequency domain data!
timeDomain=False
# Specify the folder to the data
dataDir=join('..','Data')

# Data File Name
dataFname=join(dataDir,'Resolve2.txt')
# System File Name
sysFname=join(dataDir,'FdemSystem2.stm')

if (timeDomain):
  dataInit='TdemData()'
else:
  dataInit='FdemData()'

class userParameters(_userParameters):
  """ User Interface Parameters for GeoBIPy """
  def __init__(self,D):
    """ Initialize user parameters, possibly with default values
    The user must modify this script!
    Usage: :paras=userParameters()
    """
    if isinstance(D,FdemDataPoint):
      assert not timeDomain, 'Specified time domain but passed in FdemDataPoint'
      N=D.sys.nFreq
      nSys=1
    elif isinstance(D,TdemDataPoint):
      assert timeDomain, 'Specified frequency domain but passed in TdemDataPoint'
      N=D.nSystems
      nSys=N

    ## Maximum number of Markov Chains
    self.nMC = np.int32(10001)
    ## Maximum number of layers in the 1D model
    self.kmax = np.int32(30)
    ## Minimum layer depth in metres
    self.zmin = 1.0
    ## Maximum layer depth in meters
    self.zmax = 100.0
    ## Minimum layer thickness, None = autocalculate
    self.hmin=None

    # Logicals for the priors to use
    ## Use the Prior on the parameter
    self.solveParameter = False # usePri
    ## Use the Prior on the difference in log resistivity diff(log(rho))
    self.solveGradient = True # useSmo

    ## Use the prior on the relative data errors
    self.solveRelativeError=True
    ## Use the prior on the additive data errors
    self.solveAdditiveError=True
    ## Use the prior on the data elevation
    self.solveElevation=False
    ## Use the prior on the calibration parameters for the data
    self.solveCalibration=False
 
    # Set the plotting logical
    self.plot=True
    self.verbose=False
    self.savePNG=True
    self.save=True
    self.iPlot=5000

    # Prior Details
    # 1D Model Prior Details
    # TODO: add an option for reading a prior model (e.g. borehole)
    ## Limit the parameter? If this is true, the limits are set to the mean of the conductivity prior +- 4 standard deviations
    self.LimitPar = True

    # Use another data points posterior parameter distribution as a prior reference distribution?
    self.refLine = '100101.0'
    self.refID = '155.0'
    self.referenceHitmap = None

    ## Standard Deviation of log(rho) = log(1 + factor)
    self.factor = np.float64(10.0) # fac

    ## Standard Deviation for the difference in layer resistivity
    self.GradientStd = np.float64(1.5) # smoStd

    # Data Prior Details
    # Set the Error model type
    # 0 = read absolute errors from data file
    # 1 = max(relErr*data,addErr)
    # 2 = sqrt((relErr*data)^2 + addErr^2)
    self.errorModel = 2
    ## Assign a percentage relative Error
    self.relErr = StatArray(nSys,'relativeError')+0.05
    # Relative Error Prior Details
    ## Minimum Relative Error
    self.rErrMinimum = StatArray(nSys,'Minimum Relative Error')+(0.001) # minRelErr
    ## Maximum Relative Error
    self.rErrMaximum = StatArray(nSys,'Maximum Relative Error')+(0.5) # maxRelErr

    ## Assign a noise floor
    self.addErr = StatArray(nSys,'addError')+5.0
    # Additive Error Prior Details
    ## Minimum Additive Error
    self.aErrMinimum = StatArray(nSys,'Minimum Additive Error')+3.0 # minAddErr
    ## Maximum Additive Error
    self.aErrMaximum = StatArray(nSys,'Maximum Additive Error')+20.0 # maxAddErr

    ## Elevation range allowed (m), either side of measured height
    self.zRange = np.float64(5.0) # rngEl

    ## Prior Means for gain, phase, InPhase Bias and Quadrature Bias
    if (self.solveCalibration):
      tmp = StatArray(4)
      tmp += [1.0,0.0,0.0,0.0]
      self.calMean=StatArray([N,4],'prior calibration means')
      for i in range(N):
        self.calMean[i,:]=tmp
      ## Prior variance for gain, phase, InPhase Bias and Quadrature Bias
      tmp = StatArray(4)
      tmp+=[.0225,.0019,19025.0,19025.0]
      self.calVar=StatArray([N,4],'prior calibration variance')
      for i in range(N):
        self.calVar[i,:]=tmp

    # Proposal Details
    ## Logical to determine whether to use the Steepest Descent or Stochastic Newton step direction
    self.stochasticNewton=True
    ## Variance of the elevation proposal
    self.propEl = np.float64(.01)
    ## Variance for the relative error proposal
    self.propRerr = StatArray(nSys,'Proposal relative error')+2.5e-7
    ## Variance for the additive error proposal
    self.propAerr = StatArray(nSys,'Proposal additive error')+1e-4
    ## Variance for the gain, phase, InPhase Bias and Quadrature Bias proposals
    if (self.solveCalibration):
      tmp = StatArray(4)
      tmp+=[2.5e-5,7.615e-7,25.0,25.0]
      self.propCal=StatArray([N,4],'Proposal Calibration Variance')
      for i in range(N):
        self.propCal[i,:]=tmp

    # Scaling Factors
    ## Initial scaling factor for proposal covariance
    self.covScaling = np.float64(1.5) # covScl
    ## Scaling factor for data misfit
    self.multiplier = np.float64(1.02) # chiFacInc

    # Evolution Probabilities
    ## Probability of Birth
    self.pBirth = np.float64(1.0/6.0)
    ## Probablitiy of Death
    self.pDeath = np.float64(1.0/6.0)
    ## Probability of perturbation
    self.pPerturb = np.float64(1.0/6.0)
    ## Probability of no change occuring
    self.pNochange = np.float64(0.5)

    # Set the limits for displaying resistivity
    self.dispLimits=[0.01,10000]
    # Display the resistivity?
    self.invertPar=True
    # Clipping Ratio for interface contrasts
    self.clipRatio = 0.5

    # Do not change
    # Specify the folder to the data
    self.dataDir = dataDir
    # Data File Name
    self.dataFname = dataFname
    # System File Name
    self.sysFname = sysFname
