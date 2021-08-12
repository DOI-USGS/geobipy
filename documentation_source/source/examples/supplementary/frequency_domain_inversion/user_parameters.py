from geobipy.src.inversion._userParameters import _userParameters

# General information about specifying parameters.
# The following list of parameters can be given either a single value or a list of values
# of length equal to the number of systems in the data. If one value is specified and there
# are multiple systems, that value is used for all of them.
# self.initialRelativeError
# self.minimumRelativeError
# self.maximumRelativeError
# self.initialAdditiveError
# self.minimumAdditiveError
# self.maximumAdditiveError
# self.relativeErrorProposalVariance
# self.additiveErrorProposalVariance

# -------------------------------------------------------
# General file structure information.
# -------------------------------------------------------
# Specify the folder to the data
dataDirectory = "..//Data"
# Data File Name. If there are multiple, encompass them with [ ].
dataFilename = dataDirectory + "//Resolve2.txt"
# System File Name. If there are multiple, encompass them with [ ].
systemFilename = dataDirectory + "//FdemSystem2.stm"

# Define the data type to invert

from geobipy import FdemData
data_type = FdemData()

class userParameters(_userParameters):
  """ User Interface Parameters for GeoBIPy """
  def __init__(self, DataPoint):
    """ File for the user to specify inpust to GeoBIPy. """

    ## Maximum number of Markov Chains per data point.
    self.nMarkovChains = 10000

    # -------------------------------------------------------
    # General GeoBIPy options.
    # -------------------------------------------------------
    # Interactively plot a single data point as it progresses
    self.plot = False
    # How often to update the plot. (lower is generally slower)
    self.plotEvery = 5000
    # Save a PNG of the final results for each data point.
    self.savePNG = True
    # Save the results of the McMC inversion to HDF5 files. (Generally always True)
    self.save = True

    # -------------------------------------------------------
    # Turning on or off different solvable parameters.
    # -------------------------------------------------------
    # Parameter Priors
    # solveParameter will prevent parameters from exploding very large or very small numbers.
    # solveGradient prevents large changes in parameters value from occurring.

    # Use the prior on the relative data errors
    self.solveRelativeError = True
    # Use the prior on the additive data errors
    self.solveAdditiveError = True
    # Use the prior on the data height
    self.solveHeight = True
    # Use the prior on the calibration parameters for the data
    self.solveCalibration = False

    # -------------------------------------------------------
    # Prior Details
    # -------------------------------------------------------

    # Earth model prior details
    # -------------------------
    # Maximum number of layers in the 1D model
    self.maximumNumberofLayers = 30
    # Minimum layer depth in metres
    self.minimumDepth = 1.0
    # Maximum layer depth in metres
    self.maximumDepth = 150.0
    # Minimum layer thickness.
    # If minimumThickness = None, it will be autocalculated.
    self.minimumThickness = None

    # Impose bounds on the parameter value
    # None, or a length 2 list i.e. [a, b]
    self.parameterLimits = None

    # Data prior details
    # ------------------
    # The data priors are imposed on three different aspects of the data.
    # The relative and additive error and the elevation of the data point.
    # Data uncertainty priors are used if solveRelativeError or solveAdditiveError are True.
    # If the data file contains columns of the estimated standard deviations, they are used as the initial values
    # when starting an McMC inversion. If the file does not contain these estimates, then the initial
    # values are used below as sqrt((relative * data)^2 + (additive)^2).

    # Assign an initial percentage relative Error
    # If the file contains no standard deviations, this will be used
    # to assign the initial data uncertainties.
    self.initialRelativeError = 0.05
    ## Relative Error Prior Details
    # Minimum Relative Error
    self.minimumRelativeError = 0.001
    # Maximum Relative Error
    self.maximumRelativeError = 0.5

    # Assign an initial additivr error level.
    # If the file contains no standard deviations, this will be used
    # to assign the initial data uncertainties.
    self.initialAdditiveError = 5.0
    # Additive Error Prior Details
    # Minimum Additive Error
    self.minimumAdditiveError = 3.0
    # Maximum Relative Error
    self.maximumAdditiveError = 20.0

    # Elevation range allowed (m), either side of measured height
    self.maximumElevationChange = 1.0

    # -------------------------------------------------------
    # Proposal details
    # -------------------------------------------------------

    # Data proposal details
    # ---------------------
    # The relative, additive, and elevation proposal variances are assigned to
    # normal distributions with a mean equal to its value in the current model (of the Markov chain)
    # These variances are used when we randomly choose a new value for that given variable.
    # Proposal variance for the relative error
    self.relativeErrorProposalVariance = 2.5e-7
    # Proposal variance for the additive error
    self.additiveErrorProposalVariance = 1.0e-4
    # Proposal variance of the elevation
    self.elevationProposalVariance = 0.01

    # Earth model proposal details
    # ----------------------------
    # Evolution Probabilities for earth model manipulation during the Markov chain.
    # These four values are internally scaled such that their sum is 1.0.
    # Probability that a layer is inserted into the model.
    self.pBirth = 1.0/6.0
    # Probablitiy that a layer is removed from the model.
    self.pDeath = 1.0/6.0
    # Probability that an interface in the model is perturbed.
    self.pPerturb = 1.0/6.0
    # Probability of no change occuring to the layers of the model.
    self.pNochange = 0.5

    # -------------------------------------------------------
    # Typically Defaulted parameters
    # -------------------------------------------------------
    # Standard Deviation of log(rho) = log(1 + factor)
    # Default is 10.0
    self.factor = None
    # Standard Deviation for the difference in layer resistivity
    # Default is 1.5
    self.gradientStd = None
    # Initial scaling factor for proposal covariance
    self.covScaling = None
    # Scaling factor for data misfit
    self.multiplier = None
    # Clipping Ratio for interface contrasts
    self.clipRatio = None
    # Only sample the prior
    self.ignoreLikelihood = False

    # Display the resistivity?
    self.reciprocateParameters = True

    # Don't change these.
    self.dataDirectory = dataDirectory
    self.dataFilename = dataFilename
    self.systemFilename = systemFilename

    self.verbose = False

    _userParameters.__init__(self, DataPoint)