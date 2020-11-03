""" @EMinversion1D_MCMC
Module defining a Markov Chain Monte Carlo approach to 1D EM inversion
"""
#%%
from ..classes.core.Stopwatch import Stopwatch
from ..classes.data.dataset.FdemData import FdemData
from ..classes.data.dataset.TdemData import TdemData
from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.model.Model1D import Model1D
from ..classes.core import StatArray
from ..classes.statistics.Distribution import Distribution
from ..classes.statistics.Histogram1D import Histogram1D
from ..base.customFunctions import expReal as mExp
from scipy import sparse
import numpy as np
from .Inference1D import Inference1D
from ..base.MPI import print
import matplotlib.pyplot as plt


def infer(userParameters, DataPoint, prng, LineResults=None, rank=1):
    """ Markov Chain Monte Carlo approach for inversion of geophysical data
    userParameters: User input parameters object
    DataPoint: Datapoint to invert
    ID: Datapoint label for saving results
    pHDFfile: Optional HDF5 file opened using h5py.File('name.h5','w',driver='mpio', comm=world) before calling Inv_MCMC
    """

    # Check the user input parameters against the datapoint
    userParameters.check(DataPoint)

    # Initialize the MCMC parameters and perform the initial iteration
    [userParameters, Mod, DataPoint, prior, likelihood, posterior, PhiD] = Initialize(userParameters, DataPoint, prng=prng)

    Res = Inference1D(DataPoint, Mod,
                save = userParameters.save,
                plot = userParameters.plot,
                savePNG = userParameters.savePNG,
                fiducial = DataPoint.fiducial,
                nMarkovChains = userParameters.nMarkovChains,
                plotEvery = userParameters.plotEvery,
                reciprocateParameters = userParameters.reciprocateParameters,
                verbose=userParameters.verbose)

    if Res.plotMe:
        Res.initFigure()
        plt.show(block=False)

    # Set the saved best models and data
    bestModel = Mod.deepcopy()
    bestData = DataPoint.deepcopy()
    bestPosterior = -np.inf #posterior#.copy()

    # Initialize the Chain
    i = 0
    iBest = 0
    multiplier = 1.0

    if userParameters.ignoreLikelihood:
        Res.burnedIn = True
        Res.iBurn = 0


    Res.clk.start()

    Go = True
    failed = False
    while (Go):

        # Accept or reject the new model
        [Mod, DataPoint, prior, likelihood, posterior, PhiD, posteriorComponents, ratioComponents, accepted, dimensionChange] = accept_reject(userParameters, Mod, DataPoint, prior, likelihood, posterior, PhiD, Res, prng)# ,oF, oD, oRel, oAdd, oP, oA, i)

        # Determine if we are burning in
        if (not Res.burnedIn):
            if (PhiD <= multiplier * DataPoint.data.size):
                Res.burnedIn = True  # Let the results know they are burned in
                Res.iBurn = i         # Save the burn in iteration to the results
                iBest = i
                bestModel = Mod.deepcopy()
                bestData = DataPoint.deepcopy()
                bestPosterior = posterior

        if (posterior > bestPosterior):
            iBest = i
            bestModel = Mod.deepcopy()
            bestData = DataPoint.deepcopy()
            bestPosterior = posterior

        if (np.mod(i, userParameters.plotEvery) == 0):
            tPerMod = Res.clk.lap() / userParameters.plotEvery
            tmp = "i=%i, k=%i, %4.3f s/Model, %0.3f s Elapsed\n" % (i, np.float(Mod.nCells[0]), tPerMod, Res.clk.timeinSeconds())
            if (rank == 1):
                print(tmp)

            if (not Res.burnedIn and not userParameters.solveRelativeError):
                multiplier *= userParameters.multiplier

        Res.update(i, Mod, DataPoint, iBest, bestData, bestModel, multiplier, PhiD, posterior, posteriorComponents, ratioComponents, accepted, dimensionChange, userParameters.clipRatio)

        if Res.plotMe:
            Res.plot("Fiducial {}".format(DataPoint.fiducial), increment=userParameters.plotEvery)

        i += 1

        Go = i <= userParameters.nMarkovChains + Res.iBurn

        if not Res.burnedIn:
            Go = i < userParameters.nMarkovChains
            if not Go:
                failed = True

    Res.clk.stop()
    Res.invTime = np.float64(Res.clk.timeinSeconds())
    # Does the user want to save the HDF5 results?
    if (userParameters.save):
        # No parallel write is being used, so write a single file for the data point
        if (LineResults is None):
            Res.save(outdir=userParameters.dataPointResultsDir, fiducial=DataPoint.fiducial)
        else: # Write the contents to the parallel HDF5 file
            LineResults.results2Hdf(Res)
#            Res.writeHdf(pHDFfile, str(ID), create=False) # Assumes space has been created for the data point

    # Does the user want to save the plot as a png?
    if (Res.savePNG):# and not failed):
        # To save any thing the Results must be plot
        Res.plot()
        Res.toPNG('.', DataPoint.fiducial)

    return failed


def initialize(userParameters, DataPoint, prng=None):
    """Initialize the transdimensional Markov chain Monte Carlo inversion.


    """
    # ---------------------------------------
    # Set the statistical properties of the datapoint
    # ---------------------------------------
    # Set the prior on the data
    DataPoint.predictedData.setPrior('MvLogNormal', DataPoint.data[DataPoint.active], DataPoint.std[DataPoint.active]**2.0, linearSpace=False, prng=prng)
    DataPoint.relErr = userParameters.initialRelativeError
    DataPoint.addErr = userParameters.initialAdditiveError

    # Define prior, proposal, posterior for height
    heightPrior = None
    heightProposal = None
    if userParameters.solveHeight:
        z = np.float64(DataPoint.z)
        dz = userParameters.maximumElevationChange
        heightPrior = Distribution('Uniform', z - dz, z + dz, prng=prng)
        heightProposal = Distribution('Normal', DataPoint.z, userParameters.elevationProposalVariance, prng=prng)

    # Define prior, proposal, posterior for relative error
    relativePrior = None
    relativeProposal = None
    if userParameters.solveRelativeError:
        relativePrior = Distribution('Uniform', userParameters.minimumRelativeError, userParameters.maximumRelativeError, prng=prng)
        relativeProposal = Distribution('MvNormal', DataPoint.relErr, userParameters.relativeErrorProposalVariance, prng=prng)

    # Define prior, proposal, posterior for additive error
    additivePrior = None
    additiveProposal = None
    if userParameters.solveAdditiveError:
        log = isinstance(DataPoint, TdemDataPoint)
        additivePrior = Distribution('Uniform', userParameters.minimumAdditiveError, userParameters.maximumAdditiveError, log=log, prng=prng)
        additiveProposal = Distribution('MvLogNormal', DataPoint.addErr, userParameters.additiveErrorProposalVariance, linearSpace=log, prng=prng)


    # Set the priors, proposals, and posteriors.
    DataPoint.setPriors(heightPrior=heightPrior, relativeErrorPrior=relativePrior, additiveErrorPrior=additivePrior)
    DataPoint.setProposals(heightProposal=heightProposal, relativeErrorProposal=relativeProposal, additiveErrorProposal=additiveProposal)
    DataPoint.setPosteriors()

    # Update the data errors based on user given parameters
    # if userParameters.solveRelativeError or userParameters.solveAdditiveError:
    DataPoint.updateErrors(userParameters.initialRelativeError, userParameters.initialAdditiveError)


    # # Initialize the calibration parameters
    # if (userParameters.solveCalibration):
    #     DataPoint.calibration.setPrior('Normal',
    #                            np.reshape(userParameters.calMean, np.size(userParameters.calMean), order='F'),
    #                            np.reshape(userParameters.calVar, np.size(userParameters.calVar), order='F'), prng=prng)
    #     DataPoint.calibration[:] = DataPoint.calibration.prior.mean
    #     # Initialize the calibration proposal
    #     DataPoint.calibration.setProposal('Normal', DataPoint.calibration, np.reshape(userParameters.propCal, np.size(userParameters.propCal), order='F'), prng=prng)

    # ---------------------------------
    # Set the earth model properties
    # ---------------------------------

    # Find the conductivity of a half space model that best fits the data
    halfspace = DataPoint.FindBestHalfSpace()

    # Create an initial model for the first iteration of the inversion
    # Initialize a 1D model with the half space conductivity
    # parameter = StatArray.StatArray(np.full(2, halfspaceValue), name='Conductivity', units=r'$\frac{S}{m}$')
    # Assign the depth to the interface as half the bounds
    Mod = halfspace.insertLayer(z = 0.5 * (userParameters.maximumDepth + userParameters.minimumDepth))

    # thk = np.asarray([0.5 * (userParameters.maximumDepth + userParameters.minimumDepth)])
    # Mod = Model1D(2, parameters = parameter, thickness=thk)

    # Setup the model for perturbation
    Mod.setPriors(halfspace.par[0], userParameters.minimumDepth, userParameters.maximumDepth, userParameters.maximumNumberofLayers, userParameters.solveParameter, userParameters.solveGradient, parameterLimits=userParameters.parameterLimits, minThickness=userParameters.minimumThickness, factor=userParameters.factor, prng=prng)


    # Assign a Hitmap as a prior if one is given
    # if (not userParameters.referenceHitmap is None):
    #     Mod.setReferenceHitmap(userParameters.referenceHitmap)

    # Compute the predicted data
    DataPoint.forward(Mod)


    if userParameters.ignoreLikelihood:
        inverseHessian = Mod.localParameterVariance()
    else:
        inverseHessian = Mod.localParameterVariance(DataPoint)



    # Instantiate the proposal for the parameters.
    parameterProposal = Distribution('MvLogNormal', Mod.par, inverseHessian, linearSpace=True, prng=prng)

    probabilities = [userParameters.pBirth, userParameters.pDeath, userParameters.pPerturb, userParameters.pNochange]
    Mod.setProposals(probabilities, parameterProposal=parameterProposal, prng=prng)

    Mod.setPosteriors()

    # Compute the data misfit
    PhiD = DataPoint.dataMisfit(squared=True)

    # Calibrate the response if it is being solved for
    if (userParameters.solveCalibration):
        DataPoint.calibrate()

    # Evaluate the prior for the current model
    p = Mod.priorProbability(userParameters.solveParameter, userParameters.solveGradient)
    prior = p
    # Evaluate the prior for the current data
    p = DataPoint.priorProbability(userParameters.solveRelativeError, userParameters.solveAdditiveError, userParameters.solveHeight, userParameters.solveCalibration)
    prior += p

    # Add the likelihood function to the prior
    likelihood = 1.0
    if not userParameters.ignoreLikelihood:
        likelihood = DataPoint.likelihood(log=True)

    posterior = likelihood + prior

    return (userParameters, Mod, DataPoint, prior, likelihood, posterior, PhiD)


def accept_reject(userParameters, Mod, DataPoint, prior, likelihood, posterior, PhiD, Res, prng):# ,oF, oD, oRel, oAdd, oP, oA ,curIter):
    """ Propose a new random model and accept or reject it """
    clk = Stopwatch()
    clk.start()

    perturbedDatapoint = DataPoint.deepcopy()

    # Perturb the current model
    if userParameters.ignoreLikelihood:
        remappedModel, perturbedModel = Mod.perturb()
    else:
        remappedModel, perturbedModel = Mod.perturb(perturbedDatapoint)

    # Propose a new data point, using assigned proposal distributions
    perturbedDatapoint.perturb(userParameters.solveHeight, userParameters.solveRelativeError, userParameters.solveAdditiveError, userParameters.solveCalibration)

    # Forward model the data from the candidate model
    perturbedDatapoint.forward(perturbedModel)

    # Compute the data misfit
    PhiD1 = perturbedDatapoint.dataMisfit(squared=True)

    if (userParameters.verbose):
        posteriorComponents = np.zeros(8, dtype=np.float64)
        prior1, posteriorComponents[:4] = perturbedModel.priorProbability(userParameters.solveParameter, userParameters.solveGradient, verbose=True)

        tmp, posteriorComponents[4:] = perturbedDatapoint.priorProbability(userParameters.solveRelativeError, userParameters.solveAdditiveError, userParameters.solveHeight, userParameters.solveCalibration, verbose=True)
        prior1 += tmp

    else:

        # Evaluate the prior for the current model
        posteriorComponents = None
        # Evaluate the prior for the current model
        prior1 = perturbedModel.priorProbability(userParameters.solveParameter, userParameters.solveGradient)
        # Evaluate the prior for the current data
        prior1 += perturbedDatapoint.priorProbability(userParameters.solveRelativeError, userParameters.solveAdditiveError, userParameters.solveHeight, userParameters.solveCalibration)

    # Test for early rejection
    if (prior1 == -np.inf):
        return(Mod, DataPoint, prior, likelihood, posterior, PhiD, posteriorComponents, ratioComponents, False, Mod.nCells[0] != perturbedModel.nCells[0])

    # Compute the components of each acceptance ratio
    likelihood1 = 1.0
    if not userParameters.ignoreLikelihood:
        likelihood1 = perturbedDatapoint.likelihood(log=True)
        proposal, proposal1 = perturbedModel.proposalProbabilities(remappedModel, perturbedDatapoint)
    else:
        proposal, proposal1 = perturbedModel.proposalProbabilities(remappedModel)

    posterior1 = prior1 + likelihood1


    priorRatio = prior1 - prior

    likelihoodRatio = likelihood1 - likelihood

    proposalRatio = proposal - proposal1


    try:
        log_acceptanceRatio = np.float128(priorRatio + likelihoodRatio + proposalRatio)

        acceptanceProbability = mExp(log_acceptanceRatio)
    except:
        log_acceptanceRatio = -np.inf
        acceptanceProbability = -1.0

    if userParameters.verbose:
        ratioComponents = np.squeeze(np.asarray([prior1, prior, likelihood1, likelihood, proposal, proposal1, log_acceptanceRatio]))
    else:
        ratioComponents = None

    # If we accept the model
    accepted = acceptanceProbability > prng.uniform()

    if (accepted):
        Res.acceptance += 1
        return(perturbedModel, perturbedDatapoint, prior1, likelihood1, posterior1, PhiD1, posteriorComponents, ratioComponents, True, Mod.nCells[0] != perturbedModel.nCells[0])

    else: # Rejected
        return(Mod, DataPoint, prior, likelihood, posterior, PhiD, posteriorComponents, ratioComponents, False, Mod.nCells[0] != perturbedModel.nCells[0])

    clk.stop()


    #%%


# %%
