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
from .Results import Results
from ..base.MPI import print
import matplotlib.pyplot as plt

def Inv_MCMC(userParameters, DataPoint, prng, LineResults=None, rank=1):
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

    Res = Results(DataPoint, Mod,
                save = userParameters.save,
                plot = userParameters.plot,
                savePNG = userParameters.savePNG,
                fiducial = DataPoint.fiducial,
                nMarkovChains = userParameters.nMarkovChains,
                plotEvery = userParameters.plotEvery,
                parameterDisplayLimits = userParameters.parameterDisplayLimits,
                reciprocateParameters = userParameters.reciprocateParameters,
                priMu = userParameters.priMu,
                priStd = userParameters.priStd,
                verbose=userParameters.verbose)

    # Set the saved best models and data
    bestModel = Mod.deepcopy()
    bestData = DataPoint.deepcopy()
    bestPosterior = posterior#.copy()

    # Initialize the Chain
    iBurn = 0
    i = 0
    iBest = 1
    multiplier = 1.0

    if userParameters.ignoreLikelihood:
        Res.burnedIn = True
        Res.iBurn = 0

    Res.clk.start()

    Go = True
    failed = False
    while (Go):

        # Accept or reject the new model
        [Mod, DataPoint, prior, likelihood, posterior, PhiD, posteriorComponents, ratioComponents, accepted, dimensionChange] = AcceptReject(userParameters, Mod, DataPoint, prior, likelihood, posterior, PhiD, Res, prng)# ,oF, oD, oRel, oAdd, oP, oA, i)
        
        # Determine if we are burning in
        if (not Res.burnedIn):
            if (PhiD <= multiplier * DataPoint.data.size):
                Res.burnedIn = True  # Let the results know they are burned in
                Res.iBurn = i         # Save the burn in iteration to the results
                bestModel = Mod.deepcopy()
                bestData = DataPoint.deepcopy()
                bestPosterior = posterior.copy()
            

        # Update the best best model and data if the posterior is larger
        if (posterior > bestPosterior and Res.burnedIn):
            iBest = np.int64(i)
            bestModel = Mod.deepcopy()
            bestData = DataPoint.deepcopy()
            bestPosterior = posterior.copy()

        Res.iBestV[i] = iBest

        if (np.mod(i, userParameters.plotEvery) == 0):
            tPerMod = Res.clk.lap() / userParameters.plotEvery
            tmp = "i=%i, k=%i, %4.3f s/Model, %0.3f s Elapsed\n" % (i, np.float(Mod.nCells[0]), tPerMod, Res.clk.timeinSeconds())
            if (rank == 1):
                print(tmp)

            if (not Res.burnedIn and not userParameters.solveRelativeError):
                multiplier *= userParameters.multiplier

        Res.update(i, iBest, bestData, bestModel, DataPoint, multiplier, PhiD, Mod, posterior, posteriorComponents, ratioComponents, accepted, dimensionChange, userParameters.clipRatio)

        Res.plot("Fiducial {}".format(DataPoint.fiducial))

        i += 1
        
        Go = i <= userParameters.nMarkovChains + Res.iBurn
        if failed:
            Go = False

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
    Res.plot()

    # Does the user want to save the plot as a png?
    if (Res.savePNG):# and not failed):
        # To save any thing the Results must be plot
        Res.plot(forcePlot=True)
        Res.toPNG('.', DataPoint.fiducial)

    return failed

   
def Initialize(userParameters, DataPoint, prng):
    """Initialize the transdimensional Markov chain Monte Carlo inversion.
    
    
    """
    # ---------------------------------------
    # Set the distribution of the data misfit
    # ---------------------------------------
    # Incoming standard deviations may be zero. The variance of the prior is updated
    # later with DataPoint.updateErrors.
    DataPoint._predictedData.setPrior('MvNormalLog', DataPoint._data[DataPoint.iActive], DataPoint._std[DataPoint.iActive]**2.0, prng=prng)

    # ------------------------------------
    # Set the data point height properties
    # ------------------------------------
    # Set the prior on the height
    DataPoint.z.setPrior('UniformLog', np.float64(DataPoint.z) - userParameters.maximumElevationChange, np.float64(DataPoint.z) + userParameters.maximumElevationChange)
    # Set the proposal for height
    DataPoint.z.setProposal('Normal', DataPoint.z, userParameters.elevationProposalVariance, prng=prng)
    # Create a histogram to set the height posterior.
    H = Histogram1D(bins = StatArray.StatArray(DataPoint.z.prior.getBinEdges(), name=DataPoint.z.name, units=DataPoint.z.units), relativeTo=DataPoint.z)
    DataPoint.z.setPosterior(H)

    # ---------------------------------
    # Set the relative error properties
    # ---------------------------------
    # Set the prior on the relative Errors
    DataPoint.relErr[:] = userParameters.initialRelativeError.deepcopy()
    DataPoint.setRelativeErrorPrior(userParameters.minimumRelativeError[:], userParameters.maximumRelativeError[:], prng=prng)

    # Set the proposal distribution for the relative errors
    DataPoint.setRelativeErrorProposal(userParameters.initialRelativeError, userParameters.relativeErrorProposalVariance, prng=prng)

    # Initialize the histograms for the relative errors
    rBins = DataPoint.relErr.prior.getBinEdges()
    if DataPoint.nSystems > 1:
        DataPoint.relErr.setPosterior([Histogram1D(bins = StatArray.StatArray(rBins[0, :], name='$\epsilon_{Relative}x10^{2}$', units='%')) for i in range(DataPoint.nSystems)])
    else:
        DataPoint.relErr.setPosterior(Histogram1D(bins = StatArray.StatArray(rBins, name='$\epsilon_{Relative}x10^{2}$', units='%')))

    # ---------------------------------
    # Set the additive error properties
    # ---------------------------------

    # Set the prior on the additive Errors
    DataPoint.addErr[:] = userParameters.initialAdditiveError.deepcopy()
    DataPoint.setAdditiveErrorPrior(userParameters.minimumAdditiveError[:], userParameters.maximumAdditiveError[:], prng=prng)

    # Set the proposal distribution for the additive errors
    DataPoint.setAdditiveErrorProposal(userParameters.initialAdditiveError, userParameters.additiveErrorProposalVariance, prng=prng)

    # Set the posterior for the data point.
    DataPoint.setAdditiveErrorPosterior()

    # Update the data errors based on user given parameters
    if userParameters.solveRelativeError or userParameters.solveAdditiveError:
        DataPoint.updateErrors(userParameters.initialRelativeError, userParameters.initialAdditiveError)

    DataPoint.addErr.updatePosterior()

    # Save a copy of the original errors
    userParameters.Err = DataPoint._std.deepcopy()

    # Initialize the calibration parameters
    if (userParameters.solveCalibration):
        DataPoint.calibration.setPrior('NormalLog',
                               np.reshape(userParameters.calMean, np.size(userParameters.calMean), order='F'),
                               np.reshape(userParameters.calVar, np.size(userParameters.calVar), order='F'), prng=prng)
        DataPoint.calibration[:] = DataPoint.calibration.prior.mean
        # Initialize the calibration proposal
        DataPoint.calibration.setProposal('Normal', DataPoint.calibration, np.reshape(userParameters.propCal, np.size(userParameters.propCal), order='F'), prng=prng)

    # ---------------------------------
    # Set the earth model properties
    # ---------------------------------

    # Find the conductivity of a half space model that best fits the data
    halfspaceValue = DataPoint.FindBestHalfSpace()

    # Create an initial model for the first iteration of the inversion
    # Initialize a 1D model with the half space conductivity
    parameter = StatArray.StatArray(np.full(2, halfspaceValue), name='Conductivity', units=r'$\frac{S}{m}$')
    # Assign the depth to the interface as half the bounds
    thk = np.asarray([0.5 * (userParameters.maximumDepth + userParameters.minimumDepth)])
    Mod = Model1D(2, parameters = parameter, thickness=thk)

    # Setup the model for perturbation
    pWheel = [userParameters.pBirth, userParameters.pDeath, userParameters.pPerturb, userParameters.pNochange]
    Mod.setPriors(halfspaceValue, pWheel, userParameters.minimumDepth, userParameters.maximumDepth, userParameters.maximumNumberofLayers, minThickness=userParameters.minimumThickness, prng=prng, factor=userParameters.factor)

    # Assign a Hitmap as a prior if one is given
    # if (not userParameters.referenceHitmap is None):
    #     Mod.setReferenceHitmap(userParameters.referenceHitmap)

    userParameters.priMu = Mod.par.prior.mean
    userParameters.priStd = np.sqrt(Mod.par.prior.variance)

    Mod.setPosteriors()

    userParameters.pLimits = None
    if userParameters.LimitPar:
        userParameters.pLimits = np.exp(Mod.par.prior.getBinEdges(nBins = 1, nStd = 4.0))

    # Compute the predicted data
    DataPoint.forward(Mod)
    # Compute the sensitivity wrt parameter
    DataPoint.J = DataPoint.sensitivity(Mod)

    if (userParameters.stochasticNewton):
        # Scale the sensitivity matrix by the data errors.
        J = DataPoint.scaleJ(DataPoint.J)
        # Compute a quasi-Newton based variance update
        userParameters.unscaledVariance = np.linalg.inv(np.dot(J.T, J) + np.eye(Mod.nCells[0]) * userParameters.priStd**-1.0)
    else:
        # Compute a steepest descent based variance update
        userParameters.unscaledVariance = (np.ones(Mod.nCells[0]) * (userParameters.priStd))**2.0

    # Instantiate the proposal for the parameters.
    Mod.par.setProposal('MvNormal', np.log(Mod.par), userParameters.unscaledVariance, prng=prng)

    # Assign a prior to the derivative of the model
    Mod.dpar.setPrior('MvNormalLog', 0.0, userParameters.gradientStd**2.0, prng=prng)

    # Compute the data misfit
    PhiD = DataPoint.dataMisfit(squared=True)

    # Calibrate the response if it is being solved for
    if (userParameters.solveCalibration):
        DataPoint.calibrate()

    # Evaluate the prior for the current model
    p = Mod.priorProbability(userParameters.solveParameter, userParameters.solveGradient, userParameters.pLimits)
    prior = p
    # Evaluate the prior for the current data
    p = DataPoint.priorProbability(userParameters.solveRelativeError, userParameters.solveAdditiveError, userParameters.solveElevation, userParameters.solveCalibration)
    prior += p

    # Add the likelihood function to the prior
    likelihood = 1.0
    if not userParameters.ignoreLikelihood:
        likelihood = DataPoint.likelihood()

    # print('likelhiood {}'.format(likelihood))
    posterior = likelihood + prior

    return (userParameters, Mod, DataPoint, prior, likelihood, posterior, PhiD)


def AcceptReject(userParameters, Mod, DataPoint, prior, likelihood, posterior, PhiD, Res, prng):# ,oF, oD, oRel, oAdd, oP, oA ,curIter):
    """ Propose a new random model and accept or reject it """
    clk = Stopwatch()
    clk.start()

    # Perturb the current model to produce an initial candidate model
    Mod1, option, value = Mod.perturb()

    parSaved = Mod1.par.deepcopy()
    
    # Propose a new data point, using assigned proposal distributions
    D1 = DataPoint.propose(userParameters.solveElevation, userParameters.solveRelativeError, userParameters.solveAdditiveError, userParameters.solveCalibration)

    if (option < 2):
        # Compute the sensitivity of the data to the perturbed model
        D1.J = D1.updateSensitivity(D1.J, Mod1, option, scale=False)
        J = DataPoint.scaleJ(D1.J)

        # Propose new layer conductivities
        if userParameters.stochasticNewton:
            unscaledVariance = np.linalg.inv(np.dot(J.T,J) + np.eye(Mod1.nCells[0]) * userParameters.priStd**-1.0)
            J = DataPoint.scaleJ(D1.J, 2.0)
        else:

            unscaledVariance = np.diag((userParameters.covScaling / np.sqrt(Mod1.nCells[0])) / (
                (np.dot(J.T, J)) + np.eye(Mod1.nCells[0]) * (userParameters.priStd**-1.0)))
    else:  # There was no change in the model
        # Normalize the saved sensitivity matrix by the previous data errors
        if (userParameters.stochasticNewton):
            J = DataPoint.scaleJ(D1.J, 2.0)

        unscaledVariance = userParameters.unscaledVariance

    # If we are using the stochastic Newton step, compute the gradient of the
    # objective function
    if (userParameters.stochasticNewton):
        # Compute the gradient
        gradient = np.dot(J.T, DataPoint.deltaD[DataPoint.iActive]) + \
            ((userParameters.priStd**-1.0) * (np.log(Mod1.par) - userParameters.priMu))

        scaling = userParameters.covScaling * \
            ((2.0 * np.float64(Mod1.nCells[0])) - 1)**(-1.0 / 3.0)
        # Compute the Model perturbation
        dm = 0.5 * scaling * np.dot(unscaledVariance, gradient)

        if (not Res.burnedIn):
            dm = 0.0

        Mod1.par.setProposal('MvNormal', np.log(Mod1.par) - dm, scaling * unscaledVariance, prng=Mod1.par.proposal.prng)
    else:  # Use the steepest descent method
        Mod1.par.setProposal('MvNormal', np.log(Mod1.par), unscaledVariance, prng=Mod1.par.proposal.prng)

    # Generate new conductivities
    Mod1.par[:] = np.exp(Mod1.par.proposal.rng(1))

    # Forward model the data from the candidate model
    D1.forward(Mod1)

    # Update the data errors using the updated relative errors
    if userParameters.solveRelativeError or userParameters.solveAdditiveError:
        D1.updateErrors(D1.relErr, D1.addErr)

    # Calibrate the response if it is being solved for
    if (userParameters.solveCalibration):
        D1.calibrate()

    # Compute the data misfit
    PhiD1 = D1.dataMisfit(squared=True)

    
    if (userParameters.verbose):
        posteriorComponents = np.zeros(8, dtype=np.float64)
        prior1, posteriorComponents[:4] = Mod1.priorProbability(userParameters.solveParameter, userParameters.solveGradient, userParameters.pLimits, verbose=True)

        tmp, posteriorComponents[4:] = D1.priorProbability(userParameters.solveRelativeError, userParameters.solveAdditiveError, userParameters.solveElevation, userParameters.solveCalibration, verbose=True)
        prior1 += tmp

        likelihood1 = D1.likelihood()

        posterior1 = prior1 + likelihood1


    else:

        # Evaluate the prior for the current model
        posteriorComponents = None
        # Evaluate the prior for the current model
        p = Mod1.priorProbability(userParameters.solveParameter, userParameters.solveGradient, userParameters.pLimits)
        # print('model prior {}'.format(p))
        prior1 = p
        # Evaluate the prior for the current data
        p = D1.priorProbability(userParameters.solveRelativeError, userParameters.solveAdditiveError, userParameters.solveElevation, userParameters.solveCalibration)
        # print('data prior {}'.format(p))
        prior1 += p

        likelihood = 1.0
        if not userParameters.ignoreLikelihood:
            likelihood = D1.likelihood()

        # print('likeliohood {}'.format(likelihood))
        # Add the likelihood function to the prior
        posterior1 = likelihood + prior1

    # Exchange the mean of candidate models proposal mean with the
    # pre-perturbed values (maintain the diagonal variance)

    if (userParameters.stochasticNewton):
        Mod1.par.setProposal('MvNormalLog', np.log(parSaved) - dm, Mod1.par.proposal.variance, prng=Mod1.par.proposal.prng)
    else:
        Mod1.par.setProposal('MvNormalLog', np.log(parSaved), Mod1.par.proposal.variance, prng=Mod1.par.proposal.prng)

    if (userParameters.stochasticNewton):
        # Get the pdf for the perturbed parameters
        prop1 = Mod1.par.proposal.probability(np.log(Mod1.par))  # CAN.prop

        J = D1.scaleJ(D1.J, power=2.0)
        # Compute the gradient "uphill" back towards the previous model
        gradient = np.dot(J.T, D1.deltaD[D1.iActive]) + \
            userParameters.priStd**-1.0 * (np.log(Mod1.par) - userParameters.priMu)

        # Compute the Model perturbation
        dm = 0.5 * scaling * np.dot(unscaledVariance, gradient)

        if (not Res.burnedIn):
            dm = 0.0
        tmp = Distribution('MvNormalLog', np.log(Mod1.par) - dm, scaling * unscaledVariance, prng=Mod1.par.proposal.prng)

        prop = tmp.probability(np.log(parSaved))  # CUR.prop
    else:
        # Get the pdf for the perturbed parameters
        prop1 = Mod1.par.proposal.probability(np.log(Mod1.par))  # CAN.prop

        par = StatArray.StatArray(np.log(Mod1.par))
        cov = StatArray.StatArray(Mod1.par.proposal.variance)
  
        if (Mod1.nCells > Mod.nCells):  # Layer was inserted
            tmp = np.mean(par[Mod1.perturbedLayer:Mod1.perturbedLayer + 2])
            par = par.delete(Mod1.perturbedLayer)
            par[Mod1.perturbedLayer] = tmp
            cov = cov.delete(Mod1.perturbedLayer)

        elif (Mod1.nCells < Mod.nCells):  # Layer was deleted
            tmp = par[Mod1.perturbedLayer]
            par = par.insert(Mod1.perturbedLayer, tmp)
            tmp2 = cov[Mod1.perturbedLayer]
            cov = cov.insert(Mod1.perturbedLayer, tmp2)

        tmp = Mod.deepcopy()
        tmp.par.setProposal('MvNormalLog', par, cov)
        prop = tmp.par.proposal.probability(np.log(Mod.par))  # CUR.prop

    posteriorRatio = posterior1 - posterior
    proposalRatio = prop - prop1

    P_depth  = np.log(Mod.depth.probability(Mod.nCells[0]-1))
    P_depth1 = np.log(Mod1.depth.probability(Mod1.nCells[0]-1))

    # TEMPORARY TRY EXCEPT UNTIL I FIGURE OUT THE PROBLEM
    acceptanceRatio = 0.0
    try:
        tmp = np.float128(posteriorRatio + proposalRatio + (P_depth - P_depth1))
        acceptanceRatio = mExp(tmp)
        failed = False
    except:
        failed = True
        acceptanceRatio = 0.0
    if userParameters.verbose:
        ratioComponents = np.asarray([prior1, prior, likelihood1, likelihood, proposal, proposal1, log_acceptanceRatio])
        # print('ratio comp: {}'.format(rat
        # 
        # ioComponents))
    else:
        ratioComponents = None
        
    # acceptanceProbability = np.minimum(1.0, acceptanceRatio)

    # If we accept the model
    r = prng.uniform()

    if (acceptanceProbability > r):
        userParameters.unscaledVariance = unscaledVariance
        Res.acceptance += 1
        return(Mod1, D1, prior1, likelihood1, posterior1, PhiD1, posteriorComponents, ratioComponents, True, Mod.nCells[0] != Mod1.nCells[0])

    else:
        return(Mod, DataPoint, prior, likelihood, posterior, PhiD, posteriorComponents, ratioComponents, False, Mod.nCells[0] != Mod1.nCells[0])

    clk.stop()

    
    #%%
