""" @EMinversion1D_MCMC
Module defining a Markov Chain Monte Carlo approach to 1D EM inversion
"""
#%%
from ..classes.core.Stopwatch import Stopwatch
from ..classes.data.dataset.FdemData import FdemData
from ..classes.data.dataset.TdemData import TdemData
from ..classes.model.Model1D import Model1D
from ..classes.core.StatArray import StatArray
from ..classes.statistics.Distribution import Distribution
from ..base.customFunctions import expReal as mExp
from scipy import sparse
import numpy as np
from .Results import Results
from ..base.MPI import print
import matplotlib.pyplot as plt

def Inv_MCMC(paras, DataPoint, prng, LineResults=None, rank=1):
    """ Markov Chain Monte Carlo approach for inversion of geophysical data
    paras: User input parameters object
    DataPoint: Datapoint to invert
    ID: Datapoint label for saving results
    pHDFfile: Optional HDF5 file opened using h5py.File('name.h5','w',driver='mpio', comm=world) before calling Inv_MCMC
    """
    #%%
    # Check the user input parameters against the datapoint
    paras.check(DataPoint)

    # Initialize the MCMC parameters and perform the initial iteration
    [paras, Mod, DataPoint, prior, posterior, PhiD] = Initialize(paras, DataPoint, prng=prng)

    Res = Results(DataPoint, Mod,
                save = paras.save,
                plot = paras.plot,
                savePNG = paras.savePNG,
                ID = DataPoint.fiducial,
                nMarkovChains = paras.nMarkovChains,
                plotEvery = paras.plotEvery,
                parameterDisplayLimits = paras.parameterDisplayLimits,
                reciprocateParameters = paras.reciprocateParameters,
                priMu = paras.priMu,
                priStd = paras.priStd,
                verbose=paras.verbose)

    # Set the saved best models and data
    bestModel = Mod  # .deepcopy()
    bestData = DataPoint  # .deepcopy()
    bestPosterior = posterior  # .copy()

    # Initialize the Chain
    iBurn = 0
    i = 1
    iBest = 1
    multiplier = 1.0

    Res.clk.start()

    Go = i <= paras.nMarkovChains + iBurn -1
    while (Go):

        # Accept or reject the new model
        [Mod, DataPoint, prior, posterior, PhiD, posteriorComponents, time] = AcceptReject(paras, Mod, DataPoint, prior, posterior, PhiD, Res, prng)# ,oF, oD, oRel, oAdd, oP, oA, i)

        # Determine if we are burning in
        if (not Res.burnedIn):
            if (PhiD <= multiplier * DataPoint.data.size):
                Res.burnedIn = True  # Let the results know they are burned in
                Res.iBurn = i         # Save the burn in iteration to the results

        # Update the best best model and data if the posterior is larger
        if (posterior > bestPosterior):
            iBest = np.int64(i)
            bestModel = Mod  # .deepcopy()
            bestData = DataPoint  # .deepcopy()
            bestPosterior = posterior  # .copy()

        Res.iBestV[i] = iBest

        if (np.mod(i, paras.plotEvery) == 0):
            tPerMod = Res.clk.lap() / paras.plotEvery
            tmp = "i=%i, k=%i, %4.3f s/Model, %0.3f s Elapsed\n" % (i, np.float(Mod.nCells[0]), tPerMod, Res.clk.timeinSeconds())
            if (rank == 1):
                print(tmp)

            if (not Res.burnedIn and not paras.solveRelativeError):
                multiplier *= paras.multiplier

        Res.update(i, iBest, bestData, bestModel, DataPoint, multiplier, PhiD, Mod, posterior, posteriorComponents, paras.clipRatio)
        
        Res.plot()   
                 
        i += 1
        
        Go = i <= paras.nMarkovChains + iBurn -1

    Res.clk.stop()
    Res.invTime = np.float64(Res.clk.timeinSeconds())
    # Does the user want to save the HDF5 results?
    if (paras.save):
        # No parallel write is being used, so write a single file for the data point
        if (LineResults is None):
            Res.save(outdir=paras.dataPointResultsDir, ID=DataPoint.fiducial)
        else: # Write the contents to the parallel HDF5 file
            LineResults.results2Hdf(Res)
#            Res.writeHdf(pHDFfile, str(ID), create=False) # Assumes space has been created for the data point
    Res.plot()

    # Does the user want to save the plot as a png?
    if (Res.savePNG):# and not failed):
        # To save any thing the Results must be plot
        Res.plot(forcePlot=True)
        Res.toPNG('.', DataPoint.fiducial)

   
def Initialize(paras, DataPoint, prng):
    np.set_printoptions(threshold=np.inf)
    """ Initialize variables and priors, and perform the first iteration """
    # Initialize properties of the data
    # Set the distribution of the data misfit
    # Incoming standard deviations may be zero. The variance of the prior is updated
    # later with DataPoint.updateErrors.
    DataPoint._predictedData.setPrior('MvNormalLog', DataPoint._data[DataPoint.iActive], DataPoint._std[DataPoint.iActive]**2.0, prng=prng)

    # Set the prior on the elevation height
    DataPoint.z.setPrior('UniformLog', np.float64(DataPoint.z) - paras.maximumElevationChange, np.float64(DataPoint.z) + paras.maximumElevationChange)
    # DataPoint.z.setPrior('NormalLog', DataPoint.z, 1.0, prng=prng)
    # DataPoint.z.setPrior('Normal',DataPoint.z,paras.zRange)

    DataPoint.z.setProposal('Normal', DataPoint.z, paras.elevationProposalVariance, prng=prng)

    # Set the prior on the relative Errors
    DataPoint.relErr[:] = paras.initialRelativeError.deepcopy()
    DataPoint.setRelativeErrorPrior(paras.minimumRelativeError[:], paras.maximumRelativeError[:], prng=prng)
    # Set the prior on the additive Errors
    DataPoint.addErr[:] = paras.initialAdditiveError.deepcopy()
    DataPoint.setAdditiveErrorPrior(paras.minimumAdditiveError[:], paras.maximumAdditiveError[:], prng=prng)

    # Update the data errors based on user given parameters
    DataPoint.updateErrors(paras.initialRelativeError, paras.initialAdditiveError)

    # Save a copy of the original errors
    paras.Err = DataPoint._std.deepcopy()
    # Set the proposal distribution for the relative errors
    DataPoint.setRelativeErrorProposal(paras.initialRelativeError, paras.relativeErrorProposalVariance, prng=prng)
    
    # Set the proposal distribution for the relative errors
    DataPoint.setAdditiveErrorProposal(paras.initialAdditiveError, paras.additiveErrorProposalVariance, prng=prng)

    # Initialize the calibration parameters
    if (paras.solveCalibration):
        DataPoint.calibration.setPrior('NormalLog',
                               np.reshape(paras.calMean, np.size(paras.calMean), order='F'),
                               np.reshape(paras.calVar, np.size(paras.calVar), order='F'), prng=prng)
        DataPoint.calibration[:] = DataPoint.calibration.prior.mean
        # Initialize the calibration proposal
        DataPoint.calibration.setProposal('Normal', DataPoint.calibration, np.reshape(paras.propCal, np.size(paras.propCal), order='F'), prng=prng)

    # Find the conductivity of a half space model that best fits the data
    HScond = DataPoint.FindBestHalfSpace()

    # Create an initial model for the first iteration of the inversion
    # Initialize a 1D model with the half space conductivity
    parameter = StatArray(np.asarray([HScond, HScond]), name='Conductivity', units=r'$\frac{S}{m}$')
    # Assign the depth to the interface as half the bounds
    thk = np.asarray([0.5 * (paras.maximumDepth + paras.minimumDepth)])
    Mod = Model1D(2, parameters = parameter, thickness=thk)

    # Setup the model for perturbation
    pWheel = [paras.pBirth, paras.pDeath, paras.pPerturb, paras.pNochange]
    Mod.makePerturbable(pWheel, paras.minimumDepth, paras.maximumDepth, paras.maximumNumberofLayers, prng=prng, minThickness=paras.minimumThickness)

    # Set priors on the depth interfaces, given a number of layers
    Mod.depth.setPrior('Order', Mod.minDepth, Mod.maxDepth, Mod.minThickness, paras.maximumNumberofLayers)  # priZ

    # Compute the mean and std for the parameter
    paras.priMu = np.log(HScond)
    paras.priStd = np.log(1.0 + paras.factor)
    # Assign a normal distribution to the conductivities
    Mod.par.setPrior('MvNormalLog', paras.priMu, paras.priStd**2.0, prng=prng)

    # Assign a Hitmap as a prior if one is given
    if (not paras.referenceHitmap is None):
        Mod.setReferenceHitmap(paras.referenceHitmap)

    paras.pLimits = None
    if paras.LimitPar:
        paras.pLimits = [(np.exp(paras.priMu - 3.0 * paras.priStd)),
                         (np.exp(paras.priMu + 3.0 * paras.priStd))]

    # Compute the predicted data
    DataPoint.forward(Mod)
    # Compute the sensitivity wrt parameter
    DataPoint.J = DataPoint.sensitivity(Mod)

    if (paras.stochasticNewton):
        # Scale the sensitivity matrix by the data errors.
        J = DataPoint.scaleJ(DataPoint.J)
        # Compute a quasi-Newton based variance update
        paras.unscaledVariance = np.linalg.inv(np.dot(J.T, J) + np.eye(Mod.nCells[0]) * paras.priStd**-1.0)
    else:
        # Compute a steepest descent based variance update
        paras.unscaledVariance = (np.ones(Mod.nCells[0]) * (paras.priStd))**2.0

    # Instantiate the proposal for the parameters.
    Mod.par.setProposal('MvNormal', np.log(Mod.par), paras.unscaledVariance, prng=prng)

    # Assign a prior to the derivative of the model
    Mod.dpar.setPrior('MvNormalLog', 0.0, paras.gradientStd**2.0, prng=prng)

    # Compute the data misfit
    PhiD = DataPoint.dataMisfit(squared=True)

    # Calibrate the response if it is being solved for
    if (paras.solveCalibration):
        DataPoint.calibrate()

    # Evaluate the prior for the current model
    prior = Mod.priorProbability(paras.solveParameter,paras.solveGradient,paras.pLimits)
    # Evaluate the prior for the current data
    prior += DataPoint.priorProbability(paras.solveRelativeError, paras.solveAdditiveError, paras.solveElevation, paras.solveCalibration)

    # Add the likelihood function to the prior
    posterior = DataPoint.likelihood() + prior

    return (paras, Mod, DataPoint, prior, posterior, PhiD)


def AcceptReject(paras, Mod, DataPoint, prior, posterior, PhiD, Res, prng):# ,oF, oD, oRel, oAdd, oP, oA ,curIter):
    """ Propose a new random model and accept or reject it """
    clk = Stopwatch()
    clk.start()

    # Perturb the current model to produce an initial candidate model
    Mod1, option, value = Mod.perturb()

    parSaved = Mod1.par.deepcopy()

    # Propose a new data point, using assigned proposal distributions
    D1 = DataPoint.propose(paras.solveElevation,paras.solveRelativeError,paras.solveAdditiveError,paras.solveCalibration)

    if (option < 2):
        # Compute the sensitivity of the data to the perturbed model
        D1.J = D1.updateSensitivity(D1.J, Mod1, option, scale=False)
        J = DataPoint.scaleJ(D1.J)

        # Propose new layer conductivities
        if paras.stochasticNewton:
            unscaledVariance = np.linalg.inv(np.dot(J.T,J) + np.eye(Mod1.nCells[0]) * paras.priStd**-1.0)
            J = DataPoint.scaleJ(D1.J, 2.0)
        else:
            unscaledVariance = np.diag((paras.covScaling / np.sqrt(Mod1.nCells)) / (
                (np.dot(J.T, J)) + sparse.eye(Mod1.nCells[0]) * (paras.priStd**-1.0)))
    else:  # There was no change in the model
        # Normalize the saved sensitivity matrix by the previous data errors
        if (paras.stochasticNewton):
            J = DataPoint.scaleJ(D1.J, 2.0)

        unscaledVariance = paras.unscaledVariance

    # If we are using the stochastic Newton step, compute the gradient of the
    # objective function
    if (paras.stochasticNewton):
        # Compute the gradient
        gradient = np.dot(J.T, DataPoint.deltaD[DataPoint.iActive]) + \
            ((paras.priStd**-1.0) * (np.log(Mod1.par) - paras.priMu))

        scaling = paras.covScaling * \
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
    D1.updateErrors(D1.relErr, D1.addErr)

    # Calibrate the response if it is being solved for
    if (paras.solveCalibration):
        D1.calibrate()

    # Compute the data misfit
    PhiD1 = D1.dataMisfit(squared=True)

    # Evaluate the prior for the current model
    if (paras.verbose):
        posteriorComponents = np.zeros(9, dtype=np.float64)
        prior1, posteriorComponents[:4] = Mod1.probability(paras.solveParameter, paras.solveGradient, paras.pLimits, verbose=True)
        tmp, posteriorComponents[4:-1] = D1.probability(paras.solveRelativeError, paras.solveAdditiveError, paras.solveElevation, paras.solveCalibration, verbose=True)

        prior1 += tmp
        likelihood = D1.likelihood()

        posteriorComponents[-1] = likelihood
        posterior1 = likelihood + prior1


    else:
        posteriorComponents=None
        # Evaluate the prior for the current model
        prior1 = Mod1.priorProbability(paras.solveParameter, paras.solveGradient, paras.pLimits)
        # Evaluate the prior for the current data
        prior1 += D1.priorProbability(paras.solveRelativeError, paras.solveAdditiveError, paras.solveElevation, paras.solveCalibration)

        # Add the likelihood function to the prior
        posterior1 = D1.likelihood() + prior1

    # Exchange the mean of candidate models proposal mean with the
    # pre-perturbed values (maintain the diagonal variance)

    if (paras.stochasticNewton):
        Mod1.par.setProposal('MvNormalLog', np.log(parSaved) - dm, Mod1.par.proposal.variance, prng=Mod1.par.proposal.prng)
    else:
        Mod1.par.setProposal('MvNormalLog', np.log(parSaved), Mod1.par.proposal.variance, prng=Mod1.par.proposal.prng)

    if (paras.stochasticNewton):
        # Get the pdf for the perturbed parameters
        prop1 = Mod1.par.proposal.probability(np.log(Mod1.par))  # CAN.prop

        J = D1.scaleJ(D1.J, power=2.0)
        # Compute the gradient "uphill" back towards the previous model
        gradient = np.dot(J.T, D1.deltaD[D1.iActive]) + \
            paras.priStd**-1.0 * (np.log(Mod1.par) - paras.priMu)

        # Compute the Model perturbation
        dm = 0.5 * scaling * np.dot(unscaledVariance, gradient)

        if (not Res.burnedIn):
            dm = 0.0
        tmp = Distribution('MvNormalLog', np.log(Mod1.par) - dm, scaling * unscaledVariance, prng=Mod1.par.proposal.prng)

        prop = tmp.probability(np.log(parSaved))  # CUR.prop
    else:
        # Get the pdf for the perturbed parameters
        prop1 = Mod1.par.proposal.getPdf(np.log(Mod1.par))  # CAN.prop

        par = StatArray(np.log(Mod1.par))
        cov = StatArray(Mod1.par.proposal.variance)
  
        if (Mod1.nCells > Mod.nCells):  # Layer was inserted
            tmp = np.mean(par[Mod1.iLayer:Mod1.iLayer + 2])
            par = par.delete(Mod1.iLayer)
            par[Mod1.iLayer] = tmp
            cov = cov.delete(Mod1.iLayer)

        elif (Mod1.nCells < Mod.nCells):  # Layer was deleted
            tmp = par[Mod1.iLayer]
            par = par.insert(Mod1.iLayer, tmp)
            tmp2 = cov[Mod1.iLayer]
            cov = cov.insert(Mod1.iLayer, tmp2)

        tmp = Mod.deepcopy()
        tmp.par.setProposal('MvNormalLog', par, cov)
        prop = tmp.par.proposal.getPdf(np.log(Mod.par))  # CUR.prop

    P_depth  = np.log(Mod.depth.probability(Mod.nCells[0]))
    P_depth1 = np.log(Mod1.depth.probability(Mod1.nCells[0]))

    tmp = np.float128((posterior1 + prop) -
                      (posterior + prop1) +
                      (P_depth - P_depth1))

    likeRatio = mExp(tmp)

    if (np.isnan(likeRatio)):
        likeRatio = 0.0

    cut = np.minimum(1.0, likeRatio)

    # If we accept the model
    r = prng.uniform()

    if (cut > r):
#        accepted=True
        # Make the Current model the Candidate model
        Mod0 = Mod1  # .deepcopy()
        # Make the Current data the Candidate data
        D0 = D1  # .deepcopy()
        # Transfer over the posteriors and priors
        prior0 = prior1  # .copy()
        posterior0 = posterior1  # .copy()
        PhiD0 = PhiD1  # .copy()
        paras.unscaledVariance = unscaledVariance
        if (Res.saveMe or Res.plotMe):
            Res.acceptance += 1

    else:
#        accepted=False
        # Keep the unperturbed mdel
        Mod0 = Mod  # .deepcopy()
        D0 = DataPoint  # .deepcopy()
        prior0 = prior  # .copy()
        posterior0 = posterior  # .copy()
        PhiD0 = PhiD  # .copy()

    clk.stop()

    return(Mod0, D0, prior0, posterior0, PhiD0, posteriorComponents, clk.timeinSeconds())
    #%%
