#!/usr/bin/env python
# -- coding: utf-8 --


from os import getcwd
from os import makedirs
from os.path import join
import pathlib
import argparse
from importlib import import_module
import sys
from shutil import copy
import time
from datetime import timedelta

import h5py
import numpy as np
from numpy.random import randint

# Set up shorter aliases for classes within geobipy
# Base routines
from .src.base import customFunctions
from .src.base import customPlots
from .src.base import fileIO
from .src.base import interpolation

from .src.base.HDF import hdfRead
from .src.base.HDF import hdfWrite
# Classes within geobipy
# Core
from .src.classes.core.StatArray import StatArray
from .src.classes.core.Stopwatch import Stopwatch
# Data points
from .src.classes.data.datapoint.DataPoint import DataPoint
from .src.classes.data.datapoint.EmDataPoint import EmDataPoint
from .src.classes.data.datapoint.FdemDataPoint import FdemDataPoint
from .src.classes.data.datapoint.TdemDataPoint import TdemDataPoint
# Datasets
from .src.classes.data.dataset.Data import Data
from .src.classes.data.dataset.FdemData import FdemData
from .src.classes.data.dataset.TdemData import TdemData
# Systems
from .src.classes.system.FdemSystem import FdemSystem
from .src.classes.system.TdemSystem import TdemSystem
from .src.classes.system.Waveform import Waveform
from .src.classes.system.CircularLoop import CircularLoop
from .src.classes.system.SquareLoop import SquareLoop
from .src.classes.system.filters.butterworth import butterworth
# Meshes
from .src.classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from .src.classes.mesh.RectilinearMesh2D import RectilinearMesh2D
from .src.classes.mesh.TopoRectilinearMesh2D import TopoRectilinearMesh2D
# Models
from .src.classes.model.Model1D import Model1D
from .src.classes.model.AarhusModel import AarhusModel
# Pointclouds
from .src.classes.pointcloud.PointCloud3D import PointCloud3D
from .src.classes.pointcloud.Point import Point
# Statistics
from .src.classes.statistics.Distribution import Distribution
from .src.classes.statistics.MvDistribution import MvDistribution
from .src.classes.statistics.Histogram1D import Histogram1D
from .src.classes.statistics.Histogram2D import Histogram2D
from .src.classes.statistics.Hitmap2D import Hitmap2D
from .src.classes.statistics.mixStudentT import mixStudentT
from .src.classes.statistics.mixNormal import mixNormal
# McMC Inersion
from .src.inversion.Results import Results
from .src.inversion.LineResults import LineResults
from .src.inversion.DataSetResults import DataSetResults

from .src.inversion.Inv_MCMC import Initialize, Inv_MCMC


# Set an MPI failed tag
dpFailed = 0
# Set an MPI success tag
dpWin = 1
# Set an MPI run tag
run = 2
# Set an MPI exit tag
killSwitch = 9


def checkCommandArguments():
    """Check the users command line arguments. """
    import warnings
    # warnings.filterwarnings('error')

    Parser = argparse.ArgumentParser(description="GeoBIPy",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    Parser.add_argument('inputFile', help='User input file')
    Parser.add_argument('output_directory', help='Output directory for results')
    Parser.add_argument('--skipHDF5', dest='skipHDF5', default=False, help='Skip the creation of the HDF5 files.  Only do this if you know they have been created.')
    Parser.add_argument('--seed', dest='seed', type=int, default=None, help='Specify a single integer to fix the seed of the random number generator. Only used in serial mode.')

    args = Parser.parse_args()

    inputFile = pathlib.Path(args.inputFile)
    assert inputFile.exists(), Exception("Cannot find input file {}".format(inputFile))

    output_directory = pathlib.Path(args.output_directory)

    return inputFile, output_directory, args.skipHDF5, args.seed


def serial_geobipy(inputFile, output_directory, seed=None):

    print('Running GeoBIPy in serial mode')
    print('Using user input file {}'.format(inputFile))
    print('Output files will be produced at {}'.format(output_directory))

    # Make sure the results folders exist
    makedirs(output_directory, exist_ok=True)

    # Copy the input file to the output directory for reference.
    copy(inputFile, output_directory)

    # Import the script from the input file
    userParameters = import_module(str(inputFile.with_suffix('')), package='geobipy')

    # Everyone needs the system classes read in early.
    Dataset = type(userParameters.data_type)()

    if isinstance(Dataset, DataPoint):
        serial_datapoint(userParameters, output_directory, seed=seed)
    else:
        serial_dataset(userParameters, output_directory, seed=seed)


def serial_datapoint(userParameters, output_directory, seed=None):

    datapoint = type(userParameters.data_type)()
    datapoint.read(userParameters.dataFilename)

    # Get the random number generator
    prng = np.random.RandomState(seed)

    options = userParameters.userParameters(datapoint)
    options.output_directory = output_directory

    Inv_MCMC(options, datapoint, prng=prng)


def serial_dataset(userParameters, output_directory, seed=None):

    Dataset = type(userParameters.data_type)()

    results = DataSetResults(output_directory, userParameters.systemFilename)
    results.createHDF5(Dataset, userParameters)

    # Get the random number generator
    prng = np.random.RandomState(seed)

    # Loop through data points in the file.
    for _ in range(Dataset.nPoints):
        datapoint = Dataset._readSingleDatapoint()
        options = userParameters.userParameters(datapoint)

        iLine = results.lineIndex(lineNumber=datapoint.lineNumber)
        Inv_MCMC(options, datapoint, prng=prng, LineResults=results.lines[iLine])

    results.close()
    Dataset._closeDatafiles()


def parallel_geobipy(inputFile, outputDir, skipHDF5):

    parallel_mpi(inputFile, outputDir, skipHDF5)

def parallel_mpi(inputFile, outputDir, skipHDF5):

    from mpi4py import MPI
    from geobipy.src.base import MPI as myMPI

    world = MPI.COMM_WORLD
    rank = world.rank
    nRanks = world.size
    masterRank = rank == 0

    myMPI.rankPrint(world,'Running GeoBIPy in parallel mode with {} cores'.format(nRanks))
    myMPI.rankPrint(world,'Using user input file {}'.format(inputFile))
    myMPI.rankPrint(world,'Output files will be produced at {}'.format(outputDir))

    # Start keeping track of time.
    t0 = MPI.Wtime()
    t1 = t0

    UP = import_module(str(inputFile.with_suffix('')), package='geobipy')

    # Make data and system filenames lists of str.
    if isinstance(UP.dataFilename, str):
            UP.dataFilename = [UP.dataFilename]
    if isinstance(UP.systemFilename, str):
            UP.systemFilename = [UP.systemFilename]

    # Everyone needs the system classes read in early.
    Dataset = type(UP.data_type)()
    Dataset.readSystemFile(UP.systemFilename)

    # Get the number of points in the file.
    if masterRank:
        nPoints = Dataset._readNpoints(UP.dataFilename)
        assert (nRanks-1 <= nPoints+1), Exception('Do not ask for more cores than you have data points! Cores:nData {}:{} '.format(nRanks, nPoints))

    # Create a communicator containing only the master rank.
    allGroup = world.Get_group()
    masterGroup = allGroup.Incl([0])
    masterComm = world.Create(masterGroup)

    # Create a parallel RNG on each worker with a different seed.
    prng = myMPI.getParallelPrng(world, MPI.Wtime)

    myMPI.rankPrint(world, 'Creating HDF5 files, this may take a few minutes...')
    myMPI.rankPrint(world, 'Files are being created for data files {} and system files {}'.format(UP.dataFilename, UP.systemFilename))
    ### Only do this using the Master subcommunicator!
    # Here we initialize the HDF5 files.
    if (masterComm != MPI.COMM_NULL):

        # Make sure the results folders exist
        makedirs(outputDir, exist_ok=True)

        copy(inputFile, outputDir)

        # Prepare the dataset so that we can read a point at a time.
        Dataset._initLineByLineRead(UP.dataFilename, UP.systemFilename)
        # Get a datapoint from the file.
        DataPoint = Dataset._readSingleDatapoint()

        Dataset._closeDatafiles()

        # While preparing the file, we need access to the line numbers and fiducials in the data file
        tmp = fileIO.read_columns(UP.dataFilename[0], Dataset._indicesForFile[0][:2], 1, nPoints)

        Dataset._openDatafiles(UP.dataFilename)

        # Get the line numbers in the data
        lineNumbers = np.unique(tmp[:, 0])
        lineNumbers.sort()
        nLines = lineNumbers.size
        fiducials = tmp[:, 1]

        # Read in the user parameters
        paras = UP.userParameters(DataPoint)

        # Check the parameters
        paras.check(DataPoint)

        # Initialize the inversion to obtain the sizes of everything
        paras, Mod, DataPoint, prior, likelihood, posterior, PhiD = Initialize(paras, DataPoint, prng = prng)

        # Create the results template
        Res = Results(DataPoint, Mod,
            save=paras.save, plot=paras.plot, savePNG=paras.savePNG,
            nMarkovChains=paras.nMarkovChains, plotEvery=paras.plotEvery,
            reciprocateParameters=paras.reciprocateParameters)


        # For each line. Get the fiducials, and create a HDF5 for the Line results.
        # A line results file needs an initialized Results class for a single data point.
        if not skipHDF5:
            for line in lineNumbers:
                fiducialsForLine = np.where(tmp[:, 0] == line)[0]
                nFids = fiducialsForLine.size
                # Create a filename for the current line number
                fName = join(outputDir, '{}.h5'.format(line))
                # Open a HDF5 file in parallel mode.

                with h5py.File(fName, 'w', driver='mpio', comm=masterComm) as f:
                    LR = LineResults()
                    LR.createHdf(f, tmp[fiducialsForLine, 1], Res)
                myMPI.rankPrint(world,'Time to create line {} with {} data points: {} h:m:s'.format(line, nFids, str(timedelta(seconds=MPI.Wtime()-t0))))
                t0 = MPI.Wtime()

            myMPI.print('Initialized results for writing.')


    # Everyone needs the line numbers in order to open the results files collectively.
    if masterRank:
        DataPointType = DataPoint.hdfName()
    else:
        lineNumbers = None
        DataPointType = None
    lineNumbers = myMPI.Bcast(lineNumbers, world)
    nLines = lineNumbers.size

    DataPointType = world.bcast(DataPointType)

    # Open the files collectively
    LR = [None] * nLines
    for i, line in enumerate(lineNumbers):
        fName = join(outputDir, '{}.h5'.format(line))
        LR[i] = LineResults(fName, UP.systemFilename, hdfFile = h5py.File(fName, 'a', driver='mpio', comm=world))

    world.barrier()
    myMPI.rankPrint(world,'Files created in {} h:m:s'.format(str(timedelta(seconds=MPI.Wtime()-t1))))
    t0 = MPI.Wtime()

    # Carryout the master-worker tasks
    if (world.rank == 0):
        masterTask(Dataset, world)
    else:
        DataPoint = eval(customFunctions.safeEval(DataPointType))
        workerTask(DataPoint, UP, prng, world, lineNumbers, LR)

    world.barrier()
    # Close all the files. Must be collective.
    for i in range(nLines):
        LR[i].close()

    if masterRank:
        Dataset._closeDatafiles()


def masterTask(Dataset, world):
  """ Define a Send Recv Send procedure on the master """

  from mpi4py import MPI
  from geobipy.src.base import MPI as myMPI

  # Set the total number of data points
  nPoints = Dataset.nPoints

  nFinished = 0
  nSent = 0
#   continueRunning = np.empty(1, dtype=np.int32)
#   rankRecv = np.zeros(3, dtype = np.float64)

  # Send out the first indices to the workers
  for iWorker in range(1, world.size):
    # Get a datapoint from the file.
    DataPoint = Dataset._readSingleDatapoint()

    # If DataPoint is None, then we reached the end of the file and no more points can be read in.
    if DataPoint is None:
        # Send the kill switch to the worker to shut down.
        # continueRunning[0] = 0 # Do not continue running
        continueRunning =False
        world.send(continueRunning, dest=iWorker)
    else:
        # continueRunning[0] = 1 # Yes, continue with the next point.
        continueRunning = True
        world.send(continueRunning, dest=iWorker)
        DataPoint.Isend(dest=iWorker, world=world)

    nSent += 1

  # Start a timer
  t0 = MPI.Wtime()

  myMPI.print("Initial data points sent. Master is now waiting for requests")

  # Now wait to send indices out to the workers as they finish until the entire data set is finished
  while nFinished < nPoints:
    # Wait for a worker to request the next data point
    status = MPI.Status()
    dummy = world.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
    requestingRank = status.Get_source()
    # requestingRank = np.int(rankRecv[0])
    # dataPointProcessed = rankRecv[1]

    nFinished += 1

    # Read the next data point from the file
    DataPoint = Dataset._readSingleDatapoint()

    # If DataPoint is None, then we reached the end of the file and no more points can be read in.
    if DataPoint is None:
        # Send the kill switch to the worker to shut down.
        # continueRunning[0] = 0 # Do not continue running
        continueRunning = False
        world.send(continueRunning, dest=requestingRank)
    else:
        # continueRunning[0] = 1 # Yes, continue with the next point.
        continueRunning = True
        world.send(continueRunning, dest=requestingRank)
        DataPoint.Isend(dest=requestingRank, world=world, systems=DataPoint.system)

    report = (nFinished % (world.size - 1)) == 0 or nFinished == nPoints

    if report:
        e = MPI.Wtime() - t0
        elapsed = str(timedelta(seconds=e))
        eta = str(timedelta(seconds=(nPoints / nFinished-1) * e))
        myMPI.print("Remaining Points {}/{} || Elapsed Time: {} h:m:s || ETA {} h:m:s".format(nPoints-nFinished, nPoints, elapsed, eta))
        # myMPI.print('Inverted data point {} in {:.3f}s  ||  Time: {:.3f}s  ||  QueueLength: {}/{}  ||  ETA: {:.3f}s'.format(dataPointProcessed, rankRecv[2], elapsed, nPoints-nFinished, nPoints, eta))


def workerTask(_DataPoint, UP, prng, world, lineNumbers, LineResults):
    """ Define a wait run ping procedure for each worker """

    # Import here so serial code still works...
    from mpi4py import MPI
    from geobipy.src.base import MPI as myMPI

    # Initialize the worker process to go
    Go = True

    # Wait till you are told what to process next
    continueRunning = world.recv(source=0)
    # If we continue running, receive the next DataPoint. Otherwise, shutdown the rank
    if continueRunning:
        DataPoint = _DataPoint.Irecv(source=0, world=world)
    else:
        Go = False

    # communicationTime = 0.0

    while Go:
        # initialize the parameters
        paras = UP.userParameters(DataPoint)

        # Pass through the line results file object if a parallel file system is in use.
        iLine = lineNumbers.searchsorted(DataPoint.lineNumber)
        failed = Inv_MCMC(paras, DataPoint, prng=prng, rank=world.rank, LineResults=LineResults[iLine])

        # Ping the Master to request a new index
        t0 = MPI.Wtime()
        world.send(1, dest=0)

        # Wait till you are told whether to continue or not
        continueRunning = world.recv(source=0)

        if failed:
            print("Datapoint {} failed to converge".format(DataPoint.fiducial))

        # If we continue running, receive the next DataPoint. Otherwise, shutdown the rank
        if continueRunning:
            DataPoint = _DataPoint.Irecv(source=0, world=world, systems=DataPoint.system)
#            communicationTime += MPI.Wtime() - t0
        else:
#            communicationTime += MPI.Wtime() - t0
#            s = str(timedelta(seconds = communicationTime))
#            print('Communicaton Time on Rank {} {} h:m:s'.format(world.rank, s))
            Go = False


def geobipy():
    """Run the serial implementation of GeoBIPy. """

    inputFile, output_directory, _, seed = checkCommandArguments()
    sys.path.append(getcwd())

    serial_geobipy(inputFile, output_directory, seed)


def geobipy_mpi():
    """Run the parallel implementation of GeoBIPy. """

    inputFile, output_directory, skipHDF5, _ = checkCommandArguments()
    sys.path.append(getcwd())

    parallel_geobipy(inputFile, output_directory, skipHDF5)

