#!/usr/bin/env python
# -- coding: utf-8 --


from os import getcwd
from os import makedirs
from os.path import join
import argparse
from importlib import import_module
import sys

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
from .src.classes.data.datapoint.EmDataPoint import EmDataPoint
from .src.classes.data.datapoint.FdemDataPoint import FdemDataPoint
from .src.classes.data.datapoint.TdemDataPoint import TdemDataPoint
from .src.classes.data.datapoint.MTDataPoint import MTDataPoint
# Datasets
from .src.classes.data.dataset.Data import Data
from .src.classes.data.dataset.FdemData import FdemData
from .src.classes.data.dataset.TdemData import TdemData
from .src.classes.data.dataset.MTData import MTData
# Systems
from .src.classes.system.FdemSystem import FdemSystem
from .src.classes.system.MTSystem import MTSystem
# Meshes
from .src.classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from .src.classes.mesh.RectilinearMesh2D import RectilinearMesh2D
# Models
from .src.classes.model.Model1D import Model1D
# Pointclouds
from .src.classes.pointcloud.PointCloud3D import PointCloud3D
# Statistics
from .src.classes.statistics.Distribution import Distribution
from .src.classes.statistics.Histogram1D import Histogram1D
from .src.classes.statistics.Histogram2D import Histogram2D
from .src.classes.statistics.Hitmap2D import Hitmap2D
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
    warnings.filterwarnings('error')

    Parser = argparse.ArgumentParser(description="GeoBIPy",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    Parser.add_argument('inputFile', help='User input file')
    Parser.add_argument('outputDir', help='Output directory for results')
    Parser.add_argument('--skipHDF5', dest='skipHDF5', default=False, help='Skip the creation of the HDF5 files.  Only do this if you know they have been created.')
    
    args = Parser.parse_args()

    # Strip .py from the input file name
    inputFile = args.inputFile.replace('.py','')

    return inputFile, args.outputDir, args.skipHDF5


def masterTask(myData, world):
  """ Define a Send Recv Send procedure on the master """
  
  from mpi4py import MPI
  from geobipy.src.base import MPI as myMPI
  
  # Set the total number of data points

  N = myData.N

  # Create and shuffle and integer list for the number of data points
  randomizedPointIndices = np.arange(N)
  np.random.shuffle(randomizedPointIndices)

  nFinished = 0
  nSent = 0
  dataSend = np.zeros(1, dtype = np.int64)
  rankRecv = np.zeros(3, dtype = np.float32)

  # Send out the first indices to the workers
  for iWorker in range(1,world.size):
    dataSend[:] = randomizedPointIndices[nSent]
    world.Send(dataSend, dest = iWorker, tag = run)
    nSent += 1

  # Start a timer
  t0 = MPI.Wtime()

  myMPI.print("Initial data points sent. Master is now waiting for requests")

  # Now wait to send indices out to the workers as they finish until the entire data set is finished
  while nFinished < N:
    # Wait for a worker to ping you
    
    world.Recv(rankRecv, source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = MPI.Status())
    workerRank = np.int(rankRecv[0])
    dataPointProcessed = np.int(rankRecv[1])

    nFinished += 1

    # Send out the next point if the list is not empty
    if (nSent < N):
      dataSend[:] = randomizedPointIndices[nSent]
      world.Send(dataSend, dest = workerRank, tag = run)
      nSent += 1
    else:
      dataSend[0] = -1
      world.Send(dataSend, dest = workerRank, tag = killSwitch)
     
    elapsed = MPI.Wtime() - t0
    eta = (N/nFinished-1) * elapsed
    myMPI.print('Inverted data point {} in {:.3f}s  ||  Time: {:.3f}s  ||  QueueLength: {}/{}  ||  ETA: {:.3f}s'.format(dataPointProcessed, rankRecv[2], elapsed, N-nFinished, N, eta))


def workerTask(myData, UP, prng, world, LineResults):
  """ Define a wait run ping procedure for each worker """
  
  from mpi4py import MPI
  from geobipy.src.base import MPI as myMPI
  
  # Initialize the worker process to go
  Go = True

  # Wait until the master sends you an index to process
  i = np.empty(1, dtype=np.int64)
  myRank = np.empty(3, dtype=np.float32)
  mpi_status = MPI.Status()
  world.Recv(i, source = 0, tag = MPI.ANY_TAG, status = mpi_status)
  iDataPoint = i[0]

  # Check if a killSwitch for this worker was thrown
  if mpi_status.Get_tag() == killSwitch:
      Go = False

  lines = np.unique(myData.line)
  lines.sort()

  while Go:
    t0 = MPI.Wtime()
    # Get the data point for the given index
    DataPoint = myData.getDataPoint(iDataPoint)
    paras = UP.userParameters(DataPoint)

    # Pass through the line results file object if a parallel file system is in use.
    iLine = lines.searchsorted(myData.line[iDataPoint])
    Inv_MCMC(paras, DataPoint, myData.id[iDataPoint], prng=prng, rank=world.rank, LineResults=LineResults[iLine])
    
    # Send the current rank number to the master
    myRank[:] = (world.rank, iDataPoint, MPI.Wtime() - t0)

    # With the Data Point inverted, Ping the Master to request a new index
    world.Send(myRank, dest = 0)

    # Wait till you are told what to process next
    mpi_status = MPI.Status()
    world.Recv(i, source = 0, tag = MPI.ANY_TAG, status = mpi_status)
    iDataPoint = i[0]

    # Check if a killSwitch for this worker was thrown
    if mpi_status.Get_tag() == killSwitch:
      Go = False


def multipleCore(inputFile, outputDir, skipHDF5):
    
    from mpi4py import MPI
    from geobipy.src.base import MPI as myMPI
    
    world = MPI.COMM_WORLD
    myMPI.rankPrint(world,'Running EMinv1D_MCMC')

    UP = import_module(inputFile, package=None)

    AllData = eval(UP.dataInit)
    # Initialize the data object on master
    if (world.rank == 0):
        AllData.read(UP.dataFname, UP.sysFname)

    myData = AllData.Bcast(world)
    if (world.rank == 0): myData = AllData

    myMPI.rankPrint(world,'Data Broadcast')

    assert (world.size <= myData.N+1), 'Do not ask for more cores than you have data points! Cores:nData '+str([world.size,myData.N])

    allGroup = world.Get_group()
    masterGroup = allGroup.Incl([0])
    masterComm = world.Create(masterGroup)

    t0 = MPI.Wtime()
    t1 = t0

    prng = myMPI.getParallelPrng(world, MPI.Wtime)

    # Make sure the line results folders exist
    try:
        makedirs(outputDir)
    except:
        pass

    # Get a datapoint, it doesnt matter which one
    DataPoint=myData.getDataPoint(0)
    # Read in the user parameters
    paras=UP.userParameters(DataPoint)
    # Check the parameters
    paras.check(DataPoint)
    # Initialize the inversion to obtain the sizes of everything
    [paras, Mod, D, prior, posterior, PhiD] = Initialize(paras, DataPoint, prng=prng)
    # Create the results template
    Res = Results(paras.save, paras.plot, paras.savePNG, paras, D, Mod)

    world.barrier()
    myMPI.rankPrint(world,'Initialized Results')

    # Get the line numbers in the data
    lines=np.unique(myData.line)
    lines.sort()
    nLines = lines.size

    world.barrier()
    
    myMPI.rankPrint(world,'Creating HDF5 files, this may take a few minutes...')
    ### Only do this using the subcommunicator!
    if (masterComm != MPI.COMM_NULL):
        for i in range(nLines):
            j = np.where(myData.line == lines[i])[0]
            fName = join(outputDir, str(lines[i])+'.h5')
            with h5py.File(fName, 'w', driver='mpio', comm=masterComm) as f:
                LR = LineResults()
                LR.createHdf(f, myData.id[j], Res)
            myMPI.rankPrint(world,'Time to create the line with {} data points: {:.3f} s'.format(j.size, MPI.Wtime()-t0))
            t0 = MPI.Wtime()

    world.barrier()
    
    # Open the files collectively
    LR = [None]*nLines
    for i in range(nLines):
        fName = join(outputDir,str(lines[i])+'.h5')
        LR[i] = LineResults(fName, hdfFile = h5py.File(fName,'a', driver='mpio',comm=world))
        # myMPI.print("rank {} line {} iDs {}".format(world.rank, i, LR[i].iDs))


    world.barrier()
    myMPI.rankPrint(world,'Files Created in {:.3f} s'.format(MPI.Wtime()-t1))
    t0 = MPI.Wtime()

    # Carryout the master-worker tasks
    if (world.rank == 0):
        masterTask(myData, world)
    else:
        workerTask(myData, UP, prng, world, LR)

    world.barrier()
    # Close all the files
    for i in range(nLines):
        LR[i].close()


def singleCore(inputFile, outputDir):
    # Import the script from the input file
    UP = import_module(inputFile, package=None)

    
    AllData = eval(UP.dataInit)
    AllData.read(UP.dataFname, UP.sysFname)

    # Make sure both dataPoint and line results folders exist
    try:
        makedirs(outputDir)
    except:
        pass

    # Get the random number generator
    prng = np.random.RandomState()

    # Get a datapoint, it doesnt matter which one
    DataPoint = AllData.getDataPoint(0)
    # Read in the user parameters
    paras = UP.userParameters(DataPoint)
    # Check the parameters
    paras.check(DataPoint)
    # Initialize the inversion to obtain the sizes of everything
    [paras, Mod, D, prior, posterior, PhiD] = Initialize(paras, DataPoint, prng=prng)
    # Create the results template
    Res = Results(paras.save, paras.plot, paras.savePNG, paras, D, Mod)

 
    # Get the line numbers in the data
    lines = np.unique(AllData.line)
    lines.sort()
    nLines = lines.size
    LR = [None]*nLines
    H5Files = [None]*nLines
    for i in range(nLines):
        H5Files[i] = h5py.File(join(outputDir, str(lines[i])+'.h5'), 'w')
        j = np.where(AllData.line == lines[i])[0]
        LR[i] = LineResults()
        LR[i].createHdf(H5Files[i], AllData.id[j], Res)


    for i in range(AllData.N):
        DataPoint = AllData.getDataPoint(i)
        paras = UP.userParameters(DataPoint)

        iLine = lines.searchsorted(AllData.line[i])
        Inv_MCMC(paras, DataPoint, AllData.id[i], prng=prng, LineResults=LR[iLine])

    for i in range(nLines):
        H5Files[i].close()


def runSerial():
    """Run the serial implementation of GeoBIPy. """
        
    inputFile, outputDir, skipHDF5 = checkCommandArguments()    
    sys.path.append(getcwd())

    R = singleCore(inputFile, outputDir)


def runParallel():
    """Run the parallel implementation of GeoBIPy. """

    inputFile, outputDir, skipHDF5 = checkCommandArguments()    
    sys.path.append(getcwd())

    R = multipleCore(inputFile, outputDir, skipHDF5)
