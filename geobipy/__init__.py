#!/usr/bin/env python
# -- coding: utf-8 --
from os import getcwd
from os import makedirs
import pathlib
import argparse
import sys
import shutil
from datetime import timedelta

# from .src.base import utilities
# from .src.base import plotting
# from .src.base import fileIO
# from .src.base import interpolation

# from .src.base.HDF import hdfRead
# from .src.base.HDF import hdfWrite
# Classes within geobipy
# Core
from .src.classes.core.StatArray import StatArray
# from .src.classes.core.Stopwatch import Stopwatch
# Data points
from .src.classes.data.datapoint.DataPoint import DataPoint
from .src.classes.data.datapoint.EmDataPoint import EmDataPoint
from .src.classes.data.datapoint.FdemDataPoint import FdemDataPoint
from .src.classes.data.datapoint.TdemDataPoint import TdemDataPoint
from .src.classes.data.datapoint.Tempest_datapoint import Tempest_datapoint
# Datasets
from .src.classes.data.dataset.Data import Data
from .src.classes.data.dataset.FdemData import FdemData
from .src.classes.data.dataset.TdemData import TdemData
from .src.classes.data.dataset.TempestData import TempestData
# Systems
from .src.classes.system.FdemSystem import FdemSystem
from .src.classes.system.TdemSystem import TdemSystem
from .src.classes.system.Waveform import Waveform
from .src.classes.system.CircularLoop import CircularLoop
from .src.classes.system.CircularLoops import CircularLoops
from .src.classes.system.SquareLoop import SquareLoop
from .src.classes.system.filters.butterworth import butterworth
# Meshes
from .src.classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from .src.classes.mesh.RectilinearMesh2D import RectilinearMesh2D
from .src.classes.mesh.RectilinearMesh2D_stitched import RectilinearMesh2D_stitched
from .src.classes.mesh.RectilinearMesh3D import RectilinearMesh3D
# Models
from .src.classes.model.Model import Model
# Pointclouds
from .src.classes.pointcloud.PointCloud3D import PointCloud3D
from .src.classes.pointcloud.Point import Point
# Statistics
from .src.classes.statistics.Distribution import Distribution
from .src.classes.statistics.Histogram import Histogram
from .src.classes.statistics.Mixture import Mixture
from .src.classes.statistics.mixStudentT import mixStudentT
from .src.classes.statistics.mixNormal import mixNormal
from .src.classes.statistics.mixPearson import mixPearson
# McMC Inersion
from .src.inversion.Inference1D import Inference1D
from .src.inversion.Inference2D import Inference2D
from .src.inversion.Inference3D import Inference3D
from .src.inversion.user_parameters import user_parameters

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
    Parser.add_argument('--index', dest='index', type=int, default=None, help='Invert this data point only. Only used in serial mode.')
    Parser.add_argument('--fiducial', dest='fiducial', type=float, default=None, help='Invert this fiducial only. Only used in serial mode.')
    Parser.add_argument('--line', dest='line_number', type=float, default=None, help='Invert the fiducial on this line. Only used in serial mode.')
    Parser.add_argument('--verbose', dest='verbose', default=False, help='Throw warnings as errors.')
    Parser.add_argument('--mpi', dest='mpi', default=False, help='Run geobipy with MPI libraries.')

    args = Parser.parse_args()

    if args.verbose:
        import warnings
        warnings.filterwarnings("error")

    return args.inputFile, args.output_directory, args.skipHDF5, args.seed, args.index, args.fiducial, args.line_number, args.mpi


def serial_geobipy(inputFile, output_directory, seed=None, index=None, fiducial=None, line_number=None):

    print('Running GeoBIPy in serial mode')
    print('Using user input file {}'.format(inputFile))
    print('Output files will be produced at {}'.format(output_directory))

    inputFile = pathlib.Path(inputFile)
    assert inputFile.exists(), Exception("Cannot find input file {}".format(inputFile))

    output_directory = pathlib.Path(output_directory)
    assert output_directory.exists(), Exception("Make sure the output directory exists {}".format(output_directory))

    # Make sure the results folders exist
    makedirs(output_directory, exist_ok=True)

    # Copy the input file to the output directory for reference.
    shutil.copy(inputFile, output_directory)

    options = user_parameters.read(inputFile)

    # Everyone needs the system classes read in early.
    # Dataset = userParameters.data_type()

    # if isinstance(Dataset, DataPoint):
    #     serial_datapoint(userParameters, output_directory, seed=seed)
    # else:
    serial_dataset(output_directory, seed=seed, index=index, fiducial=fiducial, line_number=line_number, **options)


# def serial_datapoint(options, output_directory, seed=None):

#     datapoint = type(options.data_type)()
#     datapoint.read(options.data_filename)

#     # Get the random number generator
#     prng = np.random.RandomState(seed)

#     # options = userParameters.userParameters(datapoint)
#     # options.output_directory = output_directory

#     infer(options, datapoint, prng=prng)


def serial_dataset(output_directory, seed=None, index=None, fiducial=None, line_number=None, **kwargs):

    dataset = kwargs['data_type'](system=kwargs['system_filename'])

    inference3d = Inference3D(output_directory, kwargs['system_filename'])
    inference3d.create_hdf5(dataset, **kwargs)

    inference3d.infer(dataset, seed=seed, index=index, fiducial=fiducial, line_number=line_number, **kwargs)

def parallel_geobipy(inputFile, outputDir, skipHDF5):

    parallel_mpi(inputFile, outputDir, skipHDF5)

def parallel_mpi(inputFile, output_directory, skipHDF5):

    from mpi4py import MPI
    from .src.base import MPI as myMPI

    world = MPI.COMM_WORLD
    rank = world.rank
    nRanks = world.size
    masterRank = rank == 0

    myMPI.rankPrint(world,'Running GeoBIPy in parallel mode with {} cores'.format(nRanks))
    myMPI.rankPrint(world,'Using user input file {}'.format(inputFile))
    myMPI.rankPrint(world,'Output files will be produced at {}'.format(output_directory))

    inputFile = pathlib.Path(inputFile)
    assert inputFile.exists(), Exception("Cannot find input file {}".format(inputFile))

    output_directory = pathlib.Path(output_directory)
    assert output_directory.exists(), Exception("Make sure the output directory exists {}".format(output_directory))

    kwargs = user_parameters.read(inputFile)

    # Everyone needs the system classes read in early.
    dataset = kwargs['data_type'](system=kwargs['system_filename'])

    # Get the number of points in the file.
    if masterRank:
    #     nPoints = dataset._csv_n_points(UP.dataFilename)
    #     assert (nRanks > 1), Exception("You need to use at least 2 ranks for the mpi version.")
    #     assert (nRanks <= nPoints+1), Exception('You requested more ranks than you have data points.  Please lower the number of ranks to a maximum of {}. '.format(nPoints+1))

    #     # Make sure the results folders exist
    #     makedirs(output_directory, exist_ok=True)
        # Copy the user_parameter file to the output directory
        shutil.copy(inputFile, output_directory)

    # Start keeping track of time.
    t0 = MPI.Wtime()

    inference3d = Inference3D(output_directory, kwargs['system_filename'], mpi_enabled=True)
    inference3d.create_hdf5(dataset, **kwargs)

    myMPI.rankPrint(world, "Created hdf5 files in {} h:m:s".format(str(timedelta(seconds=MPI.Wtime()-t0))))

    inference3d.infer(dataset, **kwargs)

def geobipy():
    """Run the serial implementation of GeoBIPy. """

    inputFile, output_directory, skipHDF5, seed, index, fiducial, line_number, mpi_enabled = checkCommandArguments()
    sys.path.append(getcwd())


    if mpi_enabled:
        parallel_geobipy(inputFile, output_directory, skipHDF5)
    else:
        serial_geobipy(inputFile, output_directory, seed, index, fiducial, line_number)
