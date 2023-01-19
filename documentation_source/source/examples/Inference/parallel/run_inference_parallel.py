"""
1D Inference of Resolve Data
----------------------------

All plotting in GeoBIPy can be carried out using the 3D inference class

"""

def parallel_mpi(parameter_file, output_directory, data_filename):

    import pathlib
    from mpi4py import MPI
    from geobipy.src.base import MPI as myMPI
    from datetime import timedelta

    world = MPI.COMM_WORLD
    rank = world.rank
    nRanks = world.size
    masterRank = rank == 0

    myMPI.rankPrint(world,'Running GeoBIPy in parallel mode with {} cores'.format(nRanks))
    myMPI.rankPrint(world,'Using user input file {}'.format(parameter_file))
    myMPI.rankPrint(world,'Output files will be produced at {}'.format(output_directory))

    inputFile = pathlib.Path(parameter_file)
    assert inputFile.exists(), Exception("Cannot find input file {}".format(inputFile))

    output_directory = pathlib.Path(output_directory)
    assert output_directory.exists(), Exception("Make sure the output directory exists {}".format(output_directory))

    kwargs = user_parameters.read(inputFile)
    kwargs['data_filename'] = kwargs['data_filename'] + '//' + data_filename + '.csv'

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


def checkCommandArguments():
    """Check the users command line arguments. """
    import warnings
    import argparse
    # warnings.filterwarnings('error')

    Parser = argparse.ArgumentParser(description="GeoBIPy",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    Parser.add_argument('--index', dest='index', type=int, default=None, help='job array index')

    args = Parser.parse_args()

    return args.index

if __name__ == '__main__':
    import os
    import shutil
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt
    from geobipy import Inference3D
    from geobipy import user_parameters
    import numpy as np

    #%%
    # Running GeoBIPy to invert data
    # ++++++++++++++++++++++++++++++
    #
    # Define some directories and paths

    index = checkCommandArguments()
    sys.path.append(os.getcwd())

    datas = ['resolve', 'skytem', 'tempest']
    keys = ['glacial', 'saline_clay', 'resistive_dolomites', 'resistive_basement', 'coastal_salt_water', 'ice_over_salt_water']

    tmp = np.unravel_index(index, (3, 6))
    data = datas[tmp[0]]
    key = keys[tmp[1]]

    ################################################################################
    # The directory where HDF files will be stored
    output_directory = data + "//" + key
    ################################################################################
    file_path = os.path.join(output_directory, output_directory)
    Path(file_path).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(output_directory):
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    ################################################################################
    # The parameter file defines the set of user parameters needed to run geobipy.
    parameter_file = "{}_options".format(data)
    ################################################################################

    data_filename = data + '_' + key
    parallel_mpi(parameter_file, output_directory, data_filename)