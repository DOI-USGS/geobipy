"""
1D Inference of Resolve Data
----------------------------

All plotting in GeoBIPy can be carried out using the 3D inference class

"""
from os.path import join
from geobipy import StatArray, Model
from geobipy import FdemData, TdemData, TempestData
from geobipy import get_prng

def parallel_mpi(data_type, model_type, output_directory):

    import pathlib
    from mpi4py import MPI
    from geobipy.src.base import MPI as myMPI
    from datetime import timedelta

    world = MPI.COMM_WORLD
    rank = world.rank
    nRanks = world.size
    masterRank = rank == 0

    # Make the data for the given test model
    # if masterRank:
    #     wedge_model = Model.create_synthetic_model(model_type)

        # if data_type == 'resolve':
        #     create_resolve(wedge_model, model_type)
        # elif data_type == 'skytem_512':
        #     create_skytem(wedge_model, model_type, 512)
        # elif data_type == 'skytem_304':
        #     create_skytem(wedge_model, model_type, 304)
        # elif data_type == 'aerotem':
        #     create_aerotem(wedge_model, model_type)
        # elif data_type == 'tempest':
        #     create_tempest(wedge_model, model_type)

    parameter_file = "../source/supplementary/options_files/{}_options".format(data_type)
    inputFile = pathlib.Path(parameter_file)
    assert inputFile.exists(), Exception("Cannot find input file {}".format(inputFile))

    output_directory = pathlib.Path(output_directory)
    assert output_directory.exists(), Exception("Make sure the output directory exists {}".format(output_directory))

    myMPI.rankPrint(world,'Running GeoBIPy in parallel mode with {} cores'.format(nRanks))
    myMPI.rankPrint(world,'Using user input file {}'.format(parameter_file))
    myMPI.rankPrint(world,'Output files will be produced at {}'.format(output_directory))

    kwargs = user_parameters.read(inputFile, n_markov_chains = 100000,
                                             update_plot_every = 5000,
                                             data_directory = "..//source//supplementary//data",
                                             data_filename = data_type + '_' + model_type + '.csv'
                                             )

    # Everyone needs the system classes read in early.
    data = kwargs['data_type']._initialize_sequential_reading(kwargs['data_filename'], kwargs['system_filename'])

    # Start keeping track of time.
    t0 = MPI.Wtime()

    prng = get_prng(seed=kwargs['seed'], world=world)

    inference3d = Inference3D(data, prng=prng, world=world)
    inference3d.create_hdf5(directory=output_directory, **kwargs)

    myMPI.rankPrint(world, "Created hdf5 files in {} h:m:s".format(str(timedelta(seconds=MPI.Wtime()-t0))))

    inference3d.infer(**kwargs)


def checkCommandArguments():
    """Check the users command line arguments. """
    import warnings
    import argparse
    # warnings.filterwarnings('error')

    Parser = argparse.ArgumentParser(description="GeoBIPy",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    Parser.add_argument('index', type=int, help='job array index 0-18')

    args = Parser.parse_args()

    return args.index

if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path
    from geobipy import Inference3D
    from geobipy import user_parameters
    import numpy as np

    #%%
    # Running GeoBIPy to invert data
    # ++++++++++++++++++++++++++++++
    #
    # Define some directories and paths

    np.random.seed(0)

    index = checkCommandArguments()
    sys.path.append(os.getcwd())

    datas = ['tempest', 'skytem', 'resolve']
    models = ['glacial', 'saline_clay', 'resistive_dolomites', 'resistive_basement', 'coastal_salt_water', 'ice_over_salt_water']

    tmp = np.unravel_index(index, (3, 6))

    data = datas[tmp[0]]
    model = models[tmp[1]]

    #%%
    # The directory where HDF files will be stored
    #%%
    file_path = os.path.join(data, model)
    Path(file_path).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(file_path):
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    parallel_mpi(data, model, file_path)
