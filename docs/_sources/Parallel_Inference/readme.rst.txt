Parallel Inference
------------------

The best way to run geobipy with MPI is through the command line entry point.
Upon install, pip will create the "geobipy" entry point into the code base.
This entry point can be used for both serial and parallel modes.

.. code-block:: bash

    srun geobipy options.py <output folder> --mpi

Please refer to the installation instructions for getting your Python environment setup with mpi4py and mpi enabled hdf5.
Install those two packages first before installing geobipy otherwise pip might inadvertently install the non-parallel-enabled hdf5 library.

Parallelization
+++++++++++++++

Geopbipy is currently parallelized using only MPI.  We do not use single machine parallel libraries like multiprocessing or joblib because we wanted scalability from the start.
We currently have no dependence between data points in a data set, so we can treat each data point independently from its neighbours.  This lends itself well to distributed parallelization using MPI.
One of the biggest bottlenecks of any parallel enabled program is file IO, we therefore alleviate this bottleneck by writing results to HDF5 files (With future scope to have these be properly georeferenced netcdf files)
Each unique line number in a data file will have its own separate hdf5 file.

Here is a sample slurm script to submit an mpi enabled job to the queue. Since we only care about total cores available, we dont need to worry too much about cores per node, or increasing RAM per core.  Geobipy operates with relatively small memory requirements, and we have tested with only 256MB per core available.
The code is currently showing linear scalability upto 9000 cores (which was our maximum available at the time).

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=geobipy
    #SBATCH -n 5000
    #SBATCH -p <partition>
    #SBATCH --account=<account name>
    #SBATCH --time=dd-hh:mm:ss
    #SBATCH -o %j.out

    # Your module load section to enable python
    module load cray-hdf5-parallel cray-python
    # FFTW is required when compiling the time domain forward modeller from Geoscience Australia
    module load cray-fftw

    # We use Numba to compile the Python frequency domain forward modeller into C
    export OMP_NUM_THREADS=1
    export NUMBA_CPU_NAME='skylake'  # Change your CPU name

    # Source your python environment how you need, either conda or venv
    source <Path to env /bin/activate>
    conda activate geobipy

    mkdir <output_folder>
    rm <output_folder>/*.h5  # We recommend this in case you have to restart a run.  HDF5 files can corrupt on unsuccessful exit.
    srun geobipy options.py <output_folder> --mpi