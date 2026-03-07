#!/bin/bash
#SBATCH --job-name=resolve_synth
#SBATCH -n 101
#SBATCH -N 3
#SBATCH --hint=nomultithread
#SBATCH -p workq
#SBATCH --account=sas
#SBATCH --time=01:00:00
#SBATCH -o %j.out
module swap PrgEnv-cray PrgEnv-gnu
module load cray-python cray-hdf5-parallel cray-fftw

export OMP_NUM_THREADS=1
export NUMBA_CPU_NAME='skylake'
srun geobipy_mpi resolve_options ./resolve
