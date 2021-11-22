#!/bin/bash
#SBATCH --job-name=tempest_synth
#SBATCH -n 101
#SBATCH -N 3
#SBATCH --hint=nomultithread
#SBATCH -p workq
#SBATCH --account=sas
#SBATCH --time=01:00:00
#SBATCH -o %j.out
module unload craype-hugepages2M
module swap PrgEnv-cray/6.0.5 PrgEnv-gnu
module load cray-hdf5-parallel/1.10.5.2
module load cray-python/3.7.3.2
export OMP_NUM_THREADS=1
export NUMBA_CPU_NAME='skylake'
mkdir tempest
srun geobipy_mpi tempest_options ./tempest 
