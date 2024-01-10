#!/bin/bash
#SBATCH --job-name=geobipy_synthetics
#SBATCH -n 80
#SBATCH -N 2
#SBATCH --hint=nomultithread
#SBATCH -p workq
#SBATCH --account=sas
#SBATCH --time=02:00:00
#SBATCH -o %j.out
#SBATCH --array=8,11,14,15,16,17
module swap PrgEnv-cray/6.0.5 PrgEnv-gnu
module load cray-hdf5-parallel cray-python cray-fftw

export OMP_NUM_THREADS=1
export NUMBA_CPU_NAME='skylake'

source /caldera/hytest_scratch/scratch/nfoks/pGeobipy/bin/activate

srun python run_test_suite_parallel.py $SLURM_ARRAY_TASK_ID
