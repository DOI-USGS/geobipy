#!/bin/bash
#SBATCH --job-name=geobipy_synthetics
#SBATCH -n 120
#SBATCH -N 3
#SBATCH --hint=nomultithread
#SBATCH -p workq
#SBATCH --account=sas
#SBATCH --time=02:00:00
#SBATCH -o %j.out
#SBATCH --array=0-18
module swap PrgEnv-cray/6.0.5 PrgEnv-gnu
module load cray-hdf5-parallel cray-python cray-fftw

export OMP_NUM_THREADS=1
export NUMBA_CPU_NAME='skylake'

srun python run_inference_parallel.py --index=$SLURM_ARRAY_ID
