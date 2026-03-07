#!/bin/bash
#SBATCH --job-name=geobipy_synthetics
#SBATCH -n 80
#SBATCH --hint=nomultithread
#SBATCH -p cpu
#SBATCH --account=sas
#SBATCH --time=02:00:00
#SBATCH -o %j.out
#SBATCH --array=0-17

source ~/bin/load_gpy_mpi.sh

srun python run_test_suite_parallel.py $SLURM_ARRAY_TASK_ID
