#!/bin/bash
#SBATCH --job-name=geobipy_SLV
#SBATCH -n 2016
# #SBATCH -N 2
#SBATCH -p large
#SBATCH --account=<account>
#SBATCH --time=16:00:00
#SBATCH --hint=compute_bound
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-user=<email>
#SBATCH -o %j.out

module load python/geobipy 

srun --mpi=pmi2 geobipyParallel -i userParameters_TD.py -o .
