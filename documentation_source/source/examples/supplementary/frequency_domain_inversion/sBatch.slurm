#!/bin/bash
#SBATCH --job-name=GeopBIPy_MTB_FD
#SBATCH -n 2016
#SBATCH -p large
#SBATCH --account=<account>
#SBATCH --time=12:00:00
#SBATCH --hint=compute_bound
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-user=<email>
#SBATCH -o %j.out

module load python/geobipy 

srun --mpi=pmi2 geobipyParallel -i userParameters_FD.py -o ./FD
