#!/bin/bash
#SBATCH --job-name=mini_analysis_exp_071826
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --output=../logs/mini_analysis_exp_071826.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u mini_analysis_secant.py