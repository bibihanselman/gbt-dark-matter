#!/bin/bash
#SBATCH --job-name=normalize_all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --output=../logs/normalize_all_082426.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u /home/bhanselman/bhanselman/scripts/normalize_all.py