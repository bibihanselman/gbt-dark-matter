#!/bin/bash
#SBATCH --job-name=bayes_weighted_ann
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=72         # CPU cores per task
#SBATCH --time=48:00:00            # Wall time limit (HH:MM:SS)
#SBATCH --output=bayes_weighted_ann.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u scripts/bayes_asymmetry_snr.py