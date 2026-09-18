#!/bin/bash
#SBATCH --job-name=inject_ann
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=72         # CPU cores per task
#SBATCH --output=inject_ann.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Execute your Python script
python -u scripts/analysis_inject_only.py