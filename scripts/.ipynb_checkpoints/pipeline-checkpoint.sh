#!/bin/bash
#SBATCH --job-name=second_analysis_weighted_optimized_v3_ann
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=72         # CPU cores per task
#SBATCH --output=second_analysis_weighted_optimized_v3_ann.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Execute your Python script
python -u scripts/analysis_weighted_optimized.py 0 -c ann