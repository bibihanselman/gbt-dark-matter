#!/bin/bash
#SBATCH --job-name=plot_asymmetry_corr
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=72         # CPU cores per task
#SBATCH --time=48:00:00            # Wall time limit (HH:MM:SS)
#SBATCH --output=plot_asymmetry_corr.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules (e.g., Python, Anaconda)
# module purge
# module load anaconda3/2024.6  # Adjust version as needed
# conda activate my_conda_env   # If using a Conda environment

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Execute your Python script
python -u scripts/plot_asymmetry_combined_pvals.py