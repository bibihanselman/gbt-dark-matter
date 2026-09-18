#!/bin/bash
#SBATCH --job-name=grid_asymmetry_more_windows
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=72         # CPU cores per task
#SBATCH --mem=16G                  # Memory per node (e.g., 1GB)
#SBATCH --time=48:00:00            # Wall time limit (HH:MM:SS)
#SBATCH --output=grid_asymmetry_more_windows.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules (e.g., Python, Anaconda)
# module purge
# module load anaconda3/2024.6  # Adjust version as needed
# conda activate my_conda_env   # If using a Conda environment

# Execute your Python script
python -u scripts/grid_asymmetry_snr_new.py