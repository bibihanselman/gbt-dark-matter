#!/bin/bash
#SBATCH --job-name=normalize_11080
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=72         # CPU cores per task
#SBATCH --mem=1G                   # Memory per node (e.g., 1GB)
#SBATCH --time=10:00:00            # Wall time limit (HH:MM:SS)
#SBATCH --output=normalize_11080_output.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules (e.g., Python, Anaconda)
# module purge
# module load anaconda3/2024.6  # Adjust version as needed
# conda activate my_conda_env   # If using a Conda environment
# #!/usr/bin/python3

# Execute your Python script
python -u scripts/inject_normalize_banks.py