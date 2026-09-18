#!/bin/bash
#SBATCH --job-name=inject_normalize
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=72         # CPU cores per task
#SBATCH --mem=1G                   # Memory per node (e.g., 1GB)
#SBATCH --output=inject_normalize_output.log

# Load necessary modules (e.g., Python, Anaconda)
# module purge
# module load anaconda3/2024.6  # Adjust version as needed
# conda activate my_conda_env   # If using a Conda environment
# #!/usr/bin/python3

# Execute your Python script
python -u scripts/inject_normalize_banks.py -i