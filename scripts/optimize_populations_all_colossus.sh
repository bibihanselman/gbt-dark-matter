#!/bin/bash
#SBATCH --job-name=optimize_populations_cband_072026_ann
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --output=../logs/optimize_populations_cband_072026_ann.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u /home/bhanselman/bhanselman/scripts/optimize_populations_all.py "/home/bhanselman/bhanselman/scripts/optimize_populations_all_config_ann.yml"