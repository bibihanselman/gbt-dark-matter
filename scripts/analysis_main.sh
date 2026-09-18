#!/bin/bash
#SBATCH --job-name=analysis_cband_072026_decay
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --output=../logs/analysis_cband_072026_decay.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u /home/bhanselman/bhanselman/scripts/analysis_main.py "/home/bhanselman/bhanselman/scripts/analysis_cband_072026_config.yml"