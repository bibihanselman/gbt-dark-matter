#!/bin/bash
#SBATCH --job-name=optimize_populations_lscx_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=480:00:00
#SBATCH --output=../logs/optimize_populations_lscx_test_ce_all_090326.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u /home/bhanselman/bhanselman/scripts/optimize_populations_lscx_test.py