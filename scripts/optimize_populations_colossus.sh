#!/bin/bash
#SBATCH --job-name=optimize_populations_wtf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --output=../logs/optimize_populations_wtf.log
#SBATCH --mail-user=bhanselman@berkeley.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u optimize_populations_v2_numba.py --name wtf -a