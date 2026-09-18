#!/bin/bash

# Load necessary modules or activate virtual environments
source /home/bibih/.virtualenvs/.venv-3/bin/activate

nohup stdbuf -oL python bhanselman/scripts/analysis_weighted_calibrated_binary.py analysis_20251130 27 23 > bhanselman/logs/analysis_20251130_annihilation.log 2>&1 &

echo "Analysis script submitted"