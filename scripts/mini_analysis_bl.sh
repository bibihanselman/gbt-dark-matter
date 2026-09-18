#!/bin/bash

source /home/bibih/.virtualenvs/.venv-3/bin/activate

nohup python /home/bhanselman/bhanselman/scripts/bayes_mini_analysis_cband.py > /home/bhanselman/bhanselman/logs/bayes_mini_analysis_bank_1_061526.log 2>&1 &

echo "Analysis script submitted"