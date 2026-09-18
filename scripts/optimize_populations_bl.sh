#!/bin/bash

source /home/bibih/.virtualenvs/.venv-3/bin/activate

nohup python /home/bhanselman/bhanselman/scripts/optimize_populations_v2_numba.py --name 071226/01 -d > /home/bhanselman/bhanselman/logs/optimize_populations_071226_01.log 2>&1 &

echo "Script submitted, see log for updates."