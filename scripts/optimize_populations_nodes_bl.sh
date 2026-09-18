#!/bin/bash

source /home/bibih/.virtualenvs/.venv-3/bin/activate

nohup python /home/bhanselman/bhanselman/scripts/optimize_populations_nodes.py --name 062726/02 -m > /home/bhanselman/bhanselman/logs/optimize_populations_062726_02.log 2>&1 &

echo "Script submitted, see log for updates."