#!/bin/bash
pip3 install yapf --upgrade
python3 -m yapf -ir -vv --style ./.style.yapf ./alpha_seed ./tasks ./scripts ./tests