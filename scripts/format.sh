#!/bin/bash
pip3 install yapf --upgrade
yapf -ir -vv --style ./.style.yapf ./alpha_seed ./tasks ./scripts ./tests