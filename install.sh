#!/bin/bash
set -euo pipefail

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
python3 -m pip install -e abides-core
python3 -m pip install -e abides-markets
python3 -m pip install -e abides-gym
