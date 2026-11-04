#!/usr/bin/env bash
# Creates a local virtual environment (.venv) and installs all dependencies.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Fertig. GUI starten mit ./run_gui.sh"
