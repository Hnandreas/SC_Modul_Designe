#!/usr/bin/env bash
# Starts the GUI in the browser (http://localhost:8501).
cd "$(dirname "$0")"
[ -d .venv ] || ./setup_env.sh
source .venv/bin/activate
streamlit run app/app.py "$@"
