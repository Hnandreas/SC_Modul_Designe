#!/usr/bin/env bash
# Starts the GUI in the browser (http://localhost:8501).
cd "$(dirname "$0")"
[ -d .venv ] || ./setup_env.sh
source .venv/bin/activate
# install missing packages (e.g. after an update of requirements.txt)
python -c "import streamlit, streamlit_image_coordinates, scipy, plotly" 2>/dev/null || pip install -q -r requirements.txt
streamlit run app/app.py "$@"
