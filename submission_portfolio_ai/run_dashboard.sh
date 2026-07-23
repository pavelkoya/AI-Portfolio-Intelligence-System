#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run reporting/dashboard.py \
    --server.port 8501 \
    --server.headless false
