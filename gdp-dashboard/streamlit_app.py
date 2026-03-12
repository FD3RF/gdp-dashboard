"""
AI Quant Trading System - Streamlit Dashboard Entry Point
==========================================================

This is the main entry point for Streamlit Cloud deployment.
"""

import sys
from pathlib import Path

# Get the root directory
ROOT_DIR = Path(__file__).parent
DASHBOARD_DIR = ROOT_DIR / "dashboard"

# Add paths
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(DASHBOARD_DIR))

# Execute the dashboard app
exec(open(DASHBOARD_DIR / "streamlit_app.py").read())
