"""
AI Quant Trading System - Streamlit Dashboard Entry Point
==========================================================

This is the main entry point for Streamlit Cloud deployment.
SECURITY: Uses runpy instead of exec() to prevent code injection.
"""

import sys
import os
from pathlib import Path

# Get the root directory
ROOT_DIR = Path(__file__).parent
DASHBOARD_DIR = ROOT_DIR / "dashboard"

# Add paths safely
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(DASHBOARD_DIR))

# Change to dashboard directory for relative imports
os.chdir(str(DASHBOARD_DIR))

# Use runpy to safely execute the dashboard module
import runpy

try:
    runpy.run_path(str(DASHBOARD_DIR / "streamlit_app.py"), run_name="__main__")
except Exception as e:
    import streamlit as st
    st.error(f"Failed to load dashboard: {e}")
    st.info("Please ensure the dashboard module is properly installed.")
    import traceback
    st.code(traceback.format_exc())
