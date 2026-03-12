"""
AI Quant Trading System - Streamlit Dashboard Entry Point
==========================================================

This is the main entry point for Streamlit Cloud deployment.
SECURITY: Uses import instead of exec() to prevent code injection.
"""

import sys
from pathlib import Path

# Get the root directory
ROOT_DIR = Path(__file__).parent
DASHBOARD_DIR = ROOT_DIR / "dashboard"

# Add paths safely
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(DASHBOARD_DIR))

# Safe import instead of exec()
# This prevents code injection vulnerabilities
try:
    from dashboard.streamlit_app import main
    main()
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to load dashboard: {e}")
    st.info("Please ensure the dashboard module is properly installed.")
except Exception as e:
    import streamlit as st
    st.error(f"Application error: {e}")
