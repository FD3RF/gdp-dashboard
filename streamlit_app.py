"""
ETH 5分钟四维共振策略 - Streamlit 入口
重定向到 main/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from main.streamlit_app import main

if __name__ == "__main__":
    main()
