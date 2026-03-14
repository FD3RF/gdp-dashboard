"""
ETH 5分钟四维共振策略
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from main.streamlit_app import main

if __name__ == "__main__":
    main()
