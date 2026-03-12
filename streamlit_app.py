"""
Oracle AI Agent - Streamlit Dashboard Entry Point
==================================================

神经链接仪表盘入口
"""

import sys
from pathlib import Path

# 添加项目路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# 运行仪表盘
from ui.neural_dashboard import run_dashboard

if __name__ == '__main__':
    run_dashboard()
