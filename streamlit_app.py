"""
Oracle AI Agent - Streamlit 入口
================================
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# 运行仪表盘
from ui.dashboard import main

if __name__ == "__main__":
    main()
