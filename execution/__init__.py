# execution/__init__.py
"""
Oracle AI Agent - 执行模块
"""

# 延迟导入torch相关模块，避免启动时CUDA加载问题
from .risk_shield import (
    RiskShield,
    RiskLevel,
    RiskCheckResult,
    PositionInfo,
)
from .performance_calculator import (
    PrecisionCalculator,
    RealtimePerformanceTracker,
)

__all__ = [
    'RiskShield',
    'RiskLevel',
    'RiskCheckResult',
    'PositionInfo',
    'PrecisionCalculator',
    'RealtimePerformanceTracker',
]

# 延迟导入策略模块
def get_strategy_matrix():
    """延迟导入策略矩阵"""
    from .strategy_matrix import StrategyMatrix
    return StrategyMatrix
