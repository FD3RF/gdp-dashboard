# features/volatility.py
"""
模块 16: 波动率聚类
==================
预判行情爆发前的寂静（变盘前兆）
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class VolatilityRegime(Enum):
    """波动率状态"""
    SILENCE = "寂静期"      # 变盘前兆
    LOW = "低波动"         # 正常市场
    NORMAL = "正常波动"    # 正常市场
    HIGH = "高波动"        # 活跃市场
    EXTREME = "极端波动"   # 可能变盘


@dataclass
class VolatilityAnalysis:
    """波动率分析结果"""
    current_vol: float
    historical_vol: float
    percentile: float
    regime: VolatilityRegime
    trend: str  # 上升/下降/稳定
    breakout_probability: float
    warning: str


class VolatilityCluster:
    """
    波动率聚类分析器
    
    功能：
    - 检测波动率状态
    - 预警变盘前兆
    - 计算突破概率
    """
    
    def __init__(self, 
                 silence_threshold: float = 0.5,
                 extreme_threshold: float = 2.0,
                 window: int = 20):
        """
        Args:
            silence_threshold: 寂静阈值 (相对于历史波动率的比值)
            extreme_threshold: 极端阈值
            window: 计算窗口
        """
        self.silence_threshold = silence_threshold
        self.extreme_threshold = extreme_threshold
        self.window = window
        self._history: List[float] = []
    
    def analyze(self, prices: np.ndarray) -> VolatilityAnalysis:
        """
        分析波动率
        
        Args:
            prices: 价格序列
            
        Returns:
            VolatilityAnalysis
        """
        if len(prices) < self.window * 2:
            return self._empty_analysis()
        
        # 计算收益率
        returns = np.diff(np.log(prices[-self.window * 2:]))
        
        # 当前波动率 (年化)
        current_vol = np.std(returns[-self.window:]) * np.sqrt(365 * 288)  # 5分钟K线
        
        # 历史波动率
        historical_vol = np.std(returns[:-self.window]) * np.sqrt(365 * 288)
        
        # 更新历史
        self._history.append(current_vol)
        if len(self._history) > 100:
            self._history = self._history[-100:]
        
        # 计算百分位
        if len(self._history) > 10:
            percentile = sum(1 for v in self._history if v <= current_vol) / len(self._history) * 100
        else:
            percentile = 50
        
        # 波动率比值
        vol_ratio = current_vol / (historical_vol + 1e-8)
        
        # 判断状态
        if vol_ratio < self.silence_threshold:
            regime = VolatilityRegime.SILENCE
            warning = "⚠️ 波动率极低，可能即将变盘"
        elif vol_ratio < 0.8:
            regime = VolatilityRegime.LOW
            warning = "波动率偏低"
        elif vol_ratio < 1.2:
            regime = VolatilityRegime.NORMAL
            warning = "波动率正常"
        elif vol_ratio < self.extreme_threshold:
            regime = VolatilityRegime.HIGH
            warning = "波动率偏高"
        else:
            regime = VolatilityRegime.EXTREME
            warning = "🚨 波动率极高，注意风险"
        
        # 趋势判断
        if len(self._history) >= 5:
            recent = self._history[-5:]
            if recent[-1] > recent[0] * 1.1:
                trend = "上升"
            elif recent[-1] < recent[0] * 0.9:
                trend = "下降"
            else:
                trend = "稳定"
        else:
            trend = "未知"
        
        # 突破概率 (寂静期时更高)
        if regime == VolatilityRegime.SILENCE:
            breakout_prob = 0.7
        elif regime == VolatilityRegime.LOW:
            breakout_prob = 0.5
        else:
            breakout_prob = 0.3
        
        return VolatilityAnalysis(
            current_vol=current_vol,
            historical_vol=historical_vol,
            percentile=percentile,
            regime=regime,
            trend=trend,
            breakout_probability=breakout_prob,
            warning=warning
        )
    
    def _empty_analysis(self) -> VolatilityAnalysis:
        """返回空分析"""
        return VolatilityAnalysis(
            current_vol=0,
            historical_vol=0,
            percentile=50,
            regime=VolatilityRegime.NORMAL,
            trend="未知",
            breakout_probability=0.3,
            warning="数据不足"
        )


def analyze_volatility(prices: np.ndarray) -> Dict[str, Any]:
    """快速分析波动率"""
    cluster = VolatilityCluster()
    result = cluster.analyze(prices)
    
    return {
        'current_vol': result.current_vol,
        'historical_vol': result.historical_vol,
        'percentile': result.percentile,
        'regime': result.regime.value,
        'trend': result.trend,
        'breakout_probability': result.breakout_probability,
        'warning': result.warning
    }
