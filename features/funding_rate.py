# features/funding_rate.py
"""
模块 42: 资金费率极值监控
=========================
实时捕捉多空拥挤度，预警"多杀多"或"空杀空"踩踏
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class FundingRateSignal(Enum):
    """资金费率信号"""
    EXTREME_LONG = "多杀多预警"
    EXTREME_SHORT = "空杀空预警"
    BULLISH = "看多情绪"
    BEARISH = "看空情绪"
    NEUTRAL = "中性"


@dataclass
class FundingRateAnalysis:
    """资金费率分析结果"""
    current_rate: float
    historical_avg: float
    percentile: float  # 历史百分位
    signal: FundingRateSignal
    crowd_bias: str  # 多/空拥挤度
    warning_level: str  # high/medium/low
    description: str
    confidence: float


class FundingRateMonitor:
    """
    资金费率监控器
    
    功能：
    - 监控资金费率极值
    - 检测多空拥挤度
    - 预警踩踏风险
    """
    
    def __init__(self, 
                 extreme_threshold: float = 0.001,  # 0.1%
                 warning_threshold: float = 0.0005):  # 0.05%
        """
        Args:
            extreme_threshold: 极值阈值
            warning_threshold: 预警阈值
        """
        self.extreme_threshold = extreme_threshold
        self.warning_threshold = warning_threshold
        self._history: List[Tuple[datetime, float]] = []
        self._max_history = 1000
    
    def update(self, rate: float, timestamp: Optional[datetime] = None) -> None:
        """更新资金费率历史"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self._history.append((timestamp, rate))
        
        # 保持历史长度
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def analyze(self, current_rate: float) -> FundingRateAnalysis:
        """
        分析资金费率
        
        Args:
            current_rate: 当前资金费率
            
        Returns:
            FundingRateAnalysis
        """
        # 更新历史
        self.update(current_rate)
        
        # 计算历史统计
        if len(self._history) > 10:
            rates = [r for _, r in self._history]
            historical_avg = np.mean(rates)
            std = np.std(rates)
            percentile = sum(1 for r in rates if r <= current_rate) / len(rates) * 100
        else:
            historical_avg = current_rate
            percentile = 50
        
        # 判断信号
        signal, crowd_bias, warning_level, description = self._interpret_rate(
            current_rate, historical_avg, percentile
        )
        
        # 计算置信度
        confidence = self._calculate_confidence(current_rate, historical_avg, std if len(self._history) > 10 else 0)
        
        return FundingRateAnalysis(
            current_rate=current_rate,
            historical_avg=historical_avg,
            percentile=percentile,
            signal=signal,
            crowd_bias=crowd_bias,
            warning_level=warning_level,
            description=description,
            confidence=confidence
        )
    
    def _interpret_rate(self, rate: float, avg: float, percentile: float) -> Tuple:
        """解读资金费率"""
        # 正费率 = 多头付费给空头 = 市场偏多
        # 负费率 = 空头付费给多头 = 市场偏空
        
        abs_rate = abs(rate)
        
        if rate > self.extreme_threshold:
            # 极端正费率 - 多头过度拥挤
            signal = FundingRateSignal.EXTREME_LONG
            crowd_bias = "多头极度拥挤"
            warning_level = "high"
            description = f"资金费率{rate*100:.3f}%极高，多头过度拥挤，警惕多杀多踩踏"
            
        elif rate < -self.extreme_threshold:
            # 极端负费率 - 空头过度拥挤
            signal = FundingRateSignal.EXTREME_SHORT
            crowd_bias = "空头极度拥挤"
            warning_level = "high"
            description = f"资金费率{rate*100:.3f}%极低，空头过度拥挤，警惕空杀空踩踏"
            
        elif rate > self.warning_threshold:
            signal = FundingRateSignal.BULLISH
            crowd_bias = "偏多情绪"
            warning_level = "low"
            description = f"资金费率{rate*100:.3f}%偏高，市场偏多"
            
        elif rate < -self.warning_threshold:
            signal = FundingRateSignal.BEARISH
            crowd_bias = "偏空情绪"
            warning_level = "low"
            description = f"资金费率{rate*100:.3f}%偏低，市场偏空"
            
        else:
            signal = FundingRateSignal.NEUTRAL
            crowd_bias = "中性"
            warning_level = "low"
            description = f"资金费率{rate*100:.3f}%正常，市场情绪中性"
        
        return signal, crowd_bias, warning_level, description
    
    def _calculate_confidence(self, rate: float, avg: float, std: float) -> float:
        """计算置信度"""
        if std < 1e-8:
            return 0.5
        
        # 越偏离均值，置信度越高
        z_score = abs(rate - avg) / std
        confidence = min(1.0, z_score / 3)  # 3个标准差内
        
        return confidence
    
    def get_crowdedness_indicator(self) -> Dict[str, Any]:
        """获取拥挤度指标"""
        if len(self._history) < 10:
            return {'status': '数据不足'}
        
        rates = [r for _, r in self._history[-100:]]
        current = rates[-1]
        avg = np.mean(rates[:-1])
        
        # 拥挤度评分 (-100 到 100)
        # 正值 = 多头拥挤，负值 = 空头拥挤
        crowdedness = (current - avg) / (abs(avg) + 0.0001) * 50
        crowdedness = max(-100, min(100, crowdedness))
        
        return {
            'current_rate': current,
            'average_rate': avg,
            'crowdedness_score': crowdedness,
            'interpretation': '多头拥挤' if crowdedness > 30 else ('空头拥挤' if crowdedness < -30 else '均衡')
        }


# 全局监控器
_funding_monitor: Optional[FundingRateMonitor] = None


def get_funding_monitor() -> FundingRateMonitor:
    """获取全局资金费率监控器"""
    global _funding_monitor
    if _funding_monitor is None:
        _funding_monitor = FundingRateMonitor()
    return _funding_monitor


def analyze_funding_rate(rate: float) -> Dict[str, Any]:
    """快速分析资金费率"""
    monitor = get_funding_monitor()
    result = monitor.analyze(rate)
    
    return {
        'rate': result.current_rate,
        'avg': result.historical_avg,
        'percentile': result.percentile,
        'signal': result.signal.value,
        'crowd_bias': result.crowd_bias,
        'warning_level': result.warning_level,
        'description': result.description,
        'confidence': result.confidence
    }
