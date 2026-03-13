"""
多时间框架共振分析 (Multi-Timeframe Analysis)
=============================================
引入多周期数据，减少假信号，提升胜率

核心逻辑：
- 5分钟看多 + 15分钟看多 + 1小时看多 → 强烈做多信号
- 5分钟看多 + 15分钟看跌 → 降低信号权重或放弃
- 多周期共振 → 提高信号置信度
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimeframeDirection(Enum):
    """时间框架方向"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TimeframeSignal:
    """时间框架信号"""
    timeframe: str          # "1m", "5m", "15m", "1h", "4h"
    direction: TimeframeDirection
    strength: float         # 0-1 信号强度
    trend_score: float      # 趋势分数 -1 到 1
    momentum: float         # 动量
    cvd_direction: str      # CVD方向
    confidence: float       # 信号置信度
    
    def to_dict(self) -> Dict:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "strength": round(self.strength, 2),
            "trend_score": round(self.trend_score, 2),
            "momentum": round(self.momentum, 4),
            "cvd_direction": self.cvd_direction,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class MultiTimeframeResult:
    """多时间框架分析结果"""
    signals: List[TimeframeSignal]
    overall_direction: TimeframeDirection
    resonance_score: float      # 共振分数 0-1
    alignment_count: int        # 一致方向数量
    total_timeframes: int
    confidence_multiplier: float  # 置信度乘数
    conflict_detected: bool
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "signals": [s.to_dict() for s in self.signals],
            "overall_direction": self.overall_direction.value,
            "resonance_score": round(self.resonance_score, 2),
            "alignment_count": self.alignment_count,
            "total_timeframes": self.total_timeframes,
            "confidence_multiplier": round(self.confidence_multiplier, 2),
            "conflict_detected": self.conflict_detected,
            "details": self.details,
        }


class MultiTimeframeAnalyzer:
    """
    多时间框架分析器
    
    共振规则：
    ┌─────────────────────────────────────────────────────────────┐
    │ 共振类型        │ 条件                    │ 置信度乘数      │
    ├─────────────────────────────────────────────────────────────┤
    │ 强共振          │ 全部周期同向            │ 1.5            │
    │ 中等共振        │ 3/4周期同向             │ 1.2            │
    │ 弱共振          │ 2/4周期同向             │ 1.0            │
    │ 冲突            │ 存在反向信号            │ 0.5-0.7        │
    │ 强冲突          │ 主要周期反向            │ 0.3 (放弃信号) │
    └─────────────────────────────────────────────────────────────┘
    """
    
    # 时间框架权重（越高越重要）
    TIMEFRAME_WEIGHTS = {
        "1h": 0.35,    # 1小时最重要
        "15m": 0.30,   # 15分钟次之
        "5m": 0.25,    # 5分钟（交易周期）
        "1m": 0.10,    # 1分钟辅助
    }
    
    def __init__(self):
        self._history: List[MultiTimeframeResult] = []
    
    def analyze_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> TimeframeSignal:
        """
        分析单个时间框架
        
        Args:
            df: 该时间框架的OHLCV数据
            timeframe: 时间框架名称
        """
        if len(df) < 20:
            return TimeframeSignal(
                timeframe=timeframe,
                direction=TimeframeDirection.NEUTRAL,
                strength=0,
                trend_score=0,
                momentum=0,
                cvd_direction="neutral",
                confidence=0,
            )
        
        close = df['close'].values
        
        # 趋势分析（EMA）
        ema9 = pd.Series(close).ewm(span=9).mean().values
        ema21 = pd.Series(close).ewm(span=21).mean().values
        
        trend_score = (ema9[-1] - ema21[-1]) / ema21[-1] if ema21[-1] > 0 else 0
        
        # 动量（RSI风格）
        delta = np.diff(close)
        gain = np.mean(delta[delta > 0]) if len(delta[delta > 0]) > 0 else 0
        loss = np.mean(np.abs(delta[delta < 0])) if len(delta[delta < 0]) > 0 else 0
        momentum = gain / (gain + loss) if (gain + loss) > 0 else 0.5
        
        # 方向判断
        if trend_score > 0.005 and momentum > 0.55:
            direction = TimeframeDirection.BULLISH
            strength = min(1, trend_score * 50 + (momentum - 0.5) * 2)
        elif trend_score < -0.005 and momentum < 0.45:
            direction = TimeframeDirection.BEARISH
            strength = min(1, abs(trend_score) * 50 + (0.5 - momentum) * 2)
        else:
            direction = TimeframeDirection.NEUTRAL
            strength = 0.3
        
        # CVD方向（模拟）
        if 'volume' in df.columns:
            vol_trend = df['volume'].tail(5).mean() / df['volume'].tail(20).mean() if df['volume'].tail(20).mean() > 0 else 1
            cvd_direction = "bullish" if direction == TimeframeDirection.BULLISH and vol_trend > 1 else \
                           "bearish" if direction == TimeframeDirection.BEARISH and vol_trend > 1 else "neutral"
        else:
            cvd_direction = "neutral"
        
        # 置信度
        confidence = strength * 0.7 + (0.3 if direction != TimeframeDirection.NEUTRAL else 0)
        
        return TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            trend_score=trend_score,
            momentum=momentum,
            cvd_direction=cvd_direction,
            confidence=confidence,
        )
    
    def analyze_all(
        self,
        data_by_timeframe: Dict[str, pd.DataFrame],
        primary_timeframe: str = "5m",
    ) -> MultiTimeframeResult:
        """
        分析所有时间框架并检测共振
        
        Args:
            data_by_timeframe: {"1m": df, "5m": df, "15m": df, "1h": df}
            primary_timeframe: 主交易周期
        """
        signals = []
        
        for tf, df in data_by_timeframe.items():
            signal = self.analyze_timeframe(df, tf)
            signals.append(signal)
        
        # 计算共振
        weighted_bullish = 0
        weighted_bearish = 0
        total_weight = 0
        
        for signal in signals:
            weight = self.TIMEFRAME_WEIGHTS.get(signal.timeframe, 0.1)
            total_weight += weight
            
            if signal.direction == TimeframeDirection.BULLISH:
                weighted_bullish += weight * signal.strength
            elif signal.direction == TimeframeDirection.BEARISH:
                weighted_bearish += weight * signal.strength
        
        # 确定总体方向
        if weighted_bullish > weighted_bearish * 1.5:
            overall_direction = TimeframeDirection.BULLISH
            resonance_score = weighted_bullish / total_weight
        elif weighted_bearish > weighted_bullish * 1.5:
            overall_direction = TimeframeDirection.BEARISH
            resonance_score = weighted_bearish / total_weight
        else:
            overall_direction = TimeframeDirection.NEUTRAL
            resonance_score = 0.5
        
        # 计算一致性数量
        bullish_count = sum(1 for s in signals if s.direction == TimeframeDirection.BULLISH)
        bearish_count = sum(1 for s in signals if s.direction == TimeframeDirection.BEARISH)
        
        if overall_direction == TimeframeDirection.BULLISH:
            alignment_count = bullish_count
        elif overall_direction == TimeframeDirection.BEARISH:
            alignment_count = bearish_count
        else:
            alignment_count = max(bullish_count, bearish_count)
        
        # 检测冲突
        conflict_detected = bullish_count > 0 and bearish_count > 0
        
        # 计算置信度乘数
        if alignment_count == len(signals):
            # 强共振：全部同向
            confidence_multiplier = 1.5
        elif alignment_count >= len(signals) * 0.75:
            # 中等共振
            confidence_multiplier = 1.2
        elif alignment_count >= len(signals) * 0.5:
            # 弱共振
            confidence_multiplier = 1.0
        elif conflict_detected:
            # 存在冲突
            primary_signal = next((s for s in signals if s.timeframe == primary_timeframe), None)
            if primary_signal:
                # 检查主周期与更高周期是否冲突
                higher_tf_signal = next((s for s in signals if s.timeframe in ["1h", "15m"]), None)
                if higher_tf_signal and primary_signal.direction != higher_tf_signal.direction:
                    confidence_multiplier = 0.3  # 强冲突
                else:
                    confidence_multiplier = 0.7  # 一般冲突
            else:
                confidence_multiplier = 0.5
        else:
            confidence_multiplier = 0.8
        
        result = MultiTimeframeResult(
            signals=signals,
            overall_direction=overall_direction,
            resonance_score=resonance_score,
            alignment_count=alignment_count,
            total_timeframes=len(signals),
            confidence_multiplier=confidence_multiplier,
            conflict_detected=conflict_detected,
            details={
                "weighted_bullish": round(weighted_bullish, 3),
                "weighted_bearish": round(weighted_bearish, 3),
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "resonance_level": self._get_resonance_level(alignment_count, len(signals)),
            }
        )
        
        self._history.append(result)
        if len(self._history) > 100:
            self._history = self._history[-100:]
        
        return result
    
    def _get_resonance_level(self, alignment: int, total: int) -> str:
        """获取共振级别描述"""
        ratio = alignment / total if total > 0 else 0
        if ratio >= 1.0:
            return "强共振"
        elif ratio >= 0.75:
            return "中等共振"
        elif ratio >= 0.5:
            return "弱共振"
        else:
            return "分歧"
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self._history:
            return {"total": 0}
        
        recent = self._history[-20:]
        return {
            "total": len(self._history),
            "avg_resonance": np.mean([r.resonance_score for r in recent]),
            "conflict_rate": sum(1 for r in recent if r.conflict_detected) / len(recent),
            "bullish_rate": sum(1 for r in recent if r.overall_direction == TimeframeDirection.BULLISH) / len(recent),
            "bearish_rate": sum(1 for r in recent if r.overall_direction == TimeframeDirection.BEARISH) / len(recent),
        }


# 全局实例
_analyzer: Optional[MultiTimeframeAnalyzer] = None


def get_analyzer() -> MultiTimeframeAnalyzer:
    """获取分析器"""
    global _analyzer
    if _analyzer is None:
        _analyzer = MultiTimeframeAnalyzer()
    return _analyzer


def analyze_multi_timeframe(data_by_timeframe: Dict[str, pd.DataFrame]) -> MultiTimeframeResult:
    """多时间框架分析（便捷函数）"""
    return get_analyzer().analyze_all(data_by_timeframe)
