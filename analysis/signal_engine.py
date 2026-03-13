"""
统一信号引擎 (Signal Engine)
============================
核心功能：统一所有模块信号，实现 Meta-Labeling 元决策
- 信号聚合 (Signal Aggregation)
- 置信度加权投票
- Meta Filter 信号过滤
- 统一决策输出
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """信号强度"""
    VERY_STRONG = 4
    STRONG = 3
    MODERATE = 2
    WEAK = 1
    NONE = 0


class SignalDirection(Enum):
    """信号方向"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class ModuleSignal:
    """模块信号"""
    module_name: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float  # 0-1
    reason: str = ""
    weight: float = 0.05  # 模块权重，默认值
    
    # 详细数据
    raw_value: float = 0.0
    threshold_used: float = 0.0


@dataclass
class AggregatedSignal:
    """聚合信号"""
    timestamp: datetime
    
    # 最终决策
    final_direction: SignalDirection
    final_confidence: float
    final_strength: SignalStrength
    
    # 概率分布
    long_probability: float
    short_probability: float
    hold_probability: float
    
    # 模块投票结果
    long_votes: int
    short_votes: int
    neutral_votes: int
    
    # 加权得分
    weighted_long_score: float
    weighted_short_score: float
    
    # 信号质量（保留核心指标）
    signal_consistency: float  # 0-1，信号一致程度
    signal_quality: float  # 0-1，信号质量
    
    # 模块详情
    module_signals: List[ModuleSignal] = field(default_factory=list)


class SignalAggregator:
    """
    信号聚合器
    
    功能：
    1. 收集所有模块信号
    2. 加权投票
    3. 计算信号一致性
    """
    
    # 模块权重配置
    MODULE_WEIGHTS = {
        # 技术面 (40%)
        "hurst": 0.10,
        "fractal": 0.05,
        "momentum": 0.10,
        "volatility": 0.05,
        "orderbook": 0.10,
        
        # 链上/资金面 (25%)
        "whale_flow": 0.10,
        "funding_rate": 0.08,
        "liquidation": 0.07,
        
        # 情绪面 (15%)
        "social_sentiment": 0.08,
        "market_regime": 0.07,
        
        # 订单流 (20%)
        "order_flow": 0.15,
        "liquidity": 0.05,
    }
    
    def __init__(self, custom_weights: Dict[str, float] = None):
        self.weights = {**self.MODULE_WEIGHTS, **(custom_weights or {})}
        self.signals: List[ModuleSignal] = []
    
    def add_signal(self, signal: ModuleSignal) -> None:
        """添加模块信号"""
        # 应用权重
        signal.weight = self.weights.get(signal.module_name, 0.05)
        self.signals.append(signal)
    
    def calculate_weighted_vote(self) -> Tuple[float, float, float]:
        """
        计算加权投票结果
        
        Returns:
            (long_score, short_score, neutral_score)
        """
        long_score = 0.0
        short_score = 0.0
        neutral_score = 0.0
        
        for signal in self.signals:
            weighted_strength = signal.strength.value * signal.weight * signal.confidence
            
            if signal.direction == SignalDirection.LONG:
                long_score += weighted_strength
            elif signal.direction == SignalDirection.SHORT:
                short_score += weighted_strength
            else:
                neutral_score += weighted_strength
        
        return long_score, short_score, neutral_score
    
    def calculate_consistency(self) -> float:
        """
        计算信号一致性
        
        高一致性 = 大部分模块指向同一方向
        """
        if not self.signals:
            return 0.0
        
        directions = [s.direction for s in self.signals]
        
        long_count = directions.count(SignalDirection.LONG)
        short_count = directions.count(SignalDirection.SHORT)
        neutral_count = directions.count(SignalDirection.NEUTRAL)
        
        total = len(directions)
        max_count = max(long_count, short_count, neutral_count)
        
        consistency = max_count / total
        
        return consistency
    
    def calculate_signal_quality(self) -> float:
        """
        计算信号质量
        
        考虑因素：
        1. 信号一致性
        2. 平均置信度
        3. 信号强度
        """
        if not self.signals:
            return 0.0
        
        consistency = self.calculate_consistency()
        avg_confidence = np.mean([s.confidence for s in self.signals])
        avg_strength = np.mean([s.strength.value for s in self.signals]) / 4  # 归一化
        
        quality = consistency * 0.4 + avg_confidence * 0.35 + avg_strength * 0.25
        
        return quality
    
    def get_vote_counts(self) -> Tuple[int, int, int]:
        """获取投票计数"""
        long_votes = sum(1 for s in self.signals if s.direction == SignalDirection.LONG)
        short_votes = sum(1 for s in self.signals if s.direction == SignalDirection.SHORT)
        neutral_votes = sum(1 for s in self.signals if s.direction == SignalDirection.NEUTRAL)
        
        return long_votes, short_votes, neutral_votes


class SignalEngine:
    """
    统一信号引擎（简化版）
    
    核心架构：
    模块信号 → Signal Aggregator → 最终决策
    
    注意：Meta Filter 已移除，过滤逻辑整合到 UnifiedFilter
    """
    
    def __init__(self):
        self.aggregator = SignalAggregator()
        # 移除: self.meta_filter = MetaFilter()
        
        # 历史记录
        self.signal_history: List[AggregatedSignal] = []
        self.max_history = 100
    
    def process_technical_signals(
        self,
        hurst: float,
        fractal_dim: float,
        momentum: float,
        volatility: float,
        rsi: float = 50,
        macd: float = 0,
    ) -> None:
        """处理技术面信号"""
        
        # Hurst 信号
        if hurst > 0.55:
            self.aggregator.add_signal(ModuleSignal(
                module_name="hurst",
                direction=SignalDirection.LONG,
                strength=SignalStrength.STRONG if hurst > 0.65 else SignalStrength.MODERATE,
                confidence=min(1, (hurst - 0.5) * 3),
                reason=f"Hurst={hurst:.3f} 趋势向上",
                raw_value=hurst,
            ))
        elif hurst < 0.45:
            self.aggregator.add_signal(ModuleSignal(
                module_name="hurst",
                direction=SignalDirection.SHORT,
                strength=SignalStrength.STRONG if hurst < 0.35 else SignalStrength.MODERATE,
                confidence=min(1, (0.5 - hurst) * 3),
                reason=f"Hurst={hurst:.3f} 均值回归",
                raw_value=hurst,
            ))
        else:
            self.aggregator.add_signal(ModuleSignal(
                module_name="hurst",
                direction=SignalDirection.NEUTRAL,
                strength=SignalStrength.WEAK,
                confidence=0.3,
                reason=f"Hurst={hurst:.3f} 随机游走",
                raw_value=hurst,
            ))
        
        # 动量信号
        if abs(momentum) > 3:
            self.aggregator.add_signal(ModuleSignal(
                module_name="momentum",
                direction=SignalDirection.LONG if momentum > 0 else SignalDirection.SHORT,
                strength=SignalStrength.STRONG if abs(momentum) > 5 else SignalStrength.MODERATE,
                confidence=min(1, abs(momentum) / 10),
                reason=f"动量={momentum:.2f}",
                raw_value=momentum,
            ))
        
        # RSI 信号
        if rsi > 70:
            self.aggregator.add_signal(ModuleSignal(
                module_name="rsi",
                direction=SignalDirection.SHORT,
                strength=SignalStrength.MODERATE,
                confidence=(rsi - 70) / 30,
                reason=f"RSI={rsi:.1f} 超买",
                raw_value=rsi,
            ))
        elif rsi < 30:
            self.aggregator.add_signal(ModuleSignal(
                module_name="rsi",
                direction=SignalDirection.LONG,
                strength=SignalStrength.MODERATE,
                confidence=(30 - rsi) / 30,
                reason=f"RSI={rsi:.1f} 超卖",
                raw_value=rsi,
            ))
    
    def process_onchain_signals(
        self,
        whale_net_flow: float,
        funding_rate: float,
        liquidation_imbalance: float,
    ) -> None:
        """处理链上/资金面信号"""
        
        # 巨鲸流动
        if whale_net_flow < -500:
            self.aggregator.add_signal(ModuleSignal(
                module_name="whale_flow",
                direction=SignalDirection.LONG,
                strength=SignalStrength.STRONG,
                confidence=min(1, abs(whale_net_flow) / 1000),
                reason=f"巨鲸净流出{abs(whale_net_flow):.0f} ETH",
                raw_value=whale_net_flow,
            ))
        elif whale_net_flow > 500:
            self.aggregator.add_signal(ModuleSignal(
                module_name="whale_flow",
                direction=SignalDirection.SHORT,
                strength=SignalStrength.STRONG,
                confidence=min(1, whale_net_flow / 1000),
                reason=f"巨鲸净流入{whale_net_flow:.0f} ETH",
                raw_value=whale_net_flow,
            ))
        
        # 资金费率
        if abs(funding_rate) > 0.0005:
            # 高费率 = 做多拥挤 = 看空
            direction = SignalDirection.SHORT if funding_rate > 0 else SignalDirection.LONG
            self.aggregator.add_signal(ModuleSignal(
                module_name="funding_rate",
                direction=direction,
                strength=SignalStrength.MODERATE,
                confidence=min(1, abs(funding_rate) * 1000),
                reason=f"资金费率={funding_rate*100:.4f}%",
                raw_value=funding_rate,
            ))
    
    def process_orderbook_signals(
        self,
        imbalance: float,
        cvd: float,
        delta: float,
    ) -> None:
        """处理订单流信号"""
        
        # 订单簿失衡
        if abs(imbalance) > 0.3:
            direction = SignalDirection.LONG if imbalance > 0 else SignalDirection.SHORT
            self.aggregator.add_signal(ModuleSignal(
                module_name="orderbook",
                direction=direction,
                strength=SignalStrength.STRONG if abs(imbalance) > 0.5 else SignalStrength.MODERATE,
                confidence=abs(imbalance),
                reason=f"订单簿失衡={imbalance:.2f}",
                raw_value=imbalance,
            ))
        
        # CVD
        if abs(cvd) > 10:
            direction = SignalDirection.LONG if cvd > 0 else SignalDirection.SHORT
            self.aggregator.add_signal(ModuleSignal(
                module_name="order_flow",
                direction=direction,
                strength=SignalStrength.MODERATE,
                confidence=min(1, abs(cvd) / 50),
                reason=f"CVD={cvd:.2f}",
                raw_value=cvd,
            ))
    
    def process_sentiment_signals(
        self,
        social_score: float,
        regime: str,
        regime_confidence: float,
    ) -> None:
        """处理情绪/状态信号"""
        
        # 社交情绪
        if social_score > 70:
            self.aggregator.add_signal(ModuleSignal(
                module_name="social_sentiment",
                direction=SignalDirection.SHORT,  # 反向
                strength=SignalStrength.WEAK,
                confidence=(social_score - 50) / 50,
                reason=f"情绪过热={social_score:.1f}",
                raw_value=social_score,
            ))
        elif social_score < 30:
            self.aggregator.add_signal(ModuleSignal(
                module_name="social_sentiment",
                direction=SignalDirection.LONG,  # 反向
                strength=SignalStrength.WEAK,
                confidence=(50 - social_score) / 50,
                reason=f"情绪恐慌={social_score:.1f}",
                raw_value=social_score,
            ))
        
        # 市场状态
        if "trend_up" in regime:
            self.aggregator.add_signal(ModuleSignal(
                module_name="market_regime",
                direction=SignalDirection.LONG,
                strength=SignalStrength.STRONG,
                confidence=regime_confidence,
                reason=f"市场状态={regime}",
                raw_value=1 if "up" in regime else -1,
            ))
        elif "trend_down" in regime:
            self.aggregator.add_signal(ModuleSignal(
                module_name="market_regime",
                direction=SignalDirection.SHORT,
                strength=SignalStrength.STRONG,
                confidence=regime_confidence,
                reason=f"市场状态={regime}",
                raw_value=-1,
            ))
    
    def generate_final_signal(
        self,
        regime: str = "neutral",
        risk_score: float = 30,
    ) -> AggregatedSignal:
        """
        生成最终信号
        """
        # 计算加权投票
        long_score, short_score, neutral_score = self.aggregator.calculate_weighted_vote()
        
        # 计算概率分布
        total_score = long_score + short_score + neutral_score + 0.001
        
        # 调整：如果neutral_score高，增加hold概率
        if neutral_score > max(long_score, short_score):
            hold_adjustment = neutral_score / total_score
        else:
            hold_adjustment = 0.3  # 基础观望概率
        
        long_prob = long_score / total_score * (1 - hold_adjustment) * 100
        short_prob = short_score / total_score * (1 - hold_adjustment) * 100
        hold_prob = max(20, hold_adjustment * 100)  # 至少20%观望
        
        # 归一化
        total_prob = long_prob + short_prob + hold_prob
        long_prob = long_prob / total_prob * 100
        short_prob = short_prob / total_prob * 100
        hold_prob = hold_prob / total_prob * 100
        
        # 确定方向
        if long_prob > short_prob and long_prob > hold_prob:
            direction = SignalDirection.LONG
            confidence = long_prob
        elif short_prob > long_prob and short_prob > hold_prob:
            direction = SignalDirection.SHORT
            confidence = short_prob
        else:
            direction = SignalDirection.NEUTRAL
            confidence = hold_prob
        
        # 计算信号强度
        if confidence > 70:
            strength = SignalStrength.VERY_STRONG
        elif confidence > 55:
            strength = SignalStrength.STRONG
        elif confidence > 45:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # 计算一致性和质量
        consistency = self.aggregator.calculate_consistency()
        quality = self.aggregator.calculate_signal_quality()
        
        # 简化：不再应用 Meta Filter
        # 过滤逻辑已移至 UnifiedFilter
        final_confidence = confidence
        
        # 获取投票计数
        long_votes, short_votes, neutral_votes = self.aggregator.get_vote_counts()
        
        signal = AggregatedSignal(
            timestamp=datetime.now(),
            final_direction=direction,
            final_confidence=final_confidence,
            final_strength=strength,
            long_probability=long_prob,
            short_probability=short_prob,
            hold_probability=hold_prob,
            long_votes=long_votes,
            short_votes=short_votes,
            neutral_votes=neutral_votes,
            weighted_long_score=long_score,
            weighted_short_score=short_score,
            signal_consistency=consistency,
            signal_quality=quality,
            module_signals=list(self.aggregator.signals),
        )
        
        # 保存历史
        self.signal_history.append(signal)
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]
        
        # 清空聚合器为下次使用
        self.aggregator.signals.clear()
        
        return signal
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """获取信号摘要"""
        if not self.signal_history:
            return {"status": "no_signals"}
        
        latest = self.signal_history[-1]
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "direction": latest.final_direction.value,
            "confidence": round(latest.final_confidence, 1),
            "strength": latest.final_strength.name,
            "probabilities": {
                "long": round(latest.long_probability, 1),
                "short": round(latest.short_probability, 1),
                "hold": round(latest.hold_probability, 1),
            },
            "votes": {
                "long": latest.long_votes,
                "short": latest.short_votes,
                "neutral": latest.neutral_votes,
            },
            "quality_metrics": {
                "consistency": round(latest.signal_consistency, 2),
                "quality": round(latest.signal_quality, 2),
            },
            "module_count": len(latest.module_signals),
        }


# 全局实例
_signal_engine: Optional[SignalEngine] = None


def get_signal_engine() -> SignalEngine:
    """获取全局信号引擎"""
    global _signal_engine
    if _signal_engine is None:
        _signal_engine = SignalEngine()
    return _signal_engine


def generate_unified_signal(
    hurst: float = 0.5,
    momentum: float = 0,
    imbalance: float = 0,
    whale_flow: float = 0,
    funding_rate: float = 0,
    cvd: float = 0,
    sentiment: float = 50,
    regime: str = "neutral",
    regime_confidence: float = 0.5,
    risk_score: float = 30,
) -> Dict[str, Any]:
    """
    生成统一信号（便捷函数）
    
    Returns:
        信号摘要
    """
    engine = get_signal_engine()
    
    # 处理各类信号
    engine.process_technical_signals(
        hurst=hurst,
        fractal_dim=1.5,
        momentum=momentum,
        volatility=0.02,
    )
    
    engine.process_onchain_signals(
        whale_net_flow=whale_flow,
        funding_rate=funding_rate,
        liquidation_imbalance=0,
    )
    
    engine.process_orderbook_signals(
        imbalance=imbalance,
        cvd=cvd,
        delta=cvd,
    )
    
    engine.process_sentiment_signals(
        social_score=sentiment,
        regime=regime,
        regime_confidence=regime_confidence,
    )
    
    # 生成最终信号
    signal = engine.generate_final_signal(
        regime=regime,
        risk_score=risk_score,
    )
    
    return engine.get_signal_summary()
