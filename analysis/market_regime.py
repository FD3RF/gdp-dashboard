"""
市场状态识别引擎 (Market Regime Engine)
=====================================
核心模块：在AI预测前先判断市场结构
不同市场状态下，策略权重不同
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态类型"""
    TREND_UP = "trend_up"           # 强势上涨趋势
    TREND_DOWN = "trend_down"       # 强势下跌趋势
    RANGE = "range"                 # 震荡区间
    VOLATILE = "volatile"           # 高波动状态
    LIQUIDATION = "liquidation"     # 清算事件
    ACCUMULATION = "accumulation"   # 吸筹阶段
    DISTRIBUTION = "distribution"   # 派发阶段
    PANIC = "panic"                 # 恐慌状态
    EUPHORIA = "euphoria"           # 狂热状态
    NEUTRAL = "neutral"             # 中性/不明


@dataclass
class RegimeState:
    """市场状态"""
    regime: MarketRegime
    confidence: float  # 0-1
    duration_minutes: int  # 状态持续时间
    
    # 状态特征
    trend_strength: float  # 趋势强度
    volatility_level: float  # 波动率水平
    momentum_direction: float  # 动量方向
    
    # 策略建议
    recommended_strategies: List[str]
    avoided_strategies: List[str]
    
    # 风险提示
    risk_factors: List[str]
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyWeights:
    """策略权重配置"""
    trend_follow: float = 0.0      # 趋势跟随
    mean_reversion: float = 0.0    # 均值回归
    volatility: float = 0.0        # 波动率策略
    liquidity: float = 0.0         # 流动性策略
    momentum: float = 0.0          # 动量策略
    sentiment: float = 0.0         # 情绪策略
    contrarian: float = 0.0        # 反向策略


# 不同市场状态下的策略权重
REGIME_STRATEGY_WEIGHTS = {
    MarketRegime.TREND_UP: {
        "trend_follow": 0.4,
        "momentum": 0.3,
        "mean_reversion": 0.0,
        "volatility": 0.1,
        "sentiment": 0.1,
        "contrarian": 0.0,
    },
    MarketRegime.TREND_DOWN: {
        "trend_follow": 0.4,
        "momentum": 0.3,
        "mean_reversion": 0.0,
        "volatility": 0.1,
        "sentiment": 0.1,
        "contrarian": 0.0,
    },
    MarketRegime.RANGE: {
        "trend_follow": 0.0,
        "momentum": 0.1,
        "mean_reversion": 0.5,
        "volatility": 0.2,
        "sentiment": 0.1,
        "contrarian": 0.1,
    },
    MarketRegime.VOLATILE: {
        "trend_follow": 0.1,
        "momentum": 0.1,
        "mean_reversion": 0.1,
        "volatility": 0.4,
        "sentiment": 0.1,
        "contrarian": 0.2,
    },
    MarketRegime.LIQUIDATION: {
        "trend_follow": 0.3,
        "momentum": 0.2,
        "mean_reversion": 0.0,
        "volatility": 0.3,
        "sentiment": 0.0,
        "contrarian": 0.2,
    },
    MarketRegime.ACCUMULATION: {
        "trend_follow": 0.1,
        "momentum": 0.1,
        "mean_reversion": 0.3,
        "volatility": 0.2,
        "sentiment": 0.1,
        "contrarian": 0.2,
    },
    MarketRegime.DISTRIBUTION: {
        "trend_follow": 0.1,
        "momentum": 0.1,
        "mean_reversion": 0.3,
        "volatility": 0.2,
        "sentiment": 0.2,
        "contrarian": 0.1,
    },
    MarketRegime.PANIC: {
        "trend_follow": 0.2,
        "momentum": 0.1,
        "mean_reversion": 0.0,
        "volatility": 0.3,
        "sentiment": 0.0,
        "contrarian": 0.4,
    },
    MarketRegime.EUPHORIA: {
        "trend_follow": 0.1,
        "momentum": 0.2,
        "mean_reversion": 0.1,
        "volatility": 0.2,
        "sentiment": 0.1,
        "contrarian": 0.3,
    },
    MarketRegime.NEUTRAL: {
        "trend_follow": 0.2,
        "momentum": 0.2,
        "mean_reversion": 0.2,
        "volatility": 0.2,
        "sentiment": 0.1,
        "contrarian": 0.1,
    },
}


class MarketRegimeEngine:
    """
    市场状态识别引擎
    
    核心功能：
    1. 识别当前市场状态
    2. 计算策略权重
    3. 提供风险提示
    
    输入特征：
    - Hurst指数 (趋势/均值回归)
    - 波动率 (ATR)
    - 订单簿失衡
    - 资金费率
    - 社交情绪
    - 巨鲸流入
    - Fractal维度
    - 动量
    """
    
    # 阈值配置
    THRESHOLDS = {
        # Hurst阈值
        "hurst_trend": 0.55,      # > 此值 = 趋势
        "hurst_mean_revert": 0.45, # < 此值 = 均值回归
        
        # 波动率阈值 (相对历史)
        "vol_high": 1.5,          # 高波动
        "vol_extreme": 2.5,       # 极端波动
        
        # 订单簿失衡
        "imbalance_strong": 0.3,  # 强失衡
        "imbalance_extreme": 0.5, # 极端失衡
        
        # 资金费率
        "funding_high": 0.0005,   # 高费率
        "funding_extreme": 0.001, # 极端费率
        
        # 情绪
        "sentiment_euphoria": 75, # 狂热
        "sentiment_panic": 25,    # 恐慌
        
        # 巨鲸流入
        "whale_significant": 500, # 显著流入 (ETH)
    }
    
    def __init__(self):
        self.history: List[RegimeState] = []
        self.max_history = 100
    
    def detect_regime(
        self,
        hurst: float,
        volatility: float,
        avg_volatility: float,
        orderbook_imbalance: float,
        funding_rate: float,
        sentiment_score: float,
        whale_net_flow: float = 0,
        fractal_dim: float = 1.5,
        momentum: float = 0,
        price_change_pct: float = 0,
        volume_ratio: float = 1.0,
    ) -> RegimeState:
        """
        识别市场状态
        
        Args:
            hurst: Hurst指数 (0-1)
            volatility: 当前波动率
            avg_volatility: 平均波动率
            orderbook_imbalance: 订单簿失衡 (-1 到 1)
            funding_rate: 资金费率
            sentiment_score: 情绪分数 (0-100)
            whale_net_flow: 巨鲸净流入 (ETH)
            fractal_dim: 分形维度 (1-2)
            momentum: 动量
            price_change_pct: 价格变化百分比
            volume_ratio: 成交量比率
        
        Returns:
            RegimeState
        """
        # 计算特征
        vol_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
        trend_strength = self._calculate_trend_strength(hurst, momentum, orderbook_imbalance)
        vol_level = self._calculate_volatility_level(vol_ratio, fractal_dim)
        mom_direction = self._calculate_momentum_direction(momentum, price_change_pct, orderbook_imbalance)
        
        # 识别状态
        regime, confidence = self._classify_regime(
            hurst=hurst,
            vol_ratio=vol_ratio,
            imbalance=orderbook_imbalance,
            funding_rate=funding_rate,
            sentiment=sentiment_score,
            whale_flow=whale_net_flow,
            momentum=momentum,
            price_change=price_change_pct,
            volume_ratio=volume_ratio,
        )
        
        # 计算持续时间
        duration = self._calculate_duration(regime)
        
        # 策略建议
        strategies, avoided = self._get_strategy_recommendations(regime)
        
        # 风险因素
        risks = self._identify_risks(
            regime=regime,
            vol_ratio=vol_ratio,
            funding_rate=funding_rate,
            sentiment=sentiment_score,
            whale_flow=whale_net_flow,
        )
        
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            duration_minutes=duration,
            trend_strength=trend_strength,
            volatility_level=vol_level,
            momentum_direction=mom_direction,
            recommended_strategies=strategies,
            avoided_strategies=avoided,
            risk_factors=risks,
        )
        
        # 记录历史
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return state
    
    def _calculate_trend_strength(
        self,
        hurst: float,
        momentum: float,
        imbalance: float
    ) -> float:
        """计算趋势强度 (0-1)"""
        # Hurst贡献
        hurst_score = abs(hurst - 0.5) * 2  # 偏离0.5的程度
        
        # 动量贡献
        mom_score = min(1, abs(momentum) / 10)
        
        # 订单簿贡献
        imb_score = abs(imbalance)
        
        # 综合评分
        strength = hurst_score * 0.5 + mom_score * 0.3 + imb_score * 0.2
        
        return min(1, strength)
    
    def _calculate_volatility_level(self, vol_ratio: float, fractal_dim: float) -> float:
        """计算波动率水平 (0-1)"""
        vol_score = min(1, (vol_ratio - 1) / 2)
        fractal_score = (fractal_dim - 1) / 1  # 1-2映射到0-1
        
        return vol_score * 0.7 + fractal_score * 0.3
    
    def _calculate_momentum_direction(
        self,
        momentum: float,
        price_change: float,
        imbalance: float
    ) -> float:
        """计算动量方向 (-1 到 1)"""
        # 正值 = 上涨动量，负值 = 下跌动量
        mom_dir = np.sign(momentum) * min(1, abs(momentum) / 5)
        price_dir = np.sign(price_change) * min(1, abs(price_change) / 2)
        imb_dir = imbalance
        
        # 综合方向
        direction = mom_dir * 0.4 + price_dir * 0.3 + imb_dir * 0.3
        
        return max(-1, min(1, direction))
    
    def _classify_regime(
        self,
        hurst: float,
        vol_ratio: float,
        imbalance: float,
        funding_rate: float,
        sentiment: float,
        whale_flow: float,
        momentum: float,
        price_change: float,
        volume_ratio: float,
    ) -> Tuple[MarketRegime, float]:
        """
        分类市场状态
        
        Returns:
            (regime, confidence)
        """
        scores = {}  # 各状态得分
        
        # === 趋势状态 ===
        if hurst > self.THRESHOLDS["hurst_trend"]:
            if momentum > 0 or imbalance > 0.1:
                scores[MarketRegime.TREND_UP] = (hurst - 0.5) * 2 + 0.3
            elif momentum < 0 or imbalance < -0.1:
                scores[MarketRegime.TREND_DOWN] = (hurst - 0.5) * 2 + 0.3
        
        # === 震荡状态 ===
        if hurst < self.THRESHOLDS["hurst_mean_revert"]:
            scores[MarketRegime.RANGE] = (0.5 - hurst) * 2 + 0.2
        
        # === 高波动状态 ===
        if vol_ratio > self.THRESHOLDS["vol_high"]:
            scores[MarketRegime.VOLATILE] = (vol_ratio - 1) / 2 + 0.3
        
        # === 恐慌/狂热状态 ===
        if sentiment < self.THRESHOLDS["sentiment_panic"]:
            scores[MarketRegime.PANIC] = (25 - sentiment) / 25 * 0.8 + 0.2
        elif sentiment > self.THRESHOLDS["sentiment_euphoria"]:
            scores[MarketRegime.EUPHORIA] = (sentiment - 75) / 25 * 0.8 + 0.2
        
        # === 清算事件 ===
        if vol_ratio > self.THRESHOLDS["vol_extreme"]:
            if abs(funding_rate) > self.THRESHOLDS["funding_extreme"]:
                scores[MarketRegime.LIQUIDATION] = 0.7
            elif volume_ratio > 3:
                scores[MarketRegime.LIQUIDATION] = 0.6
        
        # === 吸筹/派发 ===
        if abs(whale_flow) > self.THRESHOLDS["whale_significant"]:
            if whale_flow < 0 and hurst < 0.55:  # 流出 + 非强趋势
                scores[MarketRegime.ACCUMULATION] = min(abs(whale_flow) / 1000, 0.5) + 0.3
            elif whale_flow > 0 and hurst > 0.45:  # 流入
                scores[MarketRegime.DISTRIBUTION] = min(abs(whale_flow) / 1000, 0.5) + 0.3
        
        # 选择最高分状态
        if scores:
            best_regime = max(scores, key=scores.get)
            confidence = min(1, scores[best_regime])
        else:
            best_regime = MarketRegime.NEUTRAL
            confidence = 0.5
        
        return best_regime, confidence
    
    def _calculate_duration(self, current_regime: MarketRegime) -> int:
        """计算状态持续时间"""
        if not self.history:
            return 0
        
        duration = 0
        for state in reversed(self.history):
            if state.regime == current_regime:
                duration += 5  # 假设每次检查间隔5分钟
            else:
                break
        
        return duration
    
    def _get_strategy_recommendations(
        self,
        regime: MarketRegime
    ) -> Tuple[List[str], List[str]]:
        """获取策略建议"""
        weights = REGIME_STRATEGY_WEIGHTS.get(regime, REGIME_STRATEGY_WEIGHTS[MarketRegime.NEUTRAL])
        
        recommended = [k for k, v in weights.items() if v >= 0.2]
        avoided = [k for k, v in weights.items() if v == 0]
        
        return recommended, avoided
    
    def _identify_risks(
        self,
        regime: MarketRegime,
        vol_ratio: float,
        funding_rate: float,
        sentiment: float,
        whale_flow: float,
    ) -> List[str]:
        """识别风险因素"""
        risks = []
        
        if vol_ratio > self.THRESHOLDS["vol_extreme"]:
            risks.append("极端波动 - 高风险环境")
        
        if abs(funding_rate) > self.THRESHOLDS["funding_extreme"]:
            risks.append("资金费率极端 - 踩踏风险")
        
        if sentiment < 25:
            risks.append("市场恐慌 - 可能继续下跌")
        elif sentiment > 80:
            risks.append("市场狂热 - 注意回调")
        
        if whale_flow > 1000:
            risks.append("巨鲸大额转入 - 潜在抛压")
        elif whale_flow < -1000:
            risks.append("巨鲸大额转出 - 看涨信号")
        
        if regime == MarketRegime.LIQUIDATION:
            risks.append("清算事件进行中 - 极高风险")
        
        return risks
    
    def get_strategy_weights(self, regime: MarketRegime) -> StrategyWeights:
        """获取当前状态下的策略权重"""
        weights = REGIME_STRATEGY_WEIGHTS.get(regime, REGIME_STRATEGY_WEIGHTS[MarketRegime.NEUTRAL])
        
        return StrategyWeights(
            trend_follow=weights.get("trend_follow", 0),
            mean_reversion=weights.get("mean_reversion", 0),
            volatility=weights.get("volatility", 0),
            momentum=weights.get("momentum", 0),
            sentiment=weights.get("sentiment", 0),
            contrarian=weights.get("contrarian", 0),
            liquidity=weights.get("liquidity", 0),
        )
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        if not self.history:
            return {"status": "no_data"}
        
        latest = self.history[-1]
        
        return {
            "regime": latest.regime.value,
            "confidence": round(latest.confidence, 2),
            "trend_strength": round(latest.trend_strength, 2),
            "volatility_level": round(latest.volatility_level, 2),
            "momentum_direction": round(latest.momentum_direction, 2),
            "duration_minutes": latest.duration_minutes,
            "recommended_strategies": latest.recommended_strategies,
            "avoided_strategies": latest.avoided_strategies,
            "risk_factors": latest.risk_factors,
        }


# 便捷函数
def detect_market_regime(
    hurst: float,
    volatility: float,
    avg_volatility: float,
    orderbook_imbalance: float,
    funding_rate: float,
    sentiment_score: float,
    **kwargs
) -> Dict[str, Any]:
    """
    检测市场状态（便捷函数）
    
    Returns:
        市场状态摘要
    """
    engine = MarketRegimeEngine()
    state = engine.detect_regime(
        hurst=hurst,
        volatility=volatility,
        avg_volatility=avg_volatility,
        orderbook_imbalance=orderbook_imbalance,
        funding_rate=funding_rate,
        sentiment_score=sentiment_score,
        **kwargs
    )
    
    return engine.get_regime_summary()
