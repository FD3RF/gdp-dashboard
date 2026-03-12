"""
多维风险矩阵模块 (Layer 10)
整合所有风险因素，生成综合风险评估
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    VERY_LOW = "very_low"      # 0-20
    LOW = "low"                 # 20-40
    MODERATE = "moderate"       # 40-60
    HIGH = "high"               # 60-80
    VERY_HIGH = "very_high"     # 80-100
    EXTREME = "extreme"         # 100+


@dataclass
class RiskFactor:
    """风险因子"""
    name: str
    score: float  # 0-100
    weight: float
    direction: str  # "bullish" / "bearish" / "neutral"
    confidence: float
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAssessment:
    """风险评估结果"""
    total_score: float
    level: RiskLevel
    direction: str
    confidence: float
    
    # 风险因子
    factors: List[RiskFactor]
    
    # 警告和建议
    warnings: List[str]
    recommendations: List[str]
    
    # 关键指标
    key_metrics: Dict[str, float]
    
    timestamp: datetime = field(default_factory=datetime.now)


class RiskMatrix:
    """
    多维风险矩阵
    整合所有风险因素，生成综合风险评估
    """
    
    # 风险因子权重配置
    FACTOR_WEIGHTS = {
        "market_structure": 0.20,    # Hurst, Fractal, Trend
        "liquidity": 0.15,           # 流动性热图, 订单簿
        "funding": 0.15,             # 资金费率极值
        "sentiment": 0.10,           # 社交情绪
        "whale_activity": 0.15,      # 链上巨鲸
        "volatility": 0.10,          # 波动率
        "position_risk": 0.10,       # 当前仓位风险
        "signal_quality": 0.05,      # 历史信号质量
    }
    
    def __init__(self):
        self.factor_cache: Dict[str, RiskFactor] = {}
        self.assessment_history: List[RiskAssessment] = []
        self.max_history = 100
    
    def update_factor(
        self,
        factor_name: str,
        score: float,
        direction: str,
        confidence: float,
        details: str = ""
    ) -> None:
        """更新风险因子"""
        weight = self.FACTOR_WEIGHTS.get(factor_name, 0.05)
        
        self.factor_cache[factor_name] = RiskFactor(
            name=factor_name,
            score=min(100, max(0, score)),
            weight=weight,
            direction=direction,
            confidence=confidence,
            details=details,
        )
    
    def calculate_market_structure_risk(
        self,
        hurst: float,
        fractal_dim: float,
        trend: str
    ) -> RiskFactor:
        """
        计算市场结构风险
        """
        # Hurst 风险
        if hurst > 0.6:
            # 强趋势 - 顺势风险低，逆势风险高
            hurst_score = 30 if trend == "up" else 70
            direction = "bullish" if trend == "up" else "bearish"
        elif hurst < 0.4:
            # 均值回归
            hurst_score = 40
            direction = "neutral"
        else:
            # 随机游走
            hurst_score = 50
            direction = "neutral"
        
        # Fractal 风险
        if fractal_dim > 1.4:
            # 高度粗糙/波动
            fractal_score = 70
        elif fractal_dim < 1.2:
            # 平滑趋势
            fractal_score = 30
        else:
            fractal_score = 50
        
        # 综合分数
        score = (hurst_score + fractal_score) / 2
        confidence = 0.7 if abs(hurst - 0.5) > 0.1 else 0.5
        
        details = f"H={hurst:.3f}, D={fractal_dim:.3f}, 趋势={trend}"
        
        return RiskFactor(
            name="market_structure",
            score=score,
            weight=self.FACTOR_WEIGHTS["market_structure"],
            direction=direction,
            confidence=confidence,
            details=details,
        )
    
    def calculate_liquidity_risk(
        self,
        orderbook_imbalance: float,
        liquidity_zones: List[Dict],
        trap_detected: bool
    ) -> RiskFactor:
        """
        计算流动性风险
        """
        # 订单簿失衡风险
        imbalance_score = abs(orderbook_imbalance) * 50
        
        # 流动性陷阱风险
        if trap_detected:
            imbalance_score += 20
        
        # 流动性区域密度
        zone_density = len(liquidity_zones)
        if zone_density > 5:
            imbalance_score += 15
        
        score = min(100, imbalance_score)
        
        direction = "bullish" if orderbook_imbalance > 0.2 else \
                    "bearish" if orderbook_imbalance < -0.2 else "neutral"
        
        details = f"失衡={orderbook_imbalance:.2f}, 区域={zone_density}, 陷阱={trap_detected}"
        
        return RiskFactor(
            name="liquidity",
            score=score,
            weight=self.FACTOR_WEIGHTS["liquidity"],
            direction=direction,
            confidence=0.8 if trap_detected else 0.6,
            details=details,
        )
    
    def calculate_funding_risk(
        self,
        funding_rate: float,
        z_score: float,
        crowd_direction: str
    ) -> RiskFactor:
        """
        计算资金费率风险
        """
        # 基于费率绝对值
        rate_score = abs(funding_rate) * 100000  # 放大
        
        # 基于 z-score
        z_score_penalty = abs(z_score) * 15
        
        score = min(100, rate_score + z_score_penalty)
        
        # 拥挤方向是反向指标
        if crowd_direction == "long_crowded":
            direction = "bearish"  # 做多拥挤，做空机会
        elif crowd_direction == "short_crowded":
            direction = "bullish"
        else:
            direction = "neutral"
        
        details = f"费率={funding_rate*100:.4f}%, z={z_score:.2f}, 拥挤={crowd_direction}"
        
        return RiskFactor(
            name="funding",
            score=score,
            weight=self.FACTOR_WEIGHTS["funding"],
            direction=direction,
            confidence=min(1.0, abs(z_score) / 3),
            details=details,
        )
    
    def calculate_sentiment_risk(
        self,
        sentiment_score: float,
        is_extreme: bool,
        extreme_type: Optional[str]
    ) -> RiskFactor:
        """
        计算情绪风险
        """
        # 极端情绪 = 高风险
        if is_extreme:
            score = 80
            direction = "bearish" if extreme_type == "euphoria" else "bullish"
        else:
            # 偏离中性程度
            score = abs(sentiment_score - 50)
            direction = "bullish" if sentiment_score > 60 else \
                        "bearish" if sentiment_score < 40 else "neutral"
        
        details = f"情绪={sentiment_score:.1f}, 极端={is_extreme}, 类型={extreme_type}"
        
        return RiskFactor(
            name="sentiment",
            score=score,
            weight=self.FACTOR_WEIGHTS["sentiment"],
            direction=direction,
            confidence=0.7 if is_extreme else 0.5,
            details=details,
        )
    
    def calculate_whale_risk(
        self,
        net_flow: float,
        high_impact_count: int,
        recent_alerts: List[Dict]
    ) -> RiskFactor:
        """
        计算巨鲸活动风险
        """
        # 净流入 = 潜在抛压
        flow_score = min(50, abs(net_flow) / 100)
        if net_flow > 0:
            flow_score += 20  # 流入额外风险
        
        # 高影响警报数量
        alert_score = high_impact_count * 15
        
        score = min(100, flow_score + alert_score)
        
        direction = "bearish" if net_flow > 100 else \
                    "bullish" if net_flow < -100 else "neutral"
        
        details = f"净流={net_flow:.0f} ETH, 高影响={high_impact_count}"
        
        return RiskFactor(
            name="whale_activity",
            score=score,
            weight=self.FACTOR_WEIGHTS["whale_activity"],
            direction=direction,
            confidence=0.8 if high_impact_count > 0 else 0.5,
            details=details,
        )
    
    def calculate_volatility_risk(
        self,
        current_vol: float,
        avg_vol: float,
        vol_trend: str
    ) -> RiskFactor:
        """
        计算波动率风险
        """
        # 相对波动率
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        if vol_ratio > 2:
            score = 80
        elif vol_ratio > 1.5:
            score = 60
        elif vol_ratio > 1:
            score = 40
        else:
            score = 30
        
        direction = "neutral"
        
        details = f"波动率={current_vol:.4f}, 比率={vol_ratio:.2f}, 趋势={vol_trend}"
        
        return RiskFactor(
            name="volatility",
            score=score,
            weight=self.FACTOR_WEIGHTS["volatility"],
            direction=direction,
            confidence=0.7,
            details=details,
        )
    
    def calculate_position_risk(
        self,
        position_size: float,
        unrealized_pnl: float,
        drawdown: float,
        margin_usage: float
    ) -> RiskFactor:
        """
        计算仓位风险
        """
        # 仓位大小风险
        size_score = position_size * 200  # 10%仓位 = 20分
        
        # 回撤风险
        drawdown_score = abs(drawdown) * 100
        
        # 保证金使用风险
        margin_score = margin_usage * 50
        
        score = min(100, size_score + drawdown_score + margin_score)
        
        direction = "bearish" if unrealized_pnl < 0 else "bullish"
        
        details = f"仓位={position_size*100:.1f}%, PnL={unrealized_pnl:.2f}, 回撤={drawdown*100:.1f}%"
        
        return RiskFactor(
            name="position_risk",
            score=score,
            weight=self.FACTOR_WEIGHTS["position_risk"],
            direction=direction,
            confidence=0.9,
            details=details,
        )
    
    def calculate_signal_quality_risk(
        self,
        win_rate: float,
        profit_factor: float,
        recent_accuracy: float
    ) -> RiskFactor:
        """
        计算信号质量风险
        """
        # 胜率低于期望
        win_rate_score = max(0, (0.55 - win_rate) * 200)
        
        # 盈亏比
        pf_score = max(0, (1.5 - profit_factor) * 50)
        
        # 最近准确度
        accuracy_score = max(0, (0.6 - recent_accuracy) * 100)
        
        score = min(100, win_rate_score + pf_score + accuracy_score)
        
        direction = "neutral"
        
        details = f"胜率={win_rate*100:.1f}%, PF={profit_factor:.2f}, 近期={recent_accuracy*100:.1f}%"
        
        return RiskFactor(
            name="signal_quality",
            score=score,
            weight=self.FACTOR_WEIGHTS["signal_quality"],
            direction=direction,
            confidence=0.8,
            details=details,
        )
    
    def assess_risk(
        self,
        market_data: Optional[Dict] = None,
        funding_data: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
        whale_data: Optional[Dict] = None,
        position_data: Optional[Dict] = None,
        signal_history: Optional[Dict] = None,
    ) -> RiskAssessment:
        """
        执行综合风险评估
        """
        factors = []
        
        # 1. 市场结构风险
        if market_data:
            factor = self.calculate_market_structure_risk(
                hurst=market_data.get("hurst", 0.5),
                fractal_dim=market_data.get("fractal_dim", 1.5),
                trend=market_data.get("trend", "neutral"),
            )
            factors.append(factor)
        
        # 2. 流动性风险
        if market_data:
            factor = self.calculate_liquidity_risk(
                orderbook_imbalance=market_data.get("orderbook_imbalance", 0),
                liquidity_zones=market_data.get("liquidity_zones", []),
                trap_detected=market_data.get("trap_detected", False),
            )
            factors.append(factor)
        
        # 3. 资金费率风险
        if funding_data:
            factor = self.calculate_funding_risk(
                funding_rate=funding_data.get("rate", 0),
                z_score=funding_data.get("z_score", 0),
                crowd_direction=funding_data.get("crowd_direction", "neutral"),
            )
            factors.append(factor)
        
        # 4. 情绪风险
        if sentiment_data:
            factor = self.calculate_sentiment_risk(
                sentiment_score=sentiment_data.get("score", 50),
                is_extreme=sentiment_data.get("is_extreme", False),
                extreme_type=sentiment_data.get("extreme_type"),
            )
            factors.append(factor)
        
        # 5. 巨鲸活动风险
        if whale_data:
            factor = self.calculate_whale_risk(
                net_flow=whale_data.get("net_flow", 0),
                high_impact_count=whale_data.get("high_impact_count", 0),
                recent_alerts=whale_data.get("recent_alerts", []),
            )
            factors.append(factor)
        
        # 6. 波动率风险
        if market_data:
            factor = self.calculate_volatility_risk(
                current_vol=market_data.get("volatility", 0.02),
                avg_vol=market_data.get("avg_volatility", 0.02),
                vol_trend=market_data.get("vol_trend", "stable"),
            )
            factors.append(factor)
        
        # 7. 仓位风险
        if position_data:
            factor = self.calculate_position_risk(
                position_size=position_data.get("size", 0),
                unrealized_pnl=position_data.get("pnl", 0),
                drawdown=position_data.get("drawdown", 0),
                margin_usage=position_data.get("margin_usage", 0),
            )
            factors.append(factor)
        else:
            # 无仓位
            factors.append(RiskFactor(
                name="position_risk",
                score=0,
                weight=self.FACTOR_WEIGHTS["position_risk"],
                direction="neutral",
                confidence=1.0,
                details="无持仓",
            ))
        
        # 8. 信号质量风险
        if signal_history:
            factor = self.calculate_signal_quality_risk(
                win_rate=signal_history.get("win_rate", 0.5),
                profit_factor=signal_history.get("profit_factor", 1.0),
                recent_accuracy=signal_history.get("recent_accuracy", 0.5),
            )
            factors.append(factor)
        
        # 计算总分数
        total_weight = sum(f.weight for f in factors)
        weighted_score = sum(f.score * f.weight for f in factors) / total_weight if total_weight > 0 else 50
        
        # 确定风险等级
        level = self._get_risk_level(weighted_score)
        
        # 综合方向
        direction = self._calculate_direction(factors)
        
        # 置信度
        confidence = sum(f.confidence * f.weight for f in factors) / total_weight if total_weight > 0 else 0.5
        
        # 生成警告和建议
        warnings = self._generate_warnings(factors)
        recommendations = self._generate_recommendations(factors, weighted_score)
        
        # 关键指标
        key_metrics = {
            "total_risk_score": round(weighted_score, 1),
            "risk_level": level.value,
            "dominant_direction": direction,
            "confidence": round(confidence, 2),
            "factors_count": len(factors),
        }
        
        assessment = RiskAssessment(
            total_score=weighted_score,
            level=level,
            direction=direction,
            confidence=confidence,
            factors=factors,
            warnings=warnings,
            recommendations=recommendations,
            key_metrics=key_metrics,
        )
        
        # 保存历史
        self.assessment_history.append(assessment)
        if len(self.assessment_history) > self.max_history:
            self.assessment_history = self.assessment_history[-self.max_history:]
        
        return assessment
    
    def _get_risk_level(self, score: float) -> RiskLevel:
        """确定风险等级"""
        if score >= 100:
            return RiskLevel.EXTREME
        elif score >= 80:
            return RiskLevel.VERY_HIGH
        elif score >= 60:
            return RiskLevel.HIGH
        elif score >= 40:
            return RiskLevel.MODERATE
        elif score >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _calculate_direction(self, factors: List[RiskFactor]) -> str:
        """计算综合方向"""
        bullish_weight = sum(f.weight for f in factors if f.direction == "bullish")
        bearish_weight = sum(f.weight for f in factors if f.direction == "bearish")
        
        if bullish_weight > bearish_weight * 1.5:
            return "bullish"
        elif bearish_weight > bullish_weight * 1.5:
            return "bearish"
        else:
            return "neutral"
    
    def _generate_warnings(self, factors: List[RiskFactor]) -> List[str]:
        """生成警告"""
        warnings = []
        
        for f in factors:
            if f.score >= 70:
                warnings.append(f"⚠️ {f.name} 风险极高: {f.details}")
            elif f.score >= 50:
                warnings.append(f"⚡ {f.name} 风险较高: {f.details}")
        
        return warnings
    
    def _generate_recommendations(
        self,
        factors: List[RiskFactor],
        total_score: float
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if total_score >= 80:
            recommendations.append("🚨 极高风险！建议降低仓位或暂时观望")
        elif total_score >= 60:
            recommendations.append("⚠️ 风险较高，建议收紧止损")
        
        # 根据具体因子给出建议
        for f in factors:
            if f.name == "funding" and f.score >= 60:
                if f.direction == "bearish":
                    recommendations.append("💡 做多拥挤，考虑减多或观望")
                else:
                    recommendations.append("💡 做空拥挤，考虑减空或观望")
            
            if f.name == "whale_activity" and f.score >= 60:
                recommendations.append("🐋 巨鲸活动频繁，注意大额转账影响")
            
            if f.name == "position_risk" and f.score >= 50:
                recommendations.append("📊 仓位风险增加，考虑减仓")
        
        return recommendations[:5]  # 最多5条建议
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        if not self.assessment_history:
            return {"status": "no_data"}
        
        latest = self.assessment_history[-1]
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "total_score": latest.total_score,
            "level": latest.level.value,
            "direction": latest.direction,
            "confidence": latest.confidence,
            "warnings": latest.warnings,
            "recommendations": latest.recommendations,
            "factors": [
                {
                    "name": f.name,
                    "score": f.score,
                    "direction": f.direction,
                    "details": f.details,
                }
                for f in latest.factors
            ],
        }


# 便捷函数
def calculate_risk_matrix(
    market_data: Optional[Dict] = None,
    funding_data: Optional[Dict] = None,
    sentiment_data: Optional[Dict] = None,
    whale_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    计算风险矩阵
    
    Returns:
        风险评估结果
    """
    matrix = RiskMatrix()
    assessment = matrix.assess_risk(
        market_data=market_data,
        funding_data=funding_data,
        sentiment_data=sentiment_data,
        whale_data=whale_data,
    )
    
    return {
        "score": assessment.total_score,
        "level": assessment.level.value,
        "direction": assessment.direction,
        "confidence": assessment.confidence,
        "warnings": assessment.warnings,
        "recommendations": assessment.recommendations,
        "factors": {f.name: {"score": f.score, "direction": f.direction} for f in assessment.factors},
    }
