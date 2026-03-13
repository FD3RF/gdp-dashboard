"""
三层决策引擎 (Layered Decision Engine)
======================================
机构级决策流程

第一层（风控）：Meta Filter + 数据完整度 → 通过/强制观望
第二层（方向）：市场状态 + Hurst → 确定主方向
第三层（入场）：流动性热图 + CVD/Delta → 确认入场点

核心原则：
- 只有在主方向确认后，才寻找入场机会
- 反向信号直接忽略，不参与投票
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DecisionLayer(Enum):
    """决策层级"""
    RISK_CONTROL = "风控层"
    DIRECTION = "方向层"
    ENTRY = "入场层"


class Direction(Enum):
    """主方向"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class LayerResult:
    """层级结果"""
    layer: DecisionLayer
    passed: bool
    direction: Direction = Direction.NEUTRAL
    confidence: float = 0.0
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayeredDecision:
    """三层决策结果"""
    # 最终决策
    final_direction: Direction
    final_action: str  # "LONG" / "SHORT" / "HOLD"
    confidence: float
    
    # 各层结果
    risk_layer: LayerResult
    direction_layer: LayerResult
    entry_layer: LayerResult
    
    # 决策详情
    primary_direction: Direction  # 第二层确定的主方向
    entry_confirmed: bool  # 第三层是否确认入场
    entry_reason: str  # 入场/观望原因
    
    # 元信息
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "final_direction": self.final_direction.value,
            "final_action": self.final_action,
            "confidence": round(self.confidence, 1),
            "primary_direction": self.primary_direction.value,
            "entry_confirmed": self.entry_confirmed,
            "entry_reason": self.entry_reason,
            "layers": {
                "risk": {
                    "passed": self.risk_layer.passed,
                    "reason": self.risk_layer.reason,
                },
                "direction": {
                    "passed": self.direction_layer.passed,
                    "direction": self.direction_layer.direction.value,
                    "confidence": self.direction_layer.confidence,
                    "reason": self.direction_layer.reason,
                },
                "entry": {
                    "passed": self.entry_layer.passed,
                    "confirmed": self.entry_layer.passed,
                    "reason": self.entry_layer.reason,
                },
            },
            "timestamp": self.timestamp.isoformat(),
        }


class LayeredDecisionEngine:
    """
    三层决策引擎
    
    决策流程：
    第一层（风控）→ 不通过 → 强制HOLD
                  ↓ 通过
    第二层（方向）→ 确定主方向 (LONG/SHORT/NEUTRAL)
                  ↓
    第三层（入场）→ 方向一致 + 确认信号 → 执行
                                  → 方向不一致 或 无确认 → HOLD
    """
    
    # 第一层阈值
    MIN_DATA_QUALITY = 0.7
    MIN_META_FILTER_PASS = True
    
    # 第二层阈值
    MIN_HURST_TREND = 0.55  # Hurst > 0.55 = 趋势向上
    MAX_HURST_REVERSAL = 0.45  # Hurst < 0.45 = 均值回归
    MIN_REGIME_CONFIDENCE = 0.5
    
    # 第三层阈值
    MIN_CVD_CONFIRMATION = 10  # |CVD| > 10 算确认
    MIN_IMBALANCE = 0.2  # 订单簿失衡 > 0.2 算确认
    
    def __init__(self):
        self.decision_history: List[LayeredDecision] = []
        self.max_history = 100
    
    def make_decision(
        self,
        # 第一层参数
        data_quality_score: float,
        meta_filter_passed: bool,
        risk_score: float,
        
        # 第二层参数
        regime: str,
        regime_confidence: float,
        hurst: float,
        
        # 第三层参数
        current_price: float,
        support_zones: List[Tuple[float, float]],  # [(price, strength), ...]
        resistance_zones: List[Tuple[float, float]],
        cvd: float,
        delta: float,
        orderbook_imbalance: float,
        
        # 额外参数
        momentum: float = 0,
        whale_flow: float = 0,
    ) -> LayeredDecision:
        """
        执行三层决策
        """
        # === 第一层：风控检查 ===
        risk_layer = self._layer1_risk_control(
            data_quality_score=data_quality_score,
            meta_filter_passed=meta_filter_passed,
            risk_score=risk_score,
        )
        
        if not risk_layer.passed:
            # 风控不通过，直接返回HOLD
            return self._create_hold_decision(
                risk_layer, 
                LayerResult(DecisionLayer.DIRECTION, False),
                LayerResult(DecisionLayer.ENTRY, False),
                reason=risk_layer.reason
            )
        
        # === 第二层：方向决策 ===
        direction_layer = self._layer2_direction(
            regime=regime,
            regime_confidence=regime_confidence,
            hurst=hurst,
            momentum=momentum,
            whale_flow=whale_flow,
        )
        
        if direction_layer.direction == Direction.NEUTRAL:
            # 方向不明确，返回HOLD
            return self._create_hold_decision(
                risk_layer,
                direction_layer,
                LayerResult(DecisionLayer.ENTRY, False),
                reason="主方向不明确，等待趋势确认"
            )
        
        # === 第三层：入场确认 ===
        entry_layer = self._layer3_entry(
            primary_direction=direction_layer.direction,
            current_price=current_price,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            cvd=cvd,
            delta=delta,
            orderbook_imbalance=orderbook_imbalance,
        )
        
        # 构建最终决策
        if entry_layer.passed:
            # 入场确认通过
            final_direction = direction_layer.direction
            final_action = final_direction.value.upper()
            confidence = direction_layer.confidence * entry_layer.confidence
            entry_reason = entry_layer.reason
        else:
            # 入场未确认，HOLD等待
            final_direction = Direction.NEUTRAL
            final_action = "HOLD"
            confidence = 0
            entry_reason = entry_layer.reason
        
        decision = LayeredDecision(
            final_direction=final_direction,
            final_action=final_action,
            confidence=confidence,
            risk_layer=risk_layer,
            direction_layer=direction_layer,
            entry_layer=entry_layer,
            primary_direction=direction_layer.direction,
            entry_confirmed=entry_layer.passed,
            entry_reason=entry_reason,
        )
        
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]
        
        # 日志
        self._log_decision(decision)
        
        return decision
    
    def _layer1_risk_control(
        self,
        data_quality_score: float,
        meta_filter_passed: bool,
        risk_score: float,
    ) -> LayerResult:
        """
        第一层：风控检查
        
        规则：
        - 数据质量 < 70% → 不通过
        - Meta Filter 未通过 → 不通过
        - 风险分数 > 70 → 不通过
        """
        reasons = []
        passed = True
        
        # 检查数据质量
        if data_quality_score < self.MIN_DATA_QUALITY:
            passed = False
            reasons.append(f"数据质量={data_quality_score*100:.0f}% < {self.MIN_DATA_QUALITY*100:.0f}%")
        
        # 检查Meta Filter
        if not meta_filter_passed:
            passed = False
            reasons.append("Meta Filter未通过")
        
        # 检查风险分数
        if risk_score > 70:
            passed = False
            reasons.append(f"风险分数={risk_score:.0f} > 70")
        
        return LayerResult(
            layer=DecisionLayer.RISK_CONTROL,
            passed=passed,
            reason="; ".join(reasons) if reasons else "风控检查通过",
            details={
                "data_quality": data_quality_score,
                "meta_filter": meta_filter_passed,
                "risk_score": risk_score,
            }
        )
    
    def _layer2_direction(
        self,
        regime: str,
        regime_confidence: float,
        hurst: float,
        momentum: float = 0,
        whale_flow: float = 0,
    ) -> LayerResult:
        """
        第二层：方向决策
        
        规则：
        1. 市场状态优先
           - trend_up → LONG
           - trend_down → SHORT
           - panic → LONG (抄底)
           - euphoria → SHORT (反向)
        
        2. Hurst确认
           - hurst > 0.55 趋势向上
           - hurst < 0.45 均值回归
        
        3. 辅助信号
           - 巨鲸流出 → 看涨
           - 动量向上 → 增强
        """
        direction = Direction.NEUTRAL
        confidence = 0.5
        reasons = []
        
        # 市场状态判断
        if "trend_up" in regime:
            direction = Direction.LONG
            confidence = 0.7 + regime_confidence * 0.3
            reasons.append(f"市场状态=强势上涨")
        elif "trend_down" in regime:
            direction = Direction.SHORT
            confidence = 0.7 + regime_confidence * 0.3
            reasons.append(f"市场状态=强势下跌")
        elif "panic" in regime:
            direction = Direction.LONG
            confidence = 0.6
            reasons.append("市场恐慌，考虑抄底")
        elif "euphoria" in regime:
            direction = Direction.SHORT
            confidence = 0.6
            reasons.append("市场狂热，考虑反向")
        elif "range" in regime:
            # 震荡行情，需要更多确认
            reasons.append("震荡行情，等待突破")
        else:
            reasons.append(f"市场状态={regime}")
        
        # Hurst确认
        if hurst > self.MIN_HURST_TREND:
            if direction == Direction.LONG:
                confidence = min(0.95, confidence + 0.15)
                reasons.append(f"Hurst={hurst:.3f}趋势确认")
            elif direction == Direction.NEUTRAL:
                direction = Direction.LONG
                confidence = 0.55
                reasons.append(f"Hurst={hurst:.3f}趋势向上")
        elif hurst < self.MAX_HURST_REVERSAL:
            if direction == Direction.SHORT:
                confidence = min(0.95, confidence + 0.15)
                reasons.append(f"Hurst={hurst:.3f}均值回归确认")
            elif direction == Direction.NEUTRAL:
                direction = Direction.SHORT
                confidence = 0.55
                reasons.append(f"Hurst={hurst:.3f}趋势向下")
        
        # 巨鲸流动辅助
        if whale_flow < -500 and direction == Direction.LONG:
            confidence = min(0.95, confidence + 0.05)
            reasons.append(f"巨鲸净流出{abs(whale_flow):.0f}ETH确认")
        elif whale_flow > 500 and direction == Direction.SHORT:
            confidence = min(0.95, confidence + 0.05)
            reasons.append(f"巨鲸净流入{whale_flow:.0f}ETH确认")
        
        return LayerResult(
            layer=DecisionLayer.DIRECTION,
            passed=(direction != Direction.NEUTRAL),
            direction=direction,
            confidence=confidence,
            reason="; ".join(reasons),
            details={
                "regime": regime,
                "hurst": hurst,
                "momentum": momentum,
                "whale_flow": whale_flow,
            }
        )
    
    def _layer3_entry(
        self,
        primary_direction: Direction,
        current_price: float,
        support_zones: List[Tuple[float, float]],
        resistance_zones: List[Tuple[float, float]],
        cvd: float,
        delta: float,
        orderbook_imbalance: float,
    ) -> LayerResult:
        """
        第三层：入场确认
        
        规则（以做空为例）：
        1. 价格在阻力墙附近
        2. CVD开始拐头向下（CVD < 0 或 变负）
        3. 订单簿失衡偏向卖方
        
        规则（以做多为例）：
        1. 价格在支撑墙附近
        2. CVD开始拐头向上（CVD > 0 或 变正）
        3. 订单簿失衡偏向买方
        """
        if primary_direction == Direction.NEUTRAL:
            return LayerResult(
                layer=DecisionLayer.ENTRY,
                passed=False,
                reason="主方向未确定",
            )
        
        confirmed = False
        reasons = []
        entry_confidence = 0.5
        
        if primary_direction == Direction.LONG:
            # 做多入场条件
            
            # 1. 价格在支撑墙附近
            near_support = False
            for price, strength in support_zones[:3]:
                distance_pct = abs(current_price - price) / current_price
                if distance_pct < 0.01:  # 1%以内算接近
                    near_support = True
                    reasons.append(f"价格接近支撑${price:,.2f}")
                    entry_confidence += 0.1 * strength
                    break
            
            if not near_support:
                reasons.append(f"价格${current_price:,.2f}未接近支撑区")
            
            # 2. CVD确认
            if cvd > self.MIN_CVD_CONFIRMATION:
                confirmed = True
                reasons.append(f"CVD={cvd:.1f}买方主导")
                entry_confidence += 0.2
            elif cvd > 0:
                reasons.append(f"CVD={cvd:.1f}略偏买方")
                entry_confidence += 0.1
            else:
                reasons.append(f"CVD={cvd:.1f}未确认做多")
            
            # 3. 订单簿确认
            if orderbook_imbalance > self.MIN_IMBALANCE:
                confirmed = confirmed or near_support
                reasons.append(f"订单簿失衡={orderbook_imbalance:.2f}买盘优势")
                entry_confidence += 0.1
            else:
                reasons.append(f"订单簿失衡={orderbook_imbalance:.2f}未确认")
            
            # 综合判断
            entry_passed = near_support and (cvd > 0 or orderbook_imbalance > 0)
            
        else:  # SHORT
            # 做空入场条件
            
            # 1. 价格在阻力墙附近
            near_resistance = False
            for price, strength in resistance_zones[:3]:
                distance_pct = abs(current_price - price) / current_price
                if distance_pct < 0.01:  # 1%以内算接近
                    near_resistance = True
                    reasons.append(f"价格接近阻力${price:,.2f}")
                    entry_confidence += 0.1 * strength
                    break
            
            if not near_resistance:
                reasons.append(f"价格${current_price:,.2f}未接近阻力区")
            
            # 2. CVD确认（做空需要CVD < 0）
            if cvd < -self.MIN_CVD_CONFIRMATION:
                confirmed = True
                reasons.append(f"CVD={cvd:.1f}卖方主导")
                entry_confidence += 0.2
            elif cvd < 0:
                reasons.append(f"CVD={cvd:.1f}略偏卖方")
                entry_confidence += 0.1
            else:
                reasons.append(f"CVD={cvd:.1f}未确认做空")
            
            # 3. 订单簿确认
            if orderbook_imbalance < -self.MIN_IMBALANCE:
                confirmed = confirmed or near_resistance
                reasons.append(f"订单簿失衡={orderbook_imbalance:.2f}卖盘优势")
                entry_confidence += 0.1
            else:
                reasons.append(f"订单簿失衡={orderbook_imbalance:.2f}未确认")
            
            # 综合判断
            entry_passed = near_resistance and (cvd < 0 or orderbook_imbalance < 0)
        
        return LayerResult(
            layer=DecisionLayer.ENTRY,
            passed=entry_passed,
            direction=primary_direction,
            confidence=min(1.0, entry_confidence),
            reason="; ".join(reasons),
            details={
                "near_key_level": entry_passed,
                "cvd": cvd,
                "delta": delta,
                "imbalance": orderbook_imbalance,
            }
        )
    
    def _create_hold_decision(
        self,
        risk_layer: LayerResult,
        direction_layer: LayerResult,
        entry_layer: LayerResult,
        reason: str,
    ) -> LayeredDecision:
        """创建HOLD决策"""
        return LayeredDecision(
            final_direction=Direction.NEUTRAL,
            final_action="HOLD",
            confidence=0,
            risk_layer=risk_layer,
            direction_layer=direction_layer,
            entry_layer=entry_layer,
            primary_direction=Direction.NEUTRAL,
            entry_confirmed=False,
            entry_reason=reason,
        )
    
    def _log_decision(self, decision: LayeredDecision):
        """记录决策日志"""
        if decision.final_action != "HOLD":
            logger.info(
                f"🎯 三层决策: {decision.final_action} "
                f"(方向={decision.primary_direction.value}, "
                f"确认={decision.entry_confirmed}, "
                f"置信度={decision.confidence:.1f})"
            )
        else:
            logger.info(f"⏸️ 三层决策: HOLD ({decision.entry_reason})")


# 全局实例
_layered_engine: Optional[LayeredDecisionEngine] = None


def get_layered_engine() -> LayeredDecisionEngine:
    """获取全局三层决策引擎"""
    global _layered_engine
    if _layered_engine is None:
        _layered_engine = LayeredDecisionEngine()
    return _layered_engine


def make_layered_decision(
    data_quality_score: float,
    meta_filter_passed: bool,
    risk_score: float,
    regime: str,
    regime_confidence: float,
    hurst: float,
    current_price: float,
    support_zones: List[Tuple[float, float]],
    resistance_zones: List[Tuple[float, float]],
    cvd: float,
    delta: float,
    orderbook_imbalance: float,
    **kwargs
) -> LayeredDecision:
    """
    执行三层决策（便捷函数）
    """
    engine = get_layered_engine()
    return engine.make_decision(
        data_quality_score=data_quality_score,
        meta_filter_passed=meta_filter_passed,
        risk_score=risk_score,
        regime=regime,
        regime_confidence=regime_confidence,
        hurst=hurst,
        current_price=current_price,
        support_zones=support_zones,
        resistance_zones=resistance_zones,
        cvd=cvd,
        delta=delta,
        orderbook_imbalance=orderbook_imbalance,
        **kwargs
    )
