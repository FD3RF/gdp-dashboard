"""
四层决策引擎 (Four-Layer Decision Engine)
==========================================
机构级决策流程 v2.0

第一层（最高优先级）：数据质量和风控
    ├─ 数据不完整 → 强制观望
    └─ 风控报警 → 强制观望

第二层（方向判断）：选择一个主导模块
    ├─ 趋势明确时，以趋势为主（最强特征是trend）
    └─ 趋势不明时，以订单流为主

第三层（入场过滤）：用其他模块辅助
    ├─ 流动性热图找关键价位
    ├─ 清算数据找引爆点
    └─ 资金费率验证拥挤度

第四层（仓位管理）：根据信号一致性调整
    ├─ 高度一致(>75%) → 正常仓位
    ├─ 中度一致(50-75%) → 减半仓位
    └─ 低度一致(<50%) → 放弃交易
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DecisionLayer(Enum):
    """决策层级"""
    RISK_CONTROL = "第一层-风控"
    DIRECTION = "第二层-方向"
    ENTRY = "第三层-入场"
    POSITION = "第四层-仓位"


class Direction(Enum):
    """主方向"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class PositionSize(Enum):
    """仓位等级"""
    FULL = "full"           # 正常仓位
    HALF = "half"           # 减半仓位
    QUARTER = "quarter"     # 1/4仓位
    NONE = "none"           # 不交易


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
class FourLayerDecision:
    """四层决策结果"""
    # 最终决策
    final_direction: Direction
    final_action: str  # "LONG" / "SHORT" / "HOLD"
    confidence: float
    
    # 各层结果
    risk_layer: LayerResult
    direction_layer: LayerResult
    entry_layer: LayerResult
    position_layer: LayerResult
    
    # 决策详情
    primary_direction: Direction
    dominant_module: str  # "trend" 或 "order_flow"
    entry_confirmed: bool
    position_size: PositionSize
    position_multiplier: float  # 仓位乘数
    
    # 关键信息
    key_level: Optional[float] = None  # 关键价位
    liquidation_trigger: Optional[str] = None  # 清算引爆点
    crowding_warning: Optional[str] = None  # 拥挤警告
    
    # 原因说明
    entry_reason: str = ""
    
    # 元信息
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "final_direction": self.final_direction.value,
            "final_action": self.final_action,
            "confidence": round(self.confidence, 1),
            "primary_direction": self.primary_direction.value,
            "dominant_module": self.dominant_module,
            "entry_confirmed": self.entry_confirmed,
            "position_size": self.position_size.value,
            "position_multiplier": self.position_multiplier,
            "key_level": self.key_level,
            "liquidation_trigger": self.liquidation_trigger,
            "crowding_warning": self.crowding_warning,
            "entry_reason": self.entry_reason,
            "layers": {
                "risk": {
                    "passed": self.risk_layer.passed,
                    "reason": self.risk_layer.reason,
                },
                "direction": {
                    "passed": self.direction_layer.passed,
                    "direction": self.direction_layer.direction.value,
                    "dominant": self.dominant_module,
                    "confidence": self.direction_layer.confidence,
                    "reason": self.direction_layer.reason,
                },
                "entry": {
                    "passed": self.entry_layer.passed,
                    "confirmed": self.entry_layer.passed,
                    "key_level": self.key_level,
                    "reason": self.entry_layer.reason,
                },
                "position": {
                    "passed": self.position_layer.passed,
                    "size": self.position_size.value,
                    "multiplier": self.position_multiplier,
                    "reason": self.position_layer.reason,
                },
            },
            "timestamp": self.timestamp.isoformat(),
        }


class FourLayerDecisionEngine:
    """
    四层决策引擎
    
    核心改进：
    1. 第二层：趋势明确用趋势，趋势不明用订单流
    2. 第三层：增加清算数据和资金费率验证
    3. 第四层：根据一致性动态调整仓位
    """
    
    # 第一层阈值
    MIN_DATA_QUALITY = 0.7
    
    # 第二层阈值
    TREND_THRESHOLD = 0.55  # Hurst > 0.55 = 趋势明确
    ORDER_FLOW_THRESHOLD = 15  # |CVD| > 15 = 订单流明确
    
    # 第三层阈值
    LIQUIDATION_PROXIMITY = 0.005  # 0.5%以内接近清算区
    CROWDING_THRESHOLD = 0.0005  # 资金费率极值
    
    # 第四层阈值
    HIGH_CONSISTENCY = 0.75
    MEDIUM_CONSISTENCY = 0.50
    
    def __init__(self):
        self.decision_history: List[FourLayerDecision] = []
        self.max_history = 100
    
    def make_decision(
        self,
        # 第一层参数
        data_quality_score: float,
        meta_filter_passed: bool,
        risk_score: float,
        
        # 第二层参数（趋势）
        regime: str,
        regime_confidence: float,
        hurst: float,
        
        # 第二层参数（订单流）
        cvd: float,
        delta: float,
        orderbook_imbalance: float,
        
        # 第三层参数
        current_price: float,
        support_zones: List[Tuple[float, float]],
        resistance_zones: List[Tuple[float, float]],
        liquidation_zones: List[Dict],  # 清算区域
        funding_rate: float,
        funding_extreme: bool,
        
        # 第四层参数
        signal_consistency: float,
        
        # 额外参数
        momentum: float = 0,
        whale_flow: float = 0,
    ) -> FourLayerDecision:
        """
        执行四层决策
        """
        # === 第一层：数据质量和风控 ===
        risk_layer = self._layer1_risk_control(
            data_quality_score=data_quality_score,
            meta_filter_passed=meta_filter_passed,
            risk_score=risk_score,
        )
        
        if not risk_layer.passed:
            return self._create_hold_decision(
                risk_layer,
                LayerResult(DecisionLayer.DIRECTION, False),
                LayerResult(DecisionLayer.ENTRY, False),
                LayerResult(DecisionLayer.POSITION, False),
                reason=risk_layer.reason
            )
        
        # === 第二层：方向判断（选择主导模块）===
        direction_layer, dominant_module = self._layer2_direction_with_dominant(
            # 趋势参数
            regime=regime,
            regime_confidence=regime_confidence,
            hurst=hurst,
            # 订单流参数
            cvd=cvd,
            delta=delta,
            orderbook_imbalance=orderbook_imbalance,
            # 其他
            momentum=momentum,
            whale_flow=whale_flow,
        )
        
        if direction_layer.direction == Direction.NEUTRAL:
            return self._create_hold_decision(
                risk_layer,
                direction_layer,
                LayerResult(DecisionLayer.ENTRY, False),
                LayerResult(DecisionLayer.POSITION, False),
                reason="主方向不明确，等待趋势或订单流确认"
            )
        
        # === 第三层：入场过滤（多模块辅助）===
        entry_layer, key_level, liq_trigger, crowd_warning = self._layer3_entry_with_auxiliary(
            primary_direction=direction_layer.direction,
            current_price=current_price,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            liquidation_zones=liquidation_zones,
            funding_rate=funding_rate,
            funding_extreme=funding_extreme,
            cvd=cvd,
            orderbook_imbalance=orderbook_imbalance,
        )
        
        # === 第四层：仓位管理 ===
        position_layer = self._layer4_position_management(
            entry_confirmed=entry_layer.passed,
            signal_consistency=signal_consistency,
            funding_extreme=funding_extreme,
        )
        
        # 构建最终决策
        if entry_layer.passed and position_layer.passed:
            final_direction = direction_layer.direction
            final_action = final_direction.value.upper()
            confidence = direction_layer.confidence * entry_layer.confidence
        elif entry_layer.passed and not position_layer.passed:
            # 入场确认但仓位管理不通过 → 减仓或放弃
            final_direction = direction_layer.direction
            final_action = final_direction.value.upper()
            confidence = direction_layer.confidence * 0.5
        else:
            # 修复：即使最终决策是HOLD，也保留方向层的置信度用于显示
            final_direction = Direction.NEUTRAL
            final_action = "HOLD"
            # 保留方向层置信度（不设为0），让用户知道信号强度
            confidence = direction_layer.confidence if direction_layer.passed else risk_layer.confidence
        
        decision = FourLayerDecision(
            final_direction=final_direction,
            final_action=final_action,
            confidence=confidence,
            risk_layer=risk_layer,
            direction_layer=direction_layer,
            entry_layer=entry_layer,
            position_layer=position_layer,
            primary_direction=direction_layer.direction,
            dominant_module=dominant_module,
            entry_confirmed=entry_layer.passed,
            position_size=position_layer.details.get('position_size', PositionSize.NONE),
            position_multiplier=position_layer.details.get('multiplier', 0),
            key_level=key_level,
            liquidation_trigger=liq_trigger,
            crowding_warning=crowd_warning,
            entry_reason=entry_layer.reason,
        )
        
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]
        
        self._log_decision(decision)
        
        return decision
    
    def _layer1_risk_control(
        self,
        data_quality_score: float,
        meta_filter_passed: bool,
        risk_score: float,
    ) -> LayerResult:
        """
        第一层：数据质量和风控
        
        规则：
        - 数据质量 < 70% → 不通过
        - Meta Filter 未通过 → 不通过
        - 风险分数 > 70 → 不通过
        """
        reasons = []
        passed = True
        
        if data_quality_score < self.MIN_DATA_QUALITY:
            passed = False
            reasons.append(f"数据质量={data_quality_score*100:.0f}%")
        
        if not meta_filter_passed:
            passed = False
            reasons.append("Meta Filter未通过")
        
        if risk_score > 70:
            passed = False
            reasons.append(f"风险分数={risk_score:.0f}")
        
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
    
    def _layer2_direction_with_dominant(
        self,
        regime: str,
        regime_confidence: float,
        hurst: float,
        cvd: float,
        delta: float,
        orderbook_imbalance: float,
        momentum: float = 0,
        whale_flow: float = 0,
    ) -> Tuple[LayerResult, str]:
        """
        第二层：方向判断（选择主导模块）
        
        规则：
        1. 趋势明确（Hurst > 0.55 或 regime明确）→ 以趋势为主
        2. 趋势不明 → 以订单流为主
        """
        direction = Direction.NEUTRAL
        confidence = 0.5
        reasons = []
        dominant = "none"
        
        # 判断趋势是否明确
        trend_clear = (
            hurst > self.TREND_THRESHOLD or 
            hurst < (1 - self.TREND_THRESHOLD) or
            regime in ["trend_up", "trend_down", "panic", "euphoria"]
        )
        
        if trend_clear:
            # === 以趋势为主 ===
            dominant = "trend"
            
            if "trend_up" in regime or hurst > self.TREND_THRESHOLD:
                direction = Direction.LONG
                confidence = 0.7 + regime_confidence * 0.3
                reasons.append(f"趋势明确向上(H={hurst:.3f})")
            elif "trend_down" in regime or hurst < (1 - self.TREND_THRESHOLD):
                direction = Direction.SHORT
                confidence = 0.7 + regime_confidence * 0.3
                reasons.append(f"趋势明确向下(H={hurst:.3f})")
            elif "panic" in regime:
                direction = Direction.LONG
                confidence = 0.65
                reasons.append("恐慌抄底")
            elif "euphoria" in regime:
                direction = Direction.SHORT
                confidence = 0.65
                reasons.append("狂热反向")
            else:
                # regime明确但方向中性
                reasons.append(f"Regime={regime}")
            
            # 订单流作为辅助确认（不改变方向）
            if cvd != 0:
                if (direction == Direction.LONG and cvd > 0) or \
                   (direction == Direction.SHORT and cvd < 0):
                    confidence = min(0.95, confidence + 0.05)
                    reasons.append(f"订单流确认(CVD={cvd:.1f})")
                else:
                    confidence = max(0.5, confidence - 0.05)
                    reasons.append(f"订单流背离(CVD={cvd:.1f})")
            
        else:
            # === 以订单流为主 ===
            dominant = "order_flow"
            reasons.append(f"趋势不明(H={hurst:.3f})，以订单流为主")
            
            # CVD判断方向
            if abs(cvd) > self.ORDER_FLOW_THRESHOLD:
                if cvd > 0:
                    direction = Direction.LONG
                    confidence = 0.6 + min(0.2, abs(cvd) / 100)
                    reasons.append(f"订单流多头(CVD={cvd:.1f})")
                else:
                    direction = Direction.SHORT
                    confidence = 0.6 + min(0.2, abs(cvd) / 100)
                    reasons.append(f"订单流空头(CVD={cvd:.1f})")
            
            # 订单簿失衡确认
            if abs(orderbook_imbalance) > 0.2:
                if (direction == Direction.LONG and orderbook_imbalance > 0) or \
                   (direction == Direction.SHORT and orderbook_imbalance < 0):
                    confidence = min(0.85, confidence + 0.1)
                    reasons.append(f"订单簿确认(IMB={orderbook_imbalance:.2f})")
                elif direction == Direction.NEUTRAL:
                    direction = Direction.LONG if orderbook_imbalance > 0 else Direction.SHORT
                    confidence = 0.55
                    reasons.append(f"订单簿失衡={orderbook_imbalance:.2f}")
        
        # 巨鲸流动辅助
        if abs(whale_flow) > 500:
            flow_confirm = (
                (direction == Direction.LONG and whale_flow < 0) or  # 流出=看涨
                (direction == Direction.SHORT and whale_flow > 0)   # 流入=看跌
            )
            if flow_confirm:
                confidence = min(0.95, confidence + 0.05)
                reasons.append(f"巨鲸确认")
        
        return LayerResult(
            layer=DecisionLayer.DIRECTION,
            passed=(direction != Direction.NEUTRAL),
            direction=direction,
            confidence=confidence,
            reason="; ".join(reasons),
            details={
                "dominant": dominant,
                "regime": regime,
                "hurst": hurst,
                "cvd": cvd,
                "trend_clear": trend_clear,
            }
        ), dominant
    
    def _layer3_entry_with_auxiliary(
        self,
        primary_direction: Direction,
        current_price: float,
        support_zones: List[Tuple[float, float]],
        resistance_zones: List[Tuple[float, float]],
        liquidation_zones: List[Dict],
        funding_rate: float,
        funding_extreme: bool,
        cvd: float,
        orderbook_imbalance: float,
    ) -> Tuple[LayerResult, Optional[float], Optional[str], Optional[str]]:
        """
        第三层：入场过滤（多模块辅助）
        
        辅助模块：
        1. 流动性热图找关键价位
        2. 清算数据找引爆点
        3. 资金费率验证拥挤度
        """
        if primary_direction == Direction.NEUTRAL:
            return (
                LayerResult(DecisionLayer.ENTRY, False, reason="主方向未确定"),
                None, None, None
            )
        
        confirmed = False
        reasons = []
        entry_confidence = 0.5
        key_level = None
        liq_trigger = None
        crowd_warning = None
        
        # === 1. 流动性热图找关键价位 ===
        if primary_direction == Direction.LONG:
            # 找支撑位
            for price, strength in support_zones[:3]:
                distance_pct = abs(current_price - price) / current_price
                if distance_pct < 0.015:  # 1.5%以内
                    key_level = price
                    confirmed = True
                    entry_confidence += 0.15 * strength
                    reasons.append(f"接近支撑${price:,.0f}")
                    break
        else:
            # 找阻力位
            for price, strength in resistance_zones[:3]:
                distance_pct = abs(current_price - price) / current_price
                if distance_pct < 0.015:
                    key_level = price
                    confirmed = True
                    entry_confidence += 0.15 * strength
                    reasons.append(f"接近阻力${price:,.0f}")
                    break
        
        # === 2. 清算数据找引爆点 ===
        if liquidation_zones:
            for liq in liquidation_zones[:3]:
                liq_price = liq.get('price', 0)
                liq_direction = liq.get('direction', 'unknown')
                distance_pct = abs(current_price - liq_price) / current_price
                
                if distance_pct < self.LIQUIDATION_PROXIMITY:
                    # 检查是否与主方向一致
                    if (primary_direction == Direction.LONG and liq_direction == 'short') or \
                       (primary_direction == Direction.SHORT and liq_direction == 'long'):
                        liq_trigger = f"清算引爆点@${liq_price:,.0f}"
                        confirmed = True
                        entry_confidence += 0.1
                        reasons.append(f"清算引爆点")
                        break
        
        # === 3. 资金费率验证拥挤度 ===
        if funding_extreme:
            crowd_warning = f"资金费率极值={funding_rate*100:.4f}%"
            # 拥挤警告降低入场信心
            entry_confidence *= 0.8
            reasons.append(f"拥挤警告")
        
        # === 综合判断 ===
        # CVD确认
        if primary_direction == Direction.LONG:
            if cvd > 5:
                entry_confidence += 0.1
                reasons.append(f"CVD确认={cvd:.1f}")
            elif cvd < -5:
                entry_confidence -= 0.15
                reasons.append(f"CVD背离={cvd:.1f}")
        else:
            if cvd < -5:
                entry_confidence += 0.1
                reasons.append(f"CVD确认={cvd:.1f}")
            elif cvd > 5:
                entry_confidence -= 0.15
                reasons.append(f"CVD背离={cvd:.1f}")
        
        entry_passed = confirmed and entry_confidence > 0.5
        
        return LayerResult(
            layer=DecisionLayer.ENTRY,
            passed=entry_passed,
            direction=primary_direction,
            confidence=min(1.0, max(0, entry_confidence)),
            reason="; ".join(reasons),
            details={
                "key_level": key_level,
                "liquidation_trigger": liq_trigger,
                "crowding_warning": crowd_warning,
            }
        ), key_level, liq_trigger, crowd_warning
    
    def _layer4_position_management(
        self,
        entry_confirmed: bool,
        signal_consistency: float,
        funding_extreme: bool,
    ) -> LayerResult:
        """
        第四层：仓位管理
        
        规则：
        - 高度一致(>75%) → 正常仓位 (100%)
        - 中度一致(50-75%) → 减半仓位 (50%)
        - 低度一致(<50%) → 放弃交易 (0%)
        - 拥挤警告 → 额外减半
        """
        if not entry_confirmed:
            return LayerResult(
                layer=DecisionLayer.POSITION,
                passed=False,
                reason="入场未确认",
                details={
                    "position_size": PositionSize.NONE,
                    "multiplier": 0,
                }
            )
        
        # 基于一致性确定基础仓位
        if signal_consistency >= self.HIGH_CONSISTENCY:
            position_size = PositionSize.FULL
            multiplier = 1.0
            reason = f"高度一致({signal_consistency*100:.0f}%)正常仓位"
        elif signal_consistency >= self.MEDIUM_CONSISTENCY:
            position_size = PositionSize.HALF
            multiplier = 0.5
            reason = f"中度一致({signal_consistency*100:.0f}%)减半仓位"
        else:
            position_size = PositionSize.NONE
            multiplier = 0
            reason = f"低度一致({signal_consistency*100:.0f}%)放弃交易"
        
        # 拥挤警告额外减仓
        if funding_extreme and multiplier > 0:
            multiplier *= 0.5
            position_size = PositionSize.QUARTER if multiplier < 0.5 else position_size
            reason += "，拥挤警告减仓"
        
        passed = multiplier > 0
        
        return LayerResult(
            layer=DecisionLayer.POSITION,
            passed=passed,
            reason=reason,
            details={
                "position_size": position_size,
                "multiplier": multiplier,
                "consistency": signal_consistency,
                "funding_extreme": funding_extreme,
            }
        )
    
    def _create_hold_decision(
        self,
        risk_layer: LayerResult,
        direction_layer: LayerResult,
        entry_layer: LayerResult,
        position_layer: LayerResult,
        reason: str,
    ) -> FourLayerDecision:
        """创建HOLD决策"""
        return FourLayerDecision(
            final_direction=Direction.NEUTRAL,
            final_action="HOLD",
            confidence=0,
            risk_layer=risk_layer,
            direction_layer=direction_layer,
            entry_layer=entry_layer,
            position_layer=position_layer,
            primary_direction=Direction.NEUTRAL,
            dominant_module="none",
            entry_confirmed=False,
            position_size=PositionSize.NONE,
            position_multiplier=0,
            entry_reason=reason,
        )
    
    def _log_decision(self, decision: FourLayerDecision):
        """记录决策日志"""
        if decision.final_action != "HOLD":
            logger.info(
                f"🎯 四层决策: {decision.final_action} "
                f"(主导={decision.dominant_module}, "
                f"仓位={decision.position_multiplier*100:.0f}%, "
                f"置信度={decision.confidence:.1f})"
            )
        else:
            logger.info(f"⏸️ 四层决策: HOLD ({decision.entry_reason})")


# 全局实例
_four_layer_engine: Optional[FourLayerDecisionEngine] = None


def get_four_layer_engine() -> FourLayerDecisionEngine:
    """获取全局四层决策引擎"""
    global _four_layer_engine
    if _four_layer_engine is None:
        _four_layer_engine = FourLayerDecisionEngine()
    return _four_layer_engine


def make_four_layer_decision(
    data_quality_score: float,
    meta_filter_passed: bool,
    risk_score: float,
    regime: str,
    regime_confidence: float,
    hurst: float,
    cvd: float,
    delta: float,
    orderbook_imbalance: float,
    current_price: float,
    support_zones: List[Tuple[float, float]],
    resistance_zones: List[Tuple[float, float]],
    liquidation_zones: List[Dict],
    funding_rate: float,
    funding_extreme: bool,
    signal_consistency: float,
    **kwargs
) -> FourLayerDecision:
    """
    执行四层决策（便捷函数）
    """
    engine = get_four_layer_engine()
    return engine.make_decision(
        data_quality_score=data_quality_score,
        meta_filter_passed=meta_filter_passed,
        risk_score=risk_score,
        regime=regime,
        regime_confidence=regime_confidence,
        hurst=hurst,
        cvd=cvd,
        delta=delta,
        orderbook_imbalance=orderbook_imbalance,
        current_price=current_price,
        support_zones=support_zones,
        resistance_zones=resistance_zones,
        liquidation_zones=liquidation_zones,
        funding_rate=funding_rate,
        funding_extreme=funding_extreme,
        signal_consistency=signal_consistency,
        **kwargs
    )
