"""
统一决策引擎 (Unified Decision Engine)
======================================
整合Signal Engine和四层决策，避免冲突

核心原则：
1. Signal Engine提供模块投票和一致性
2. 四层决策提供主方向和入场判断
3. 硬规则过滤器覆盖所有层级
4. 最终输出唯一决策
5. 置信度校准确保概率准确性
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .signal_engine import SignalEngine, SignalDirection, SignalStrength
from .layered_decision import FourLayerDecision, PositionSize, Direction
from .risk_filter import HardRiskFilter, FilterResult

logger = logging.getLogger(__name__)

# 置信度校准器（延迟导入避免循环依赖）
_calibrator = None

def _get_calibrator():
    """获取校准器（延迟导入）"""
    global _calibrator
    if _calibrator is None:
        try:
            from ai.confidence_calibration import get_calibrator
            _calibrator = get_calibrator("platt")
        except ImportError:
            logger.warning("Confidence calibration module not available")
    return _calibrator


@dataclass
class UnifiedDecision:
    """统一决策结果"""
    # 最终决策
    action: str  # "LONG" / "SHORT" / "HOLD"
    confidence: float
    position_multiplier: float
    
    # 来源追踪
    decision_source: str  # "four_layer" / "hard_filter_block" / "signal_engine"
    
    # 各引擎结果
    signal_engine_result: Dict[str, Any]
    four_layer_result: Dict[str, Any]
    hard_filter_result: Dict[str, Any]
    
    # 冲突检测
    has_conflict: bool = False
    conflict_reason: str = ""
    
    # 元信息
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 1),
            "position_multiplier": self.position_multiplier,
            "decision_source": self.decision_source,
            "has_conflict": self.has_conflict,
            "conflict_reason": self.conflict_reason,
            "signal_engine": self.signal_engine_result,
            "four_layer": self.four_layer_result,
            "hard_filter": self.hard_filter_result,
            "timestamp": self.timestamp.isoformat(),
        }


class UnifiedDecisionEngine:
    """
    统一决策引擎
    
    决策流程：
    1. Signal Engine → 模块投票 + 一致性
    2. 四层决策 → 主方向 + 入场判断
    3. 冲突检测 → 两者方向是否一致
    4. 硬规则过滤 → 最终校验
    5. 输出统一决策
    """
    
    def __init__(self):
        self.signal_engine = SignalEngine()
        self.hard_filter = HardRiskFilter()
    
    def make_unified_decision(
        self,
        # Signal Engine 参数
        hurst: float,
        momentum: float,
        imbalance: float,
        whale_flow: float,
        funding_rate: float,
        cvd: float,
        sentiment: float,
        regime: str,
        regime_confidence: float,
        risk_score: float,
        
        # 四层决策参数
        data_quality_score: float,
        current_price: float,
        support_zones: List[Tuple[float, float]],
        resistance_zones: List[Tuple[float, float]],
        liquidation_zones: List[Dict],
        funding_extreme: bool,
        
        # 硬规则参数
        reliability: Dict,
        trade_plan: Any,
        
        # 额外参数
        **kwargs
    ) -> UnifiedDecision:
        """
        执行统一决策
        """
        # === Step 1: Signal Engine 模块投票 ===
        self.signal_engine.process_technical_signals(
            hurst=hurst,
            fractal_dim=1.5,
            momentum=momentum,
            volatility=0.02,
        )
        
        self.signal_engine.process_onchain_signals(
            whale_net_flow=whale_flow,
            funding_rate=funding_rate,
            liquidation_imbalance=0,
        )
        
        self.signal_engine.process_orderbook_signals(
            imbalance=imbalance,
            cvd=cvd,
            delta=cvd,
        )
        
        self.signal_engine.process_sentiment_signals(
            social_score=sentiment,
            regime=regime,
            regime_confidence=regime_confidence,
        )
        
        signal_result = self.signal_engine.generate_final_signal(
            regime=regime,
            risk_score=risk_score,
        )
        
        # Signal Engine 方向
        se_direction = signal_result.final_direction.value  # "long" / "short" / "neutral"
        se_confidence = signal_result.final_confidence
        se_consistency = signal_result.signal_consistency
        
        signal_engine_dict = {
            "direction": se_direction,
            "confidence": se_confidence,
            "consistency": se_consistency,
            "votes": {
                "long": signal_result.long_votes,
                "short": signal_result.short_votes,
                "neutral": signal_result.neutral_votes,
            },
            "meta_passed": signal_result.meta_filter_passed,
        }
        
        # === Step 2: 四层决策 ===
        from .layered_decision import make_four_layer_decision
        
        four_layer = make_four_layer_decision(
            data_quality_score=data_quality_score,
            meta_filter_passed=signal_result.meta_filter_passed,
            risk_score=risk_score,
            regime=regime,
            regime_confidence=regime_confidence,
            hurst=hurst,
            current_price=current_price,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            liquidation_zones=liquidation_zones,
            funding_rate=funding_rate,
            funding_extreme=funding_extreme,
            signal_consistency=se_consistency,
            cvd=cvd,
            delta=cvd,
            orderbook_imbalance=imbalance,
            momentum=momentum,
            whale_flow=whale_flow,
        )
        
        fl_action = four_layer.final_action  # "LONG" / "SHORT" / "HOLD"
        fl_direction = four_layer.primary_direction.value
        fl_confidence = four_layer.confidence
        fl_position = four_layer.position_multiplier
        
        four_layer_dict = four_layer.to_dict()
        
        # === Step 3: 冲突检测 ===
        has_conflict = False
        conflict_reason = ""
        
        # 检测 Signal Engine 和 四层决策方向冲突
        if fl_action != "HOLD" and se_direction != "neutral":
            if (fl_action.lower() != se_direction):
                has_conflict = True
                conflict_reason = f"Signal Engine={se_direction}, 四层={fl_action}"
                logger.warning(f"⚠️ 决策冲突: {conflict_reason}")
        
        # === Step 4: 解决冲突 ===
        # 规则：四层决策优先（更严格的逻辑）
        final_action = fl_action
        final_confidence = fl_confidence
        final_position = fl_position
        decision_source = "four_layer"
        
        # 如果有冲突，降低置信度
        if has_conflict:
            final_confidence *= 0.7
            # 如果四层入场未确认，改为HOLD
            if not four_layer.entry_confirmed:
                final_action = "HOLD"
                final_position = 0
                conflict_reason += " → 入场未确认，转为观望"
        
        # === Step 5: 硬规则过滤（覆盖所有层级）===
        from .decision_maker import Signal
        signal_enum = Signal.LONG if final_action == "LONG" else Signal.SHORT if final_action == "SHORT" else Signal.HOLD
        
        hard_result = self.hard_filter.check_all(
            signal=final_action,
            trade_plan=trade_plan,
            reliability=reliability,
            data_quality_score=data_quality_score,
            unified_signal=signal_engine_dict,
            probabilities={"confidence": final_confidence},
        )
        
        hard_filter_dict = hard_result.to_dict()
        
        # 硬规则覆盖
        if not hard_result.is_trading_allowed():
            final_action = "HOLD"
            final_confidence = 0
            final_position = 0
            decision_source = "hard_filter_block"
        
        # === Step 6: 置信度校准 ===
        # 将模型输出映射为真实概率
        calibrated_confidence = final_confidence
        calibrator = _get_calibrator()
        if calibrator and final_action != "HOLD":
            try:
                calib_result = calibrator.calibrate(final_confidence / 100.0)  # 转为0-1
                calibrated_confidence = calib_result.calibrated_confidence * 100  # 转回百分比
                logger.debug(f"Confidence calibrated: {final_confidence:.1f}% → {calibrated_confidence:.1f}%")
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
        
        return UnifiedDecision(
            action=final_action,
            confidence=calibrated_confidence,
            position_multiplier=final_position,
            decision_source=decision_source,
            signal_engine_result=signal_engine_dict,
            four_layer_result=four_layer_dict,
            hard_filter_result=hard_filter_dict,
            has_conflict=has_conflict,
            conflict_reason=conflict_reason,
        )


# 全局实例
_unified_engine: Optional[UnifiedDecisionEngine] = None


def get_unified_engine() -> UnifiedDecisionEngine:
    """获取统一决策引擎"""
    global _unified_engine
    if _unified_engine is None:
        _unified_engine = UnifiedDecisionEngine()
    return _unified_engine


def make_unified_decision(**kwargs) -> UnifiedDecision:
    """执行统一决策（便捷函数）"""
    engine = get_unified_engine()
    return engine.make_unified_decision(**kwargs)
