"""
分析决策模块 (Layer 9-10)
"""

from .decision_maker import Signal, Decision, make_decision
from .trade_plan import TradePlan, generate_trade_plan
from .risk_monitor import RiskLevel, RiskWarning, risk_warning
from .risk_matrix import (
    RiskMatrix,
    RiskAssessment,
    RiskFactor,
    RiskLevel as MatrixRiskLevel,
    calculate_risk_matrix,
)
from .risk_filter import (
    HardRiskFilter,
    RiskFilterDecision,
    FilterResult,
    get_risk_filter,
    apply_hard_filter,
)
from .market_regime import (
    MarketRegime,
    MarketRegimeEngine,
    RegimeState,
    StrategyWeights,
    detect_market_regime,
)
from .signal_engine import (
    SignalEngine,
    SignalAggregator,
    AggregatedSignal,
    ModuleSignal,
    SignalDirection,
    SignalStrength,
    generate_unified_signal,
    get_signal_engine,
)
from .unified_filter import (
    UnifiedFilter,
    FilterDecision,
    FilterResult as UnifiedFilterResult,
)
from .layered_decision import (
    FourLayerDecisionEngine,
    FourLayerDecision,
    LayerResult,
    DecisionLayer,
    Direction,
    PositionSize,
    get_four_layer_engine,
    make_four_layer_decision,
)
from .unified_decision import (
    UnifiedDecisionEngine,
    UnifiedDecision,
    get_unified_engine,
    make_unified_decision,
)

from .position_manager import (
    KellyPositionManager,
    DynamicStopLoss,
    DynamicTakeProfit,
    PositionSizing,
    calculate_optimal_position,
)

__all__ = [
    "Signal",
    "Decision",
    "make_decision",
    "TradePlan",
    "generate_trade_plan",
    "RiskLevel",
    "RiskWarning",
    "risk_warning",
    "RiskMatrix",
    "RiskAssessment",
    "RiskFactor",
    "MatrixRiskLevel",
    "calculate_risk_matrix",
    # Risk Filter
    "HardRiskFilter",
    "RiskFilterDecision",
    "FilterResult",
    "get_risk_filter",
    "apply_hard_filter",
    # Market Regime
    "MarketRegime",
    "MarketRegimeEngine",
    "RegimeState",
    "StrategyWeights",
    "detect_market_regime",
    # Signal Engine
    "SignalEngine",
    "SignalAggregator",
    "AggregatedSignal",
    "ModuleSignal",
    "SignalDirection",
    "SignalStrength",
    "generate_unified_signal",
    "get_signal_engine",
    # Unified Filter
    "UnifiedFilter",
    "FilterDecision",
    "UnifiedFilterResult",
    # Layered Decision
    "FourLayerDecisionEngine",
    "FourLayerDecision",
    "LayerResult",
    "DecisionLayer",
    "Direction",
    "PositionSize",
    "get_four_layer_engine",
    "make_four_layer_decision",
    # Unified Decision
    "UnifiedDecisionEngine",
    "UnifiedDecision",
    "get_unified_engine",
    "make_unified_decision",
    # Position Management
    "KellyPositionManager",
    "DynamicStopLoss",
    "DynamicTakeProfit",
    "PositionSizing",
    "calculate_optimal_position",
]
