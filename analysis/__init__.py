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
    MetaFilter,
    AggregatedSignal,
    ModuleSignal,
    SignalDirection,
    SignalStrength,
    generate_unified_signal,
    get_signal_engine,
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
    "MetaFilter",
    "AggregatedSignal",
    "ModuleSignal",
    "SignalDirection",
    "SignalStrength",
    "generate_unified_signal",
    "get_signal_engine",
]
