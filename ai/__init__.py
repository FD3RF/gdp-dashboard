"""
AI 决策模块 (Layer 6-8)
"""

from .probability_model import ProbabilityResult, calculate_probabilities

# RL Agent 延迟导入
try:
    from .rl_agent import (
        PPOAgent,
        MarketState,
        TradingAction,
        SelfPlayTrainer,
        AdversarialAgent,
        rl_decision,
    )
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

__all__ = [
    "ProbabilityResult",
    "calculate_probabilities",
]

if RL_AVAILABLE:
    __all__.extend([
        "PPOAgent",
        "MarketState",
        "TradingAction",
        "SelfPlayTrainer",
        "AdversarialAgent",
        "rl_decision",
    ])
