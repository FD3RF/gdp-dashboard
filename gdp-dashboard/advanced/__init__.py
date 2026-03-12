"""
Advanced Institutional-Grade Modules
=====================================

Core modules for institutional quantitative trading:
- AI Strategy Auto-Evolution
- AI Market Prediction
- Cross-Exchange Arbitrage
- Auto Capital Management
- High-Frequency Trading
"""

from .ai_evolution import AIStrategyEvolution
from .market_prediction import AIMarketPredictor
from .cross_exchange_arb import CrossExchangeArbitrage
from .auto_capital import AutoCapitalManager
from .hft import HighFrequencyTrading

__all__ = [
    'AIStrategyEvolution',
    'AIMarketPredictor',
    'CrossExchangeArbitrage',
    'AutoCapitalManager',
    'HighFrequencyTrading'
]
