"""
Data collectors module for AI Quant Trading System.
"""

from .market_data import MarketDataCollector
from .orderbook import OrderBookCollector
from .funding_rate import FundingRateCollector
from .onchain import OnChainDataCollector
from .news import NewsCollector
from .social_sentiment import SocialSentimentCollector
from .macro_data import MacroDataCollector

__all__ = [
    'MarketDataCollector',
    'OrderBookCollector',
    'FundingRateCollector',
    'OnChainDataCollector',
    'NewsCollector',
    'SocialSentimentCollector',
    'MacroDataCollector'
]
