"""
数据层模块 (Layer 2)
"""

from .market_stream import (
    MultiExchangeManager,
    get_exchange_manager,
    generate_simulated_data,
    get_realtime_eth_data,
    get_orderbook_data,
    get_funding_rate,
)
from .kline_builder import calculate_indicators, get_latest_indicators
from .social_stream import (
    SocialStreamCollector,
    SentimentAnalyzer,
    SocialPost,
    social_sentiment_score,
)
from .feature_sync import (
    FeatureSyncLayer,
    FeatureMatrix,
    TimestampedFeature,
    get_feature_sync,
    sync_features,
)
from .trades_stream import (
    TradesStream,
    AggTrade,
    SimulatedTradesStream,
    get_trades_stream,
    get_real_cvd,
)

__all__ = [
    "MultiExchangeManager",
    "get_exchange_manager",
    "generate_simulated_data",
    "get_realtime_eth_data",
    "get_orderbook_data",
    "get_funding_rate",
    "calculate_indicators",
    "get_latest_indicators",
    "SocialStreamCollector",
    "SentimentAnalyzer",
    "SocialPost",
    "social_sentiment_score",
    # 特征同步层
    "FeatureSyncLayer",
    "FeatureMatrix",
    "TimestampedFeature",
    "get_feature_sync",
    "sync_features",
    # Trades数据流
    "TradesStream",
    "AggTrade",
    "SimulatedTradesStream",
    "get_trades_stream",
    "get_real_cvd",
]
