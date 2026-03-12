# config.py
"""
Oracle AI Agent 配置中心
========================
12层架构系统配置
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class ExchangeType(Enum):
    """交易所类型"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRaken = "kraken"
    OKX = "okx"
    BYBIT = "bybit"


class MarketState(Enum):
    """市场状态"""
    TRENDING_UP = "趋势上涨"
    TRENDING_DOWN = "趋势下跌"
    RANGING = "震荡整理"
    EXTREME = "极端行情"
    LOW_LIQUIDITY = "流动性枯竭"


# 智能体参数
AGENT_CONFIG = {
    "state_dim": 256,      # 状态向量维度 (全息感知层输出)
    "action_dim": 4,       # 动作空间: [开多, 开空, 平仓, 观望]
    "hidden_dim": 512,     # 神经网络隐藏层维度
    "max_position": 0.2,   # 最大仓位比例 (凯利公式约束)
    "symbols": ["ETH/USDT", "BTC/USDT", "SOL/USDT"],
    "default_symbol": "ETH/USDT",
    "timeframe": "5m",     # 核心决策周期
}

# 风控参数
RISK_CONFIG = {
    "single_loss_limit": 0.02,   # 单笔最大亏损 2%
    "daily_loss_limit": 0.05,    # 日内最大亏损 5%
    "max_position": 0.2,         # 最大仓位 20%
    "max_leverage": 5,           # 最大杠杆
    "api_timeout": 500,          # API 超时熔断
    "black_swan_threshold": 0.1, # 黑天鹅阈值 10%
    "funding_rate_extreme": 0.001, # 资金费率极值
}

# 交易所配置
EXCHANGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "binance": {
        "name": "Binance",
        "enabled": True,
        "priority": 1,
        "rate_limit": 1200,  # 每分钟请求数
        "features": ["spot", "futures", "options"],
        "websocket": True,
    },
    "coinbase": {
        "name": "Coinbase Pro",
        "enabled": True,
        "priority": 2,
        "rate_limit": 600,
        "features": ["spot"],
        "websocket": True,
    },
    "kraken": {
        "name": "Kraken",
        "enabled": True,
        "priority": 3,
        "rate_limit": 500,
        "features": ["spot", "futures"],
        "websocket": True,
    },
    "okx": {
        "name": "OKX",
        "enabled": True,
        "priority": 2,
        "rate_limit": 1000,
        "features": ["spot", "futures", "swap"],
        "websocket": True,
    },
}

# 交易对配置
SYMBOLS = {
    "ETH/USDT": {
        "name": "Ethereum",
        "base_price": 3500,
        "tick_size": 0.01,
        "min_order": 0.001,
        "price_precision": 2,
    },
    "BTC/USDT": {
        "name": "Bitcoin",
        "base_price": 95000,
        "tick_size": 0.1,
        "min_order": 0.0001,
        "price_precision": 1,
    },
    "SOL/USDT": {
        "name": "Solana",
        "base_price": 150,
        "tick_size": 0.001,
        "min_order": 0.01,
        "price_precision": 3,
    },
}

# 技术指标参数
INDICATOR_CONFIG = {
    "ma_periods": [5, 10, 20, 60],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "std_dev": 2},
    "atr_period": 14,
    "hurst_window": 100,
}

# AI 模型参数
AI_CONFIG = {
    "confidence_threshold": {
        "high": 30,     # 高信心
        "medium": 15,   # 中信心
        "low": 0,       # 低信心
    },
    "signal_weights": {
        "trend": 0.25,
        "mean_reversion": 0.20,
        "breakout": 0.20,
        "volume": 0.15,
        "orderbook": 0.20,
    },
}

# 信号历史追踪
SIGNAL_TRACKING = {
    "history_size": 100,
    "min_samples": 10,
}

# 基础价格映射 (更新为更准确的当前价格)
BASE_PRICES = {
    'ETH/USDT': 3200,  # ETH当前价格约$3000-3500
    'BTC/USDT': 85000,  # BTC当前价格约$80,000-90,000
    'SOL/USDT': 140,   # SOL当前价格约$120-160
}
