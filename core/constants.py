"""
Constants and enumerations for the AI Quant Trading System.
"""

from enum import Enum, auto
from typing import Dict, Any
import pandas as pd


class TimeFrame(Enum):
    """Supported timeframes for market data."""
    TICK = 'tick'
    S1 = '1s'
    S5 = '5s'
    S15 = '15s'
    S30 = '30s'
    M1 = '1m'
    M3 = '3m'
    M5 = '5m'
    M15 = '15m'
    M30 = '30m'
    H1 = '1h'
    H2 = '2h'
    H4 = '4h'
    H6 = '6h'
    H8 = '8h'
    H12 = '12h'
    D1 = '1d'
    D3 = '3d'
    W1 = '1w'
    M1_MONTH = '1M'
    
    @property
    def seconds(self) -> int:
        """Convert timeframe to seconds."""
        mapping = {
            TimeFrame.TICK: 0,
            TimeFrame.S1: 1,
            TimeFrame.S5: 5,
            TimeFrame.S15: 15,
            TimeFrame.S30: 30,
            TimeFrame.M1: 60,
            TimeFrame.M3: 180,
            TimeFrame.M5: 300,
            TimeFrame.M15: 900,
            TimeFrame.M30: 1800,
            TimeFrame.H1: 3600,
            TimeFrame.H2: 7200,
            TimeFrame.H4: 14400,
            TimeFrame.H6: 21600,
            TimeFrame.H8: 28800,
            TimeFrame.H12: 43200,
            TimeFrame.D1: 86400,
            TimeFrame.D3: 259200,
            TimeFrame.W1: 604800,
            TimeFrame.M1_MONTH: 2592000,
        }
        return mapping.get(self, 0)
    
    @property
    def pandas_freq(self) -> str:
        """Get pandas frequency string."""
        mapping = {
            TimeFrame.TICK: 'ms',
            TimeFrame.S1: 's',
            TimeFrame.S5: '5s',
            TimeFrame.S15: '15s',
            TimeFrame.S30: '30s',
            TimeFrame.M1: 'min',
            TimeFrame.M3: '3min',
            TimeFrame.M5: '5min',
            TimeFrame.M15: '15min',
            TimeFrame.M30: '30min',
            TimeFrame.H1: 'h',
            TimeFrame.H2: '2h',
            TimeFrame.H4: '4h',
            TimeFrame.H6: '6h',
            TimeFrame.H8: '8h',
            TimeFrame.H12: '12h',
            TimeFrame.D1: 'D',
            TimeFrame.D3: '3D',
            TimeFrame.W1: 'W',
            TimeFrame.M1_MONTH: 'M',
        }
        return mapping.get(self, 'min')


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = 'buy'
    SELL = 'sell'
    LONG = 'long'
    SHORT = 'short'
    
    def opposite(self) -> 'OrderSide':
        """Get opposite side."""
        if self in (OrderSide.BUY, OrderSide.LONG):
            return OrderSide.SELL
        return OrderSide.BUY


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    TRAILING_STOP = 'trailing_stop'
    TAKE_PROFIT = 'take_profit'
    TAKE_PROFIT_LIMIT = 'take_profit_limit'
    ICEBERG = 'iceberg'
    TWAP = 'twap'
    VWAP = 'vwap'


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = 'pending'
    OPEN = 'open'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'
    FAILED = 'failed'


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = 'long'
    SHORT = 'short'
    FLAT = 'flat'
    BOTH = 'both'


class StrategyStatus(Enum):
    """Strategy status enumeration."""
    CREATED = 'created'
    INITIALIZING = 'initializing'
    READY = 'ready'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    ERROR = 'error'
    BACKTESTING = 'backtesting'
    OPTIMIZING = 'optimizing'


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'
    
    @property
    def priority(self) -> int:
        """Get priority level (higher = more severe)."""
        priorities = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        return priorities[self]


class AlertType(Enum):
    """Alert type enumeration."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'
    TRADE = 'trade'
    RISK = 'risk'
    SYSTEM = 'system'


class ExchangeType(Enum):
    """Supported exchange types."""
    BINANCE_SPOT = 'binance_spot'
    BINANCE_FUTURES = 'binance_futures'
    BYBIT_SPOT = 'bybit_spot'
    BYBIT_FUTURES = 'bybit_futures'
    OKX_SPOT = 'okx_spot'
    OKX_FUTURES = 'okx_futures'
    COINBASE = 'coinbase'
    KRAKEN = 'kraken'
    GATEIO = 'gateio'
    HUOBI = 'huobi'


class DataType(Enum):
    """Data type enumeration."""
    OHLCV = 'ohlcv'
    TICKER = 'ticker'
    ORDERBOOK = 'orderbook'
    TRADE = 'trade'
    FUNDING_RATE = 'funding_rate'
    LIQUIDATION = 'liquidation'
    OPEN_INTEREST = 'open_interest'
    LONG_SHORT_RATIO = 'long_short_ratio'
    LIQUIDATIONS = 'liquidations'


class AgentTaskType(Enum):
    """Agent task type enumeration."""
    RESEARCH = 'research'
    STRATEGY_GENERATION = 'strategy_generation'
    BACKTEST = 'backtest'
    OPTIMIZATION = 'optimization'
    CODE_GENERATION = 'code_generation'
    CODE_REVIEW = 'code_review'
    RISK_ANALYSIS = 'risk_analysis'
    EXECUTION = 'execution'
    MONITORING = 'monitoring'
    SELF_IMPROVEMENT = 'self_improvement'


class MarketRegime(Enum):
    """Market regime enumeration."""
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGING = 'ranging'
    HIGH_VOLATILITY = 'high_volatility'
    LOW_VOLATILITY = 'low_volatility'
    BREAKOUT = 'breakout'


# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    'system': {
        'name': 'AI Quant Trading System',
        'version': '1.0.0',
        'environment': 'development',
        'debug': False
    },
    'database': {
        'host': 'localhost',
        'port': 5432,
        'name': 'quant_system',
        'user': 'quant_user',
        'pool_size': 10
    },
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'password': None
    },
    'ollama': {
        'host': 'localhost',
        'port': 11434,
        'model': 'qwen2.5-coder:latest',
        'timeout': 300
    },
    'risk': {
        'max_position_size_pct': 0.02,
        'max_portfolio_leverage': 3.0,
        'max_drawdown_pct': 0.15,
        'max_daily_loss_pct': 0.05,
        'max_single_trade_risk_pct': 0.01
    },
    'execution': {
        'default_slippage': 0.0005,
        'default_timeout': 30,
        'max_retries': 3
    }
}

# Common trading pairs
MAJOR_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
    'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT',
    'DOT/USDT', 'MATIC/USDT', 'LINK/USDT', 'ATOM/USDT'
]

# Technical indicator periods
INDICATOR_PERIODS = {
    'sma_fast': 10,
    'sma_slow': 20,
    'ema_fast': 12,
    'ema_slow': 26,
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2.0,
    'atr_period': 14,
    'adx_period': 14
}
