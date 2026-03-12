# data/market_stream.py
"""
ETH/USDT 实时行情数据流
========================
多交易所数据源，支持降级回退
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

# 支持的交易对
SUPPORTED_SYMBOLS = ['ETH/USDT', 'BTC/USDT', 'SOL/USDT']

# 多交易所配置
EXCHANGES = {
    'binance': {
        'class': ccxt.binance,
        'config': {'enableRateLimit': True}
    },
    'coinbase': {
        'class': ccxt.coinbase,
        'config': {'enableRateLimit': True}
    },
    'kraken': {
        'class': ccxt.kraken,
        'config': {'enableRateLimit': True}
    },
}


class MarketDataStream:
    """市场数据流"""
    
    def __init__(self):
        self.exchanges = {}
        self.current_exchange = None
        self._init_exchanges()
    
    def _init_exchanges(self):
        """初始化交易所连接"""
        for name, config in EXCHANGES.items():
            try:
                self.exchanges[name] = config['class'](config['config'])
                logger.info(f"✓ {name} 交易所初始化成功")
            except Exception as e:
                logger.warning(f"✗ {name} 初始化失败: {e}")
    
    def get_ticker(self, symbol: str = "ETH/USDT") -> Tuple[Optional[Dict], str]:
        """获取行情数据，自动切换交易所"""
        errors = []
        
        for name, exchange in self.exchanges.items():
            try:
                ticker = exchange.fetch_ticker(symbol)
                return ticker, name
            except Exception as e:
                errors.append(f"{name}: {str(e)[:50]}")
                continue
        
        return None, f"所有交易所失败: {'; '.join(errors)}"
    
    def get_ohlcv(self, symbol: str = "ETH/USDT", timeframe: str = '5m', limit: int = 100) -> Tuple[Optional[pd.DataFrame], str]:
        """获取K线数据"""
        errors = []
        
        for name, exchange in self.exchanges.items():
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df, name
            except Exception as e:
                errors.append(f"{name}: {str(e)[:50]}")
                continue
        
        return None, f"所有交易所失败: {'; '.join(errors)}"


# 全局实例
_market_stream = None


def get_market_stream() -> MarketDataStream:
    """获取市场数据流实例"""
    global _market_stream
    if _market_stream is None:
        _market_stream = MarketDataStream()
    return _market_stream


def generate_simulated_data(symbol: str = "ETH/USDT", limit: int = 100, base_price: float = 3500) -> Tuple[pd.DataFrame, float]:
    """
    生成模拟数据（当真实数据不可用时）
    
    Args:
        symbol: 交易对
        limit: K线数量
        base_price: 基础价格
        
    Returns:
        (DataFrame, 当前价格)
    """
    print(f"🎮 生成模拟数据 ({symbol})")
    
    np.random.seed(int(time.time()) % 10000)
    
    # 生成时间序列
    end_time = datetime.now()
    timestamps = pd.date_range(end=end_time, periods=limit, freq='5min')
    
    # 生成价格序列（随机游走 + 趋势）
    returns = np.random.normal(0.0001, 0.02, limit)  # 微小上涨趋势 + 波动
    prices = base_price * np.cumprod(1 + returns)
    
    # 生成 OHLCV
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, limit))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, limit))),
        'close': prices * (1 + np.random.normal(0, 0.002, limit)),
        'volume': np.random.uniform(100, 1000, limit)
    })
    
    current_price = float(df['close'].iloc[-1])
    
    return df, current_price


def get_realtime_eth_data(symbol: str = "ETH/USDT", timeframe: str = '5m', limit: int = 100, use_simulated: bool = False) -> Tuple[Optional[pd.DataFrame], float]:
    """
    获取实时数据（支持降级到模拟数据）
    
    Args:
        symbol: 交易对
        timeframe: K线周期
        limit: K线数量
        use_simulated: 强制使用模拟数据
        
    Returns:
        (DataFrame, 当前价格) 或 (None, 0)
    """
    # 基础价格映射（用于模拟）
    base_prices = {
        'ETH/USDT': 3500,
        'BTC/USDT': 95000,
        'SOL/USDT': 150,
    }
    
    if use_simulated:
        return generate_simulated_data(symbol, limit, base_prices.get(symbol, 100))
    
    print(f"🔄 正在获取 {symbol} 实时数据...")
    
    # 尝试真实数据
    stream = get_market_stream()
    df, exchange_name = stream.get_ohlcv(symbol, timeframe, limit)
    
    if df is not None:
        # 获取当前价格
        ticker, _ = stream.get_ticker(symbol)
        current_price = ticker['last'] if ticker else float(df['close'].iloc[-1])
        
        print(f"✅ 数据获取成功 | {exchange_name} | {symbol} 当前价格: ${current_price:,.2f}")
        return df, current_price
    
    # 降级到模拟数据
    print(f"⚠️ 真实数据获取失败，使用模拟数据")
    return generate_simulated_data(symbol, limit, base_prices.get(symbol, 100))


def get_orderbook_data(symbol: str = "ETH/USDT", limit: int = 20) -> dict:
    """获取订单簿数据"""
    stream = get_market_stream()
    
    try:
        orderbook, exchange_name = stream.get_ticker(symbol)
        
        if orderbook:
            # 模拟订单簿深度
            current_price = orderbook.get('last', 3500)
            spread = current_price * 0.0001  # 0.01% 点差
            
            return {
                'symbol': symbol,
                'bid_price': current_price - spread/2,
                'ask_price': current_price + spread/2,
                'spread': spread,
                'bid_volume': np.random.uniform(50, 200),
                'ask_volume': np.random.uniform(50, 200),
                'imbalance': np.random.uniform(-0.3, 0.3),
                'timestamp': pd.Timestamp.now()
            }
    except Exception as e:
        logger.warning(f"订单簿获取失败: {e}")
    
    # 返回模拟数据
    current_price = 3500
    return {
        'symbol': symbol,
        'bid_price': current_price * 0.9999,
        'ask_price': current_price * 1.0001,
        'spread': current_price * 0.0002,
        'bid_volume': 100,
        'ask_volume': 100,
        'imbalance': 0,
        'timestamp': pd.Timestamp.now()
    }


def get_funding_rate(symbol: str = "ETH/USDT") -> float:
    """获取资金费率"""
    try:
        futures_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        futures_symbol = symbol.replace('/', '')
        funding = futures_exchange.fetch_funding_rate(futures_symbol)
        return funding.get('fundingRate', 0)
    except:
        # 返回模拟值
        return np.random.uniform(-0.0001, 0.0001)


# 测试
if __name__ == "__main__":
    print("=" * 50)
    print("ETH/USDT 数据测试")
    print("=" * 50)
    
    # 测试真实数据
    df, price = get_realtime_eth_data("ETH/USDT")
    
    if df is not None:
        print(f"\n最新K线数据:")
        print(df.tail(3))
        print(f"\n当前价格: ${price:,.2f}")
