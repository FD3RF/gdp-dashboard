# data/market_stream.py
"""
ETH/USDT 实时行情数据流
========================
从 Binance 获取真实 ETH 行情数据
"""

import ccxt
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# 初始化交易所
exchange = ccxt.binance({
    'enableRateLimit': True,  # 启用速率限制
    'options': {
        'defaultType': 'spot',  # 现货市场
    }
})

# 支持的交易对
SUPPORTED_SYMBOLS = ['ETH/USDT', 'BTC/USDT', 'SOL/USDT']


def get_realtime_eth_data(symbol: str = "ETH/USDT", timeframe: str = '5m', limit: int = 100) -> Tuple[Optional[pd.DataFrame], float]:
    """
    获取 ETH/USDT 实时数据
    
    Args:
        symbol: 交易对，默认 ETH/USDT
        timeframe: K线周期，默认 5m
        limit: K线数量，默认 100
        
    Returns:
        (DataFrame, 当前价格) 或 (None, 0) 如果失败
    """
    # 严格限定为指定交易对
    if symbol not in SUPPORTED_SYMBOLS:
        symbol = "ETH/USDT"
    
    print(f"🔄 正在获取 {symbol} 实时数据...")
    
    try:
        # 1. 获取最新价格
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # 2. 获取 K 线数据
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 转换时间格式
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"✅ 数据获取成功 | {symbol} 当前价格: ${current_price:,.2f}")
        
        return df, current_price
        
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        return None, 0


def get_orderbook_data(symbol: str = "ETH/USDT", limit: int = 20) -> dict:
    """
    获取订单簿数据
    
    Args:
        symbol: 交易对
        limit: 深度限制
        
    Returns:
        订单簿数据字典
    """
    try:
        orderbook = exchange.fetch_order_book(symbol, limit)
        
        bids = orderbook['bids']  # 买单
        asks = orderbook['asks']  # 卖单
        
        # 计算买卖盘压力
        bid_volume = sum([bid[1] for bid in bids[:5]])
        ask_volume = sum([ask[1] for ask in asks[:5]])
        
        # 订单簿不平衡度
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
        
        return {
            'symbol': symbol,
            'bid_price': bids[0][0] if bids else 0,
            'ask_price': asks[0][0] if asks else 0,
            'spread': (asks[0][0] - bids[0][0]) if bids and asks else 0,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            'timestamp': pd.Timestamp.now()
        }
        
    except Exception as e:
        print(f"❌ 获取订单簿失败: {e}")
        return {}


def get_funding_rate(symbol: str = "ETH/USDT") -> float:
    """
    获取资金费率 (合约市场)
    
    Args:
        symbol: 交易对
        
    Returns:
        资金费率
    """
    try:
        # 切换到合约市场
        futures_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
        # 转换交易对格式 ETH/USDT -> ETHUSDT
        futures_symbol = symbol.replace('/', '')
        
        funding = futures_exchange.fetch_funding_rate(futures_symbol)
        return funding.get('fundingRate', 0)
        
    except Exception as e:
        print(f"⚠️ 获取资金费率失败: {e}")
        return 0.0


# 测试运行
if __name__ == "__main__":
    print("=" * 50)
    print("ETH/USDT 实时数据测试")
    print("=" * 50)
    
    df, price = get_realtime_eth_data()
    
    if df is not None:
        print(f"\n最新K线数据:")
        print(df.tail(5))
        print(f"\n当前价格: ${price:,.2f}")
        
        # 测试订单簿
        ob = get_orderbook_data()
        print(f"\n订单簿:")
        print(f"  买一: ${ob.get('bid_price', 0):,.2f}")
        print(f"  卖一: ${ob.get('ask_price', 0):,.2f}")
        print(f"  不平衡度: {ob.get('imbalance', 0):.4f}")
