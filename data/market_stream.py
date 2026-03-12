# data/market_stream.py
"""
Layer 2: 数据采集层 - 全息感知
==============================
多交易所冗余接入，防止单点故障
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timedelta
import time
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum

from config import EXCHANGE_CONFIG, BASE_PRICES

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """数据源类型"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRaken = "kraken"
    OKX = "okx"
    SIMULATED = "simulated"


@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    exchange: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    funding_rate: Optional[float] = None


class MultiExchangeManager:
    """
    多交易所管理器
    
    功能：
    - 模块4: 多交易所行情API冗余
    - 模块5: Tick数据抓取
    - 模块6: 订单簿数据
    - 自动故障转移
    """
    
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.current_primary: Optional[str] = None
        self._last_success: Dict[str, datetime] = {}
        self._error_count: Dict[str, int] = {}
        
        self._init_exchanges()
    
    def _init_exchanges(self) -> None:
        """初始化所有启用的交易所"""
        for exchange_id, config in EXCHANGE_CONFIG.items():
            if not config.get('enabled', False):
                continue
            
            try:
                exchange_class = getattr(ccxt, exchange_id)
                self.exchanges[exchange_id] = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 10000,
                    'options': {
                        'defaultType': 'spot',
                    }
                })
                logger.info(f"✓ {config['name']} 交易所初始化成功")
                
                if self.current_primary is None:
                    self.current_primary = exchange_id
                    
            except Exception as e:
                logger.warning(f"✗ {config['name']} 初始化失败: {e}")
    
    def _get_sorted_exchanges(self) -> List[str]:
        """获取按优先级排序的交易所列表"""
        sorted_exchanges = sorted(
            EXCHANGE_CONFIG.items(),
            key=lambda x: x[1].get('priority', 99)
        )
        return [ex_id for ex_id, _ in sorted_exchanges if ex_id in self.exchanges]
    
    def fetch_ticker(self, symbol: str) -> Tuple[Optional[Dict], str]:
        """
        获取行情数据，自动切换交易所
        
        Args:
            symbol: 交易对
            
        Returns:
            (ticker数据, 交易所名称)
        """
        exchanges = self._get_sorted_exchanges()
        
        for exchange_id in exchanges:
            try:
                exchange = self.exchanges[exchange_id]
                ticker = exchange.fetch_ticker(symbol)
                
                self._last_success[exchange_id] = datetime.now()
                self._error_count[exchange_id] = 0
                
                return ticker, exchange_id
                
            except Exception as e:
                self._error_count[exchange_id] = self._error_count.get(exchange_id, 0) + 1
                logger.debug(f"{exchange_id} 获取行情失败: {e}")
                continue
        
        return None, "所有交易所失败"
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '5m', 
                    limit: int = 100) -> Tuple[Optional[pd.DataFrame], str]:
        """
        获取K线数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            limit: 数量限制
            
        Returns:
            (DataFrame, 交易所名称)
        """
        exchanges = self._get_sorted_exchanges()
        
        for exchange_id in exchanges:
            try:
                exchange = self.exchanges[exchange_id]
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['exchange'] = exchange_id
                
                return df, exchange_id
                
            except Exception as e:
                logger.debug(f"{exchange_id} 获取K线失败: {e}")
                continue
        
        return None, "所有交易所失败"
    
    def fetch_orderbook(self, symbol: str, limit: int = 50) -> Tuple[Optional[Dict], str]:
        """
        获取订单簿数据
        
        Args:
            symbol: 交易对
            limit: 深度限制
            
        Returns:
            (订单簿数据, 交易所名称)
        """
        exchanges = self._get_sorted_exchanges()
        
        for exchange_id in exchanges:
            try:
                exchange = self.exchanges[exchange_id]
                orderbook = exchange.fetch_order_book(symbol, limit)
                
                return orderbook, exchange_id
                
            except Exception as e:
                logger.debug(f"{exchange_id} 获取订单簿失败: {e}")
                continue
        
        return None, "所有交易所失败"
    
    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """
        获取资金费率 (合约市场)
        
        Args:
            symbol: 交易对
            
        Returns:
            资金费率
        """
        # 尝试从Binance期货获取
        try:
            futures = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            futures_symbol = symbol.replace('/', '')
            funding = futures.fetch_funding_rate(futures_symbol)
            return funding.get('fundingRate', 0)
        except:
            pass
        
        # 尝试从OKX获取
        try:
            okx = ccxt.okx({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            funding = okx.fetch_funding_rate(symbol)
            return funding.get('fundingRate', 0)
        except:
            pass
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取交易所状态"""
        status = {}
        
        for exchange_id in self.exchanges:
            last_success = self._last_success.get(exchange_id)
            error_count = self._error_count.get(exchange_id, 0)
            
            status[exchange_id] = {
                'connected': True,
                'last_success': last_success.isoformat() if last_success else None,
                'error_count': error_count,
                'is_primary': exchange_id == self.current_primary
            }
        
        return status


# 全局管理器实例
_exchange_manager: Optional[MultiExchangeManager] = None


def get_exchange_manager() -> MultiExchangeManager:
    """获取全局交易所管理器"""
    global _exchange_manager
    if _exchange_manager is None:
        _exchange_manager = MultiExchangeManager()
    return _exchange_manager


def generate_simulated_data(symbol: str = "ETH/USDT", limit: int = 100, 
                            base_price: Optional[float] = None) -> Tuple[pd.DataFrame, float]:
    """
    生成模拟数据（当真实数据不可用时）
    
    Args:
        symbol: 交易对
        limit: K线数量
        base_price: 基础价格
        
    Returns:
        (DataFrame, 当前价格)
    """
    if base_price is None:
        base_price = BASE_PRICES.get(symbol, 100)
    
    print(f"🎮 生成模拟数据 ({symbol})")
    
    np.random.seed(int(time.time()) % 10000)
    
    # 生成时间序列
    end_time = datetime.now()
    timestamps = pd.date_range(end=end_time, periods=limit, freq='5min')
    
    # 生成价格序列（随机游走 + 微小趋势）
    returns = np.random.normal(0.0001, 0.015, limit)
    prices = base_price * np.cumprod(1 + returns)
    
    # 生成 OHLCV
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, limit))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, limit))),
        'close': prices * (1 + np.random.normal(0, 0.001, limit)),
        'volume': np.random.uniform(100, 1000, limit)
    })
    
    current_price = float(df['close'].iloc[-1])
    
    return df, current_price


def get_realtime_eth_data(symbol: str = "ETH/USDT", timeframe: str = '5m', 
                          limit: int = 100, use_simulated: bool = False) -> Tuple[Optional[pd.DataFrame], float]:
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
    if use_simulated:
        return generate_simulated_data(symbol, limit)
    
    print(f"🔄 正在获取 {symbol} 实时数据...")
    
    # 尝试真实数据
    manager = get_exchange_manager()
    df, exchange_name = manager.fetch_ohlcv(symbol, timeframe, limit)
    
    if df is not None:
        # 获取当前价格
        ticker, _ = manager.fetch_ticker(symbol)
        current_price = ticker['last'] if ticker else float(df['close'].iloc[-1])
        
        print(f"✅ 数据获取成功 | {exchange_name} | {symbol} 当前价格: ${current_price:,.2f}")
        return df, current_price
    
    # 降级到模拟数据
    print(f"⚠️ 真实数据获取失败，使用模拟数据")
    return generate_simulated_data(symbol, limit)


def get_orderbook_data(symbol: str = "ETH/USDT", limit: int = 50) -> Dict[str, Any]:
    """获取订单簿数据"""
    manager = get_exchange_manager()
    orderbook, exchange = manager.fetch_orderbook(symbol, limit)
    
    if orderbook:
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if bids and asks:
            return {
                'symbol': symbol,
                'exchange': exchange,
                'bids': bids,
                'asks': asks,
                'bid_price': bids[0][0] if bids else 0,
                'ask_price': asks[0][0] if asks else 0,
                'spread': asks[0][0] - bids[0][0] if bids and asks else 0,
                'timestamp': datetime.now()
            }
    
    # 返回模拟订单簿
    base_price = BASE_PRICES.get(symbol, 100)
    return {
        'symbol': symbol,
        'exchange': 'simulated',
        'bids': [[base_price * 0.9999 - i * 0.001, 1] for i in range(limit)],
        'asks': [[base_price * 1.0001 + i * 0.001, 1] for i in range(limit)],
        'bid_price': base_price * 0.9999,
        'ask_price': base_price * 1.0001,
        'spread': base_price * 0.0002,
        'timestamp': datetime.now()
    }


def get_funding_rate(symbol: str = "ETH/USDT") -> float:
    """获取资金费率"""
    manager = get_exchange_manager()
    rate = manager.fetch_funding_rate(symbol)
    return rate if rate is not None else np.random.uniform(-0.0001, 0.0001)


# 测试
if __name__ == "__main__":
    print("=" * 50)
    print("多交易所数据测试")
    print("=" * 50)
    
    manager = get_exchange_manager()
    print(f"\n交易所状态: {manager.get_status()}")
    
    df, price = get_realtime_eth_data("ETH/USDT")
    if df is not None:
        print(f"\n价格: ${price:,.2f}")
        print(df.tail(3))
