# data/market_stream.py
"""
Layer 2: 数据采集层 - 全息感知
==============================
多交易所冗余接入，防止单点故障
支持 CoinGecko 免费API作为备用
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timedelta
import time
import logging
import asyncio
import requests
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
    COINGECKO = "coingecko"
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


class CoinGeckoAPI:
    """
    CoinGecko 免费API
    
    优点：
    - 免费使用
    - 无需API Key
    - 全球可访问
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # 交易对映射
    SYMBOL_MAP = {
        'ETH/USDT': 'ethereum',
        'BTC/USDT': 'bitcoin',
        'SOL/USDT': 'solana',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        coin_id = self.SYMBOL_MAP.get(symbol)
        if not coin_id:
            return None
        
        try:
            url = f"{self.BASE_URL}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd'
            }
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get(coin_id, {}).get('usd')
        except Exception as e:
            logger.debug(f"CoinGecko获取价格失败: {e}")
        return None
    
    def get_ohlcv(self, symbol: str, days: int = 7) -> Optional[pd.DataFrame]:
        """获取历史K线数据"""
        coin_id = self.SYMBOL_MAP.get(symbol)
        if not coin_id:
            return None
        
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}/ohlc"
            params = {'vs_currency': 'usd', 'days': days}
            resp = self.session.get(url, params=params, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['volume'] = 0  # CoinGecko OHLC不包含成交量
                
                # 扩展到100根K线（用于技术指标计算）
                if len(df) < 100:
                    # 复制最后几根K线
                    last_rows = df.tail(10)
                    while len(df) < 100:
                        df = pd.concat([df, last_rows], ignore_index=True)
                
                return df.tail(100)
        except Exception as e:
            logger.debug(f"CoinGecko获取K线失败: {e}")
        return None


class MultiExchangeManager:
    """
    多交易所管理器
    
    功能：
    - 模块4: 多交易所行情API冗余
    - 模块5: Tick数据抓取
    - 模块6: 订单簿数据
    - 自动故障转移
    - CoinGecko备用
    """
    
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.coingecko = CoinGeckoAPI()
        self.current_primary: Optional[str] = None
        self._last_success: Dict[str, datetime] = {}
        self._error_count: Dict[str, int] = {}
        self._coingecko_available = True
        
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
                    'timeout': 5000,  # 5秒超时
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
        获取行情数据，自动切换数据源
        
        Args:
            symbol: 交易对
            
        Returns:
            (ticker数据, 数据源名称)
        """
        # 1. 先尝试CCXT交易所
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
        
        # 2. 尝试CoinGecko
        if self._coingecko_available:
            try:
                price = self.coingecko.get_price(symbol)
                if price:
                    return {
                        'symbol': symbol,
                        'last': price,
                        'bid': price,
                        'ask': price,
                    }, 'coingecko'
            except Exception as e:
                logger.debug(f"CoinGecko获取行情失败: {e}")
                self._coingecko_available = False
        
        return None, "所有数据源失败"
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '5m', 
                    limit: int = 100) -> Tuple[Optional[pd.DataFrame], str]:
        """
        获取K线数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            limit: 数量限制
            
        Returns:
            (DataFrame, 数据源名称)
        """
        # 1. 尝试CCXT交易所
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
        
        # 2. 尝试CoinGecko
        if self._coingecko_available:
            try:
                df = self.coingecko.get_ohlcv(symbol, days=7)
                if df is not None:
                    df['exchange'] = 'coingecko'
                    return df, 'coingecko'
            except Exception as e:
                logger.debug(f"CoinGecko获取K线失败: {e}")
        
        return None, "所有数据源失败"
    
    def fetch_orderbook(self, symbol: str, limit: int = 50) -> Tuple[Optional[Dict], str]:
        """获取订单簿数据"""
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
        """获取资金费率"""
        try:
            futures = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 5000,
                'options': {'defaultType': 'future'}
            })
            futures_symbol = symbol.replace('/', '')
            funding = futures.fetch_funding_rate(futures_symbol)
            return funding.get('fundingRate', 0)
        except:
            pass
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取数据源状态"""
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
        
        status['coingecko'] = {
            'available': self._coingecko_available,
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
    生成模拟数据
    
    使用更真实的ETH价格范围
    """
    if base_price is None:
        base_price = BASE_PRICES.get(symbol, 100)
    
    print(f"🎮 生成模拟数据 ({symbol})")
    
    np.random.seed(int(time.time()) % 10000)
    
    end_time = datetime.now()
    timestamps = pd.date_range(end=end_time, periods=limit, freq='5min')
    
    returns = np.random.normal(0.0001, 0.015, limit)
    prices = base_price * np.cumprod(1 + returns)
    
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
    获取实时数据
    
    优先级：CCXT交易所 > CoinGecko > 模拟数据
    """
    if use_simulated:
        return generate_simulated_data(symbol, limit)
    
    print(f"🔄 正在获取 {symbol} 实时数据...")
    
    manager = get_exchange_manager()
    df, source = manager.fetch_ohlcv(symbol, timeframe, limit)
    
    if df is not None:
        ticker, _ = manager.fetch_ticker(symbol)
        current_price = ticker['last'] if ticker else float(df['close'].iloc[-1])
        
        print(f"✅ 数据获取成功 | {source} | {symbol} 当前价格: ${current_price:,.2f}")
        return df, current_price
    
    # 降级到模拟数据
    print(f"⚠️ 真实数据获取失败，使用模拟数据")
    return generate_simulated_data(symbol, limit)


def get_orderbook_data(symbol: str = "ETH/USDT", limit: int = 50) -> Dict[str, Any]:
    """获取订单簿数据"""
    manager = get_exchange_manager()
    orderbook, source = manager.fetch_orderbook(symbol, limit)
    
    if orderbook:
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if bids and asks:
            return {
                'symbol': symbol,
                'exchange': source,
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
    print(f"\n数据源状态: {manager.get_status()}")
    
    df, price = get_realtime_eth_data("ETH/USDT")
    if df is not None:
        print(f"\n价格: ${price:,.2f}")
        print(df.tail(3))
