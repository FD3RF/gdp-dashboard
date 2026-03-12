# execution/realtime_data.py
"""
实时数据模块 - 精准同步
======================

功能：
- 交易所数据获取
- WebSocket 实时推送
- 数据验证
- 缓存管理
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json

try:
    import ccxt.async_support as ccxt
except ImportError:
    import ccxt


@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    timestamp: datetime
    # 价格
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    # 行情
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    # 统计
    quote_volume: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'quote_volume': self.quote_volume,
        }


@dataclass
class OHLCV:
    """K线数据"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_list(self) -> List:
        return [
            self.timestamp.timestamp() * 1000,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
        ]


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_ticker(data: Dict) -> bool:
        """验证行情数据"""
        required_fields = ['symbol', 'last', 'bid', 'ask']
        for field in required_fields:
            if field not in data:
                return False
            if data[field] is None:
                return False
        
        # 价格合理性检查
        if data['bid'] > data['ask']:
            return False
        if data['last'] <= 0:
            return False
        
        return True
    
    @staticmethod
    def validate_ohlcv(data: List) -> bool:
        """验证K线数据"""
        if not data or len(data) < 6:
            return False
        
        # 数值检查
        try:
            for i, val in enumerate(data):
                if val is None:
                    return False
                if i in [1, 2, 3, 4, 5]:  # OHLCV
                    if float(val) < 0:
                        return False
            
            # 价格逻辑检查
            o, h, l, c = data[1], data[2], data[3], data[4]
            if not (l <= o <= h and l <= c <= h):
                return False
            
            return True
        except (ValueError, TypeError):
            return False


class RealtimeDataFeed:
    """
    实时数据源
    
    功能：
    - REST API 轮询
    - WebSocket 推送（如果支持）
    - 数据缓存
    - 自动重连
    """
    
    def __init__(self, exchange_config: Optional[Dict[str, Any]] = None):
        self.config = exchange_config or {
            'name': 'okx',
            'testnet': True,
        }
        self.logger = logging.getLogger("RealtimeDataFeed")
        
        # 交易所实例
        self.exchange: Optional[ccxt.Exchange] = None
        
        # 数据缓存
        self._ticker_cache: Dict[str, MarketData] = {}
        self._ohlcv_cache: Dict[str, List[OHLCV]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 5  # 秒
        
        # 验证器
        self.validator = DataValidator()
        
        # 回调
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # 状态
        self._running = False
        self._last_update: Dict[str, datetime] = {}
    
    async def connect(self) -> bool:
        """连接交易所"""
        try:
            exchange_name = self.config.get('name', 'okx')
            exchange_class = getattr(ccxt, exchange_name)
            
            exchange_config = {
                'enableRateLimit': True,
                'rateLimit': 100,
            }
            
            if self.config.get('testnet'):
                exchange_config['options'] = {'defaultType': 'swap'}
            
            self.exchange = exchange_class(exchange_config)
            await self.exchange.load_markets()
            
            self.logger.info(f"Connected to {exchange_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
    
    async def fetch_ticker(self, symbol: str) -> Optional[MarketData]:
        """获取行情"""
        if not self.exchange:
            return None
        
        # 检查缓存
        cache_key = f"ticker:{symbol}"
        if cache_key in self._cache_time:
            elapsed = (datetime.now() - self._cache_time[cache_key]).total_seconds()
            if elapsed < self._cache_ttl and cache_key in self._ticker_cache:
                return self._ticker_cache[cache_key]
        
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            
            if not self.validator.validate_ticker(ticker):
                self.logger.warning(f"Invalid ticker data: {symbol}")
                return None
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=ticker.get('open', ticker.get('last', 0)),
                high=ticker.get('high', ticker.get('last', 0)),
                low=ticker.get('low', ticker.get('last', 0)),
                close=ticker.get('last', 0),
                volume=ticker.get('baseVolume', 0),
                bid=ticker.get('bid', 0),
                ask=ticker.get('ask', 0),
                last=ticker.get('last', 0),
                quote_volume=ticker.get('quoteVolume', 0),
            )
            
            # 更新缓存
            self._ticker_cache[cache_key] = market_data
            self._cache_time[cache_key] = datetime.now()
            
            # 通知回调
            self._notify_callbacks('ticker', market_data)
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker {symbol}: {e}")
            return None
    
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '5m',
        limit: int = 100
    ) -> List[OHLCV]:
        """获取K线"""
        if not self.exchange:
            return []
        
        try:
            ohlcv_data = await self.exchange.fetch_ohlcv(
                symbol, 
                timeframe, 
                limit=limit
            )
            
            result = []
            for candle in ohlcv_data:
                if not self.validator.validate_ohlcv(candle):
                    continue
                
                result.append(OHLCV(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                ))
            
            # 更新缓存
            cache_key = f"ohlcv:{symbol}:{timeframe}"
            self._ohlcv_cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV {symbol}: {e}")
            return []
    
    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """获取订单簿"""
        if not self.exchange:
            return None
        
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            
            return {
                'symbol': symbol,
                'bids': orderbook.get('bids', [])[:limit],
                'asks': orderbook.get('asks', [])[:limit],
                'timestamp': datetime.now().isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching orderbook {symbol}: {e}")
            return None
    
    def get_cached_ticker(self, symbol: str) -> Optional[MarketData]:
        """获取缓存的行情"""
        return self._ticker_cache.get(f"ticker:{symbol}")
    
    def get_cached_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '5m'
    ) -> List[OHLCV]:
        """获取缓存的K线"""
        return self._ohlcv_cache.get(f"ohlcv:{symbol}:{timeframe}", [])
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册回调"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def _notify_callbacks(self, event_type: str, data: Any):
        """通知回调"""
        for callback in self._callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(data))
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")


class DataAggregator:
    """
    数据聚合器
    
    功能：
    - 多时间周期聚合
    - 技术指标计算
    - 数据标准化
    """
    
    @staticmethod
    def calculate_sma(data: List[float], period: int) -> List[float]:
        """计算简单移动平均"""
        if len(data) < period:
            return []
        
        result = []
        for i in range(period - 1, len(data)):
            avg = sum(data[i-period+1:i+1]) / period
            result.append(avg)
        
        return result
    
    @staticmethod
    def calculate_ema(data: List[float], period: int) -> List[float]:
        """计算指数移动平均"""
        if len(data) < period:
            return []
        
        multiplier = 2 / (period + 1)
        result = [sum(data[:period]) / period]  # SMA as first EMA
        
        for i in range(period, len(data)):
            ema = (data[i] - result[-1]) * multiplier + result[-1]
            result.append(ema)
        
        return result
    
    @staticmethod
    def calculate_rsi(closes: List[float], period: int = 14) -> List[float]:
        """计算 RSI"""
        if len(closes) < period + 1:
            return []
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        result = []
        if avg_loss == 0:
            result.append(100)
        else:
            rs = avg_gain / avg_loss
            result.append(100 - (100 / (1 + rs)))
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                result.append(100)
            else:
                rs = avg_gain / avg_loss
                result.append(100 - (100 / (1 + rs)))
        
        return result
    
    @staticmethod
    def calculate_macd(
        closes: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, List[float]]:
        """计算 MACD"""
        if len(closes) < slow_period + signal_period:
            return {'macd': [], 'signal': [], 'histogram': []}
        
        ema_fast = DataAggregator.calculate_ema(closes, fast_period)
        ema_slow = DataAggregator.calculate_ema(closes, slow_period)
        
        # MACD 线
        macd = [f - s for f, s in zip(ema_fast[-(len(ema_slow)-len(ema_fast)+1):], ema_slow)]
        
        # 信号线
        signal = DataAggregator.calculate_ema(macd, signal_period)
        
        # 柱状图
        histogram = [m - s for m, s in zip(macd[-len(signal):], signal)]
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram,
        }


# 导出
__all__ = [
    'MarketData',
    'OHLCV',
    'DataValidator',
    'RealtimeDataFeed',
    'DataAggregator',
]
