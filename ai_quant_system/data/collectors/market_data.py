"""
Market Data Collector for real-time and historical market data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from core.base import BaseModule
from core.constants import TimeFrame, DataType
from core.exceptions import DataSourceException, DataValidationException
from core.utils import async_retry, validate_symbol


class MarketDataCollector(BaseModule):
    """
    Collects real-time and historical market data from exchanges.
    Supports OHLCV, ticker, and trade data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('market_data_collector', config)
        self._exchanges: Dict[str, ccxt.Exchange] = {}
        self._callbacks: List[Callable] = []
        self._subscriptions: Dict[str, Dict] = {}
        self._data_buffer: Dict[str, List[Dict]] = {}
        self._buffer_size = self.config.get('buffer_size', 1000)
        
        # Default settings
        self._default_exchange = self.config.get('default_exchange', 'binance')
        self._default_timeframe = self.config.get('default_timeframe', '1h')
        self._rate_limit_delay = self.config.get('rate_limit_delay', 0.1)
    
    async def initialize(self) -> bool:
        """Initialize exchange connections."""
        self.logger.info("Initializing market data collector...")
        
        # Initialize default exchanges
        exchanges_config = self.config.get('exchanges', {})
        
        if not exchanges_config:
            # Add default Binance exchange
            exchanges_config['binance'] = {
                'apiKey': self.config.get('binance_api_key', ''),
                'secret': self.config.get('binance_api_secret', ''),
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            }
        
        for name, ex_config in exchanges_config.items():
            try:
                exchange_class = getattr(ccxt, name)
                self._exchanges[name] = exchange_class(ex_config)
                await self._exchanges[name].load_markets()
                self.logger.info(f"Initialized exchange: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize exchange {name}: {e}")
        
        self._initialized = len(self._exchanges) > 0
        return self._initialized
    
    async def start(self) -> bool:
        """Start the data collector."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the data collector and close connections."""
        self._running = False
        
        for name, exchange in self._exchanges.items():
            try:
                await exchange.close()
                self.logger.info(f"Closed exchange: {name}")
            except Exception as e:
                self.logger.error(f"Error closing exchange {name}: {e}")
        
        return True
    
    def add_callback(self, callback: Callable) -> None:
        """Add a callback for new data."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def _notify_callbacks(self, data: Dict[str, Any]) -> None:
        """Notify all callbacks with new data."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    @async_retry(max_attempts=3, delay=1.0)
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = None,
        since: Optional[int] = None,
        limit: int = 500,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '15m')
            since: Timestamp to fetch from
            limit: Number of candles
            exchange: Exchange name
        
        Returns:
            DataFrame with OHLCV data
        """
        exchange_name = exchange or self._default_exchange
        timeframe = timeframe or self._default_timeframe
        
        if exchange_name not in self._exchanges:
            raise DataSourceException(f"Exchange not initialized: {exchange_name}")
        
        if not validate_symbol(symbol):
            raise DataValidationException(f"Invalid symbol: {symbol}")
        
        ex = self._exchanges[exchange_name]
        
        try:
            ohlcv = await ex.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['timeframe'] = timeframe
            df.attrs['exchange'] = exchange_name
            
            return df
            
        except Exception as e:
            raise DataSourceException(
                f"Failed to fetch OHLCV: {e}",
                source=exchange_name
            )
    
    async def fetch_ticker(
        self,
        symbol: str,
        exchange: str = None
    ) -> Dict[str, Any]:
        """
        Fetch current ticker data.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
        
        Returns:
            Ticker data dictionary
        """
        exchange_name = exchange or self._default_exchange
        
        if exchange_name not in self._exchanges:
            raise DataSourceException(f"Exchange not initialized: {exchange_name}")
        
        ex = self._exchanges[exchange_name]
        
        try:
            ticker = await ex.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'exchange': exchange_name,
                'timestamp': datetime.now(),
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'last': ticker.get('last'),
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'volume': ticker.get('baseVolume'),
                'quote_volume': ticker.get('quoteVolume'),
                'vwap': ticker.get('vwap'),
                'open': ticker.get('open'),
                'close': ticker.get('close'),
                'change': ticker.get('change'),
                'percentage': ticker.get('percentage')
            }
        except Exception as e:
            raise DataSourceException(f"Failed to fetch ticker: {e}", source=exchange_name)
    
    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
        exchange: str = None
    ) -> Dict[str, Any]:
        """
        Fetch order book data.
        
        Args:
            symbol: Trading symbol
            limit: Depth limit
            exchange: Exchange name
        
        Returns:
            Order book dictionary
        """
        exchange_name = exchange or self._default_exchange
        
        if exchange_name not in self._exchanges:
            raise DataSourceException(f"Exchange not initialized: {exchange_name}")
        
        ex = self._exchanges[exchange_name]
        
        try:
            orderbook = await ex.fetch_order_book(symbol, limit)
            return {
                'symbol': symbol,
                'exchange': exchange_name,
                'timestamp': datetime.now(),
                'bids': orderbook['bids'],
                'asks': orderbook['asks']
            }
        except Exception as e:
            raise DataSourceException(f"Failed to fetch order book: {e}", source=exchange_name)
    
    async def fetch_trades(
        self,
        symbol: str,
        limit: int = 100,
        exchange: str = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent trades.
        
        Args:
            symbol: Trading symbol
            limit: Number of trades
            exchange: Exchange name
        
        Returns:
            List of trades
        """
        exchange_name = exchange or self._default_exchange
        
        if exchange_name not in self._exchanges:
            raise DataSourceException(f"Exchange not initialized: {exchange_name}")
        
        ex = self._exchanges[exchange_name]
        
        try:
            trades = await ex.fetch_trades(symbol, limit=limit)
            return [
                {
                    'id': t['id'],
                    'symbol': t['symbol'],
                    'timestamp': pd.to_datetime(t['timestamp'], unit='ms'),
                    'price': t['price'],
                    'amount': t['amount'],
                    'side': t['side'],
                    'exchange': exchange_name
                }
                for t in trades
            ]
        except Exception as e:
            raise DataSourceException(f"Failed to fetch trades: {e}", source=exchange_name)
    
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical data for a date range.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date (default: now)
            exchange: Exchange name
        
        Returns:
            DataFrame with historical data
        """
        exchange_name = exchange or self._default_exchange
        end_date = end_date or datetime.now()
        
        all_data = []
        current_date = start_date
        
        limit = 500
        timeframe_ms = {
            '1m': 60000,
            '5m': 300000,
            '15m': 900000,
            '30m': 1800000,
            '1h': 3600000,
            '4h': 14400000,
            '1d': 86400000
        }.get(timeframe, 3600000)
        
        while current_date < end_date:
            try:
                since_ms = int(current_date.timestamp() * 1000)
                df = await self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=limit,
                    exchange=exchange_name
                )
                
                if df.empty:
                    break
                
                all_data.append(df)
                
                # Move to next batch
                last_timestamp = df.index[-1]
                current_date = last_timestamp + timedelta(milliseconds=timeframe_ms)
                
                # Rate limiting
                await asyncio.sleep(self._rate_limit_delay)
                
            except Exception as e:
                self.logger.error(f"Error fetching historical data: {e}")
                break
        
        if all_data:
            result = pd.concat(all_data).drop_duplicates()
            result = result[~result.index.duplicated(keep='first')]
            return result.sort_index()
        
        return pd.DataFrame()
    
    async def subscribe_to_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable
    ) -> str:
        """
        Subscribe to real-time OHLCV updates.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            callback: Callback function
        
        Returns:
            Subscription ID
        """
        import uuid
        sub_id = str(uuid.uuid4())[:8]
        
        self._subscriptions[sub_id] = {
            'type': 'ohlcv',
            'symbol': symbol,
            'timeframe': timeframe,
            'callback': callback,
            'active': True
        }
        
        self.logger.info(f"Subscribed to OHLCV: {symbol} {timeframe}")
        return sub_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from data feed."""
        if subscription_id in self._subscriptions:
            self._subscriptions[subscription_id]['active'] = False
            del self._subscriptions[subscription_id]
            return True
        return False
    
    def get_supported_symbols(self, exchange: str = None) -> List[str]:
        """Get list of supported trading symbols."""
        exchange_name = exchange or self._default_exchange
        
        if exchange_name not in self._exchanges:
            return []
        
        return list(self._exchanges[exchange_name].markets.keys())
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        exchange_status = {}
        for name, ex in self._exchanges.items():
            try:
                await ex.fetch_ticker('BTC/USDT')
                exchange_status[name] = 'healthy'
            except Exception:
                exchange_status[name] = 'unhealthy'
        
        return {
            'healthy': all(v == 'healthy' for v in exchange_status.values()),
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'exchanges': exchange_status
        }
