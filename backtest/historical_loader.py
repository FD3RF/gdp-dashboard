"""
Historical Data Loader for backtesting.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from core.base import BaseModule
from core.exceptions import DataException


class HistoricalDataLoader(BaseModule):
    """
    Loads and manages historical market data for backtesting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('historical_loader', config)
        
        self._data_dir = Path(self.config.get('data_dir', 'data/historical'))
        self._cache: Dict[str, pd.DataFrame] = {}
        self._max_cache_size = self.config.get('max_cache_size', 100)
    
    async def initialize(self) -> bool:
        """Initialize the loader."""
        self.logger.info("Initializing historical data loader...")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the loader."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the loader."""
        self._cache.clear()
        self._running = False
        return True
    
    async def load_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1h',
        source: str = 'file'
    ) -> pd.DataFrame:
        """
        Load historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            source: Data source ('file', 'api', 'cache')
        
        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Load from file
        if source == 'file':
            df = await self._load_from_file(symbol, timeframe, start_date, end_date)
        else:
            df = await self._generate_synthetic(symbol, start_date, end_date, timeframe)
        
        if df is not None and len(df) > 0:
            # Filter date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Cache
            if len(self._cache) >= self._max_cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = df.copy()
        
        return df if df is not None else pd.DataFrame()
    
    async def _load_from_file(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load data from file."""
        filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        filepath = self._data_dir / filename
        
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                if isinstance(df.index, pd.DatetimeIndex):
                    return df
            except Exception as e:
                self.logger.error(f"Error loading {filepath}: {e}")
        
        # Generate synthetic data if file not found
        return await self._generate_synthetic(symbol, start_date, end_date, timeframe)
    
    async def _generate_synthetic(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Generate synthetic price data for testing."""
        # Time delta based on timeframe
        td_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
        }
        td = td_map.get(timeframe, timedelta(hours=1))
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq=td)
        n = len(timestamps)
        
        # Generate synthetic prices using random walk
        np.random.seed(hash(symbol) % (2**32))
        
        # Base price
        if 'BTC' in symbol:
            base_price = 50000
        elif 'ETH' in symbol:
            base_price = 3000
        else:
            base_price = 100
        
        # Random walk with drift
        returns = np.random.normal(0.0001, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': prices * (1 + np.random.uniform(0, 0.01, n)),
            'low': prices * (1 - np.random.uniform(0, 0.01, n)),
            'close': prices,
            'volume': np.random.uniform(100, 10000, n) * prices
        }, index=timestamps)
        
        # Ensure high >= open, close, low
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    async def save_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> bool:
        """Save data to file."""
        filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        filepath = self._data_dir / filename
        
        try:
            data.to_parquet(filepath)
            return True
        except Exception as e:
            self.logger.error(f"Error saving {filepath}: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in data directory."""
        symbols = []
        for f in self._data_dir.glob('*.parquet'):
            # Extract symbol from filename
            parts = f.stem.rsplit('_', 1)
            if len(parts) >= 2:
                symbols.append(parts[0].replace('_', '/'))
        return list(set(symbols))
