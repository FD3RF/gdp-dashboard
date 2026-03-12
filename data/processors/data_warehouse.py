"""
Data Warehouse for centralized data storage and retrieval.
"""

import asyncio
import logging
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from core.base import BaseModule
from core.exceptions import DataException


class DataWarehouse(BaseModule):
    """
    Centralized data storage system.
    Supports file-based storage with optional database integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('data_warehouse', config)
        self._data_dir = Path(self.config.get('data_dir', 'data/warehouse'))
        self._cache: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, Dict] = {}
        self._max_cache_size = self.config.get('max_cache_size', 100)
        self._compression = self.config.get('compression', 'snappy')
    
    async def initialize(self) -> bool:
        """Initialize the data warehouse."""
        self.logger.info("Initializing data warehouse...")
        
        # Create directories
        self._data_dir.mkdir(parents=True, exist_ok=True)
        (self._data_dir / 'ohlcv').mkdir(exist_ok=True)
        (self._data_dir / 'tickers').mkdir(exist_ok=True)
        (self._data_dir / 'features').mkdir(exist_ok=True)
        (self._data_dir / 'metadata').mkdir(exist_ok=True)
        
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the warehouse."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the warehouse."""
        # Save cache to disk
        await self._persist_cache()
        self._running = False
        return True
    
    async def _persist_cache(self) -> None:
        """Persist cached data to disk."""
        for key, df in self._cache.items():
            try:
                await self.save(key, df)
            except Exception as e:
                self.logger.error(f"Error persisting {key}: {e}")
    
    def _get_path(self, key: str, data_type: str = 'ohlcv') -> Path:
        """Get file path for a key."""
        return self._data_dir / data_type / f"{key}.parquet"
    
    async def save(
        self,
        key: str,
        data: pd.DataFrame,
        data_type: str = 'ohlcv',
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save data to warehouse.
        
        Args:
            key: Data key (e.g., 'BTC_USDT_1h')
            data: DataFrame to save
            data_type: Type of data
            metadata: Optional metadata
        
        Returns:
            Success status
        """
        try:
            path = self._get_path(key, data_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            data.to_parquet(
                path,
                compression=self._compression,
                index=True
            )
            
            # Update cache
            self._cache[key] = data
            if len(self._cache) > self._max_cache_size:
                # Remove oldest cached item
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            # Save metadata
            meta = {
                'key': key,
                'data_type': data_type,
                'rows': len(data),
                'columns': list(data.columns),
                'start_date': str(data.index.min()) if len(data) > 0 else None,
                'end_date': str(data.index.max()) if len(data) > 0 else None,
                'saved_at': datetime.now().isoformat(),
                **(metadata or {})
            }
            
            self._metadata[key] = meta
            meta_path = self._data_dir / 'metadata' / f"{key}.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            self.logger.debug(f"Saved data: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {key}: {e}")
            return False
    
    async def load(
        self,
        key: str,
        data_type: str = 'ohlcv',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load data from warehouse.
        
        Args:
            key: Data key
            data_type: Type of data
            start_date: Optional start date filter
            end_date: Optional end date filter
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame or None
        """
        # Check cache first
        if use_cache and key in self._cache:
            df = self._cache[key]
        else:
            path = self._get_path(key, data_type)
            
            if not path.exists():
                self.logger.warning(f"Data not found: {key}")
                return None
            
            try:
                df = pd.read_parquet(path)
                
                if use_cache:
                    self._cache[key] = df
            except Exception as e:
                self.logger.error(f"Error loading {key}: {e}")
                return None
        
        # Apply date filters
        if start_date and isinstance(df.index, pd.DatetimeIndex):
            df = df[df.index >= start_date]
        if end_date and isinstance(df.index, pd.DatetimeIndex):
            df = df[df.index <= end_date]
        
        return df
    
    async def delete(self, key: str, data_type: str = 'ohlcv') -> bool:
        """Delete data from warehouse."""
        try:
            path = self._get_path(key, data_type)
            
            if path.exists():
                path.unlink()
            
            meta_path = self._data_dir / 'metadata' / f"{key}.json"
            if meta_path.exists():
                meta_path.unlink()
            
            if key in self._cache:
                del self._cache[key]
            
            if key in self._metadata:
                del self._metadata[key]
            
            return True
        except Exception as e:
            self.logger.error(f"Error deleting {key}: {e}")
            return False
    
    async def exists(self, key: str, data_type: str = 'ohlcv') -> bool:
        """Check if data exists."""
        return self._get_path(key, data_type).exists()
    
    async def get_metadata(self, key: str) -> Optional[Dict]:
        """Get metadata for a key."""
        if key in self._metadata:
            return self._metadata[key]
        
        meta_path = self._data_dir / 'metadata' / f"{key}.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self._metadata[key] = meta
                return meta
        
        return None
    
    async def list_keys(self, data_type: Optional[str] = None) -> List[str]:
        """List all keys in warehouse."""
        keys = []
        
        if data_type:
            dir_path = self._data_dir / data_type
            if dir_path.exists():
                keys = [f.stem for f in dir_path.glob('*.parquet')]
        else:
            for dtype_dir in self._data_dir.iterdir():
                if dtype_dir.is_dir() and dtype_dir.name != 'metadata':
                    keys.extend(f.stem for f in dtype_dir.glob('*.parquet'))
        
        return keys
    
    async def get_data_range(
        self,
        key: str,
        data_type: str = 'ohlcv'
    ) -> Optional[tuple]:
        """Get date range of stored data."""
        meta = await self.get_metadata(key)
        
        if meta:
            start = meta.get('start_date')
            end = meta.get('end_date')
            
            if start and end:
                return (
                    pd.to_datetime(start),
                    pd.to_datetime(end)
                )
        
        return None
    
    async def append(
        self,
        key: str,
        data: pd.DataFrame,
        data_type: str = 'ohlcv'
    ) -> bool:
        """Append data to existing dataset."""
        existing = await self.load(key, data_type, use_cache=False)
        
        if existing is None:
            return await self.save(key, data, data_type)
        
        # Combine and deduplicate
        combined = pd.concat([existing, data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        
        return await self.save(key, combined, data_type)
    
    async def get_latest(
        self,
        key: str,
        data_type: str = 'ohlcv',
        n: int = 1
    ) -> Optional[pd.DataFrame]:
        """Get latest n rows."""
        df = await self.load(key, data_type)
        
        if df is not None and len(df) > 0:
            return df.tail(n)
        
        return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get warehouse statistics."""
        keys = await self.list_keys()
        
        total_size = 0
        for dtype_dir in self._data_dir.iterdir():
            if dtype_dir.is_dir():
                for f in dtype_dir.glob('*.parquet'):
                    total_size += f.stat().st_size
        
        return {
            'total_keys': len(keys),
            'cached_keys': len(self._cache),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'data_dir': str(self._data_dir)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        stats = await self.get_stats()
        return {
            'healthy': self._initialized,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        }
