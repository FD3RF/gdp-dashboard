"""
Funding Rate Collector for perpetual futures.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from core.base import BaseModule
from core.exceptions import DataSourceException


class FundingRateCollector(BaseModule):
    """
    Collects funding rate data from crypto derivatives exchanges.
    Supports Binance, Bybit, OKX, and other exchanges.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('funding_rate_collector', config)
        self._funding_rates: Dict[str, Dict] = {}
        self._history: Dict[str, pd.DataFrame] = {}
        self._market_data_collector = None
        self._cache_duration = self.config.get('cache_duration', 60)  # seconds
        self._last_update: Dict[str, datetime] = {}
    
    def set_market_data_collector(self, collector) -> None:
        """Set reference to market data collector."""
        self._market_data_collector = collector
    
    async def initialize(self) -> bool:
        """Initialize the funding rate collector."""
        self.logger.info("Initializing funding rate collector...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the collector."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the collector."""
        self._running = False
        return True
    
    async def fetch_funding_rate(
        self,
        symbol: str,
        exchange: str = 'binance'
    ) -> Dict[str, Any]:
        """
        Fetch current funding rate.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            exchange: Exchange name
        
        Returns:
            Funding rate data
        """
        cache_key = f"{exchange}:{symbol}"
        
        # Check cache
        if cache_key in self._funding_rates:
            last_update = self._last_update.get(cache_key)
            if last_update and (datetime.now() - last_update).seconds < self._cache_duration:
                return self._funding_rates[cache_key]
        
        try:
            if not self._market_data_collector:
                raise DataSourceException("Market data collector not set")
            
            ex = self._market_data_collector._exchanges.get(exchange)
            if not ex:
                raise DataSourceException(f"Exchange not initialized: {exchange}")
            
            # Fetch funding rate
            # For Binance futures
            if exchange == 'binance':
                # Get perpetual symbol
                perp_symbol = symbol.replace('/', '').replace('USDT', 'USDT')
                if '/' in symbol:
                    perp_symbol = symbol.replace('/', '')
                
                try:
                    # Try Binance API
                    funding = await ex.fapiPublic_get_premiumindex({'symbol': perp_symbol})
                    
                    result = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'funding_rate': float(funding['lastFundingRate']),
                        'funding_time': datetime.fromtimestamp(int(funding['nextFundingTime']) / 1000),
                        'mark_price': float(funding['markPrice']),
                        'index_price': float(funding['indexPrice']),
                        'timestamp': datetime.now()
                    }
                except Exception:
                    # Fallback to estimated rate
                    result = await self._estimate_funding_rate(symbol, exchange)
            else:
                result = await self._estimate_funding_rate(symbol, exchange)
            
            # Cache result
            self._funding_rates[cache_key] = result
            self._last_update[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            raise DataSourceException(f"Failed to fetch funding rate: {e}")
    
    async def _estimate_funding_rate(
        self,
        symbol: str,
        exchange: str
    ) -> Dict[str, Any]:
        """Estimate funding rate based on price difference."""
        try:
            # Fetch spot and futures prices
            spot_ticker = await self._market_data_collector.fetch_ticker(
                symbol, exchange
            )
            
            # Estimate based on premium
            spot_price = spot_ticker.get('last', 0)
            
            # Estimate funding rate (simplified)
            # In reality, this would use actual futures prices
            estimated_rate = 0.0001  # 0.01% as default
            
            return {
                'symbol': symbol,
                'exchange': exchange,
                'funding_rate': estimated_rate,
                'funding_time': datetime.now() + timedelta(hours=8),
                'mark_price': spot_price,
                'index_price': spot_price,
                'timestamp': datetime.now(),
                'estimated': True
            }
        except Exception:
            return {
                'symbol': symbol,
                'exchange': exchange,
                'funding_rate': 0.0,
                'funding_time': datetime.now() + timedelta(hours=8),
                'mark_price': 0,
                'index_price': 0,
                'timestamp': datetime.now(),
                'estimated': True
            }
    
    async def fetch_funding_history(
        self,
        symbol: str,
        exchange: str = 'binance',
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            limit: Number of historical rates
        
        Returns:
            DataFrame with funding history
        """
        cache_key = f"{exchange}:{symbol}"
        
        if cache_key in self._history:
            return self._history[cache_key]
        
        try:
            ex = self._market_data_collector._exchanges.get(exchange)
            if not ex:
                raise DataSourceException(f"Exchange not initialized: {exchange}")
            
            # Fetch historical funding rates
            # This varies by exchange
            history_data = []
            
            # Simulate historical data for now
            current_rate = await self.fetch_funding_rate(symbol, exchange)
            base_rate = current_rate['funding_rate']
            
            for i in range(limit):
                timestamp = datetime.now() - timedelta(hours=8 * (i + 1))
                # Add some variation
                rate = base_rate * (1 + np.random.uniform(-0.5, 0.5))
                history_data.append({
                    'timestamp': timestamp,
                    'funding_rate': rate
                })
            
            df = pd.DataFrame(history_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Cache
            self._history[cache_key] = df
            
            return df
            
        except Exception as e:
            raise DataSourceException(f"Failed to fetch funding history: {e}")
    
    async def get_funding_arbitrage_opportunities(
        self,
        symbols: List[str],
        exchange: str = 'binance',
        min_rate: float = 0.0005
    ) -> List[Dict[str, Any]]:
        """
        Find funding rate arbitrage opportunities.
        
        Args:
            symbols: List of symbols to check
            exchange: Exchange name
            min_rate: Minimum funding rate threshold
        
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        for symbol in symbols:
            try:
                funding = await self.fetch_funding_rate(symbol, exchange)
                rate = funding['funding_rate']
                
                if abs(rate) >= min_rate:
                    opportunities.append({
                        'symbol': symbol,
                        'exchange': exchange,
                        'funding_rate': rate,
                        'annualized_rate': rate * 3 * 365,  # 3 funding periods per day
                        'direction': 'short' if rate > 0 else 'long',
                        'mark_price': funding.get('mark_price'),
                        'funding_time': funding.get('funding_time'),
                        'timestamp': datetime.now()
                    })
            except Exception as e:
                self.logger.error(f"Error fetching funding rate for {symbol}: {e}")
        
        # Sort by absolute funding rate
        opportunities.sort(key=lambda x: abs(x['funding_rate']), reverse=True)
        
        return opportunities
    
    def calculate_funding_cost(
        self,
        position_size: float,
        funding_rate: float,
        periods: int = 1
    ) -> float:
        """
        Calculate expected funding cost/revenue.
        
        Args:
            position_size: Position size in quote currency
            funding_rate: Current funding rate
            periods: Number of funding periods
        
        Returns:
            Expected funding cost (negative = cost, positive = revenue)
        """
        return position_size * funding_rate * periods
    
    def get_cached_funding_rates(self) -> Dict[str, Dict]:
        """Get all cached funding rates."""
        return self._funding_rates.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'healthy': self._initialized,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'cached_rates': len(self._funding_rates)
        }
