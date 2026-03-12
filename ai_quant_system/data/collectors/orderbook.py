"""
Order Book Collector for market depth data.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from core.base import BaseModule
from core.exceptions import DataSourceException


@dataclass
class OrderBookLevel:
    """Represents a single level in the order book."""
    price: float
    amount: float
    total: float = 0.0  # Cumulative amount
    
    def to_dict(self) -> Dict[str, float]:
        return {'price': self.price, 'amount': self.amount, 'total': self.total}


@dataclass
class OrderBookSnapshot:
    """Snapshot of the order book at a point in time."""
    symbol: str
    exchange: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0
    
    @property
    def imbalance(self) -> float:
        """Calculate order book imbalance (-1 to 1)."""
        if not self.bids or not self.asks:
            return 0.0
        
        bid_volume = sum(b.amount for b in self.bids[:10])
        ask_volume = sum(a.amount for a in self.asks[:10])
        total = bid_volume + ask_volume
        
        if total == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat(),
            'spread': self.spread,
            'mid_price': self.mid_price,
            'imbalance': self.imbalance,
            'bids': [b.to_dict() for b in self.bids],
            'asks': [a.to_dict() for a in self.asks]
        }


class OrderBookCollector(BaseModule):
    """
    Collects and manages order book data from exchanges.
    Provides depth analysis and liquidity metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('orderbook_collector', config)
        self._orderbooks: Dict[str, OrderBookSnapshot] = {}
        self._callbacks: List[Callable] = []
        self._history: Dict[str, List[OrderBookSnapshot]] = {}
        self._history_size = self.config.get('history_size', 100)
        self._depth = self.config.get('depth', 20)
        self._market_data_collector = None
    
    def set_market_data_collector(self, collector) -> None:
        """Set reference to market data collector."""
        self._market_data_collector = collector
    
    async def initialize(self) -> bool:
        """Initialize the order book collector."""
        self.logger.info("Initializing order book collector...")
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
    
    async def fetch_orderbook(
        self,
        symbol: str,
        exchange: str = 'binance',
        depth: Optional[int] = None
    ) -> OrderBookSnapshot:
        """
        Fetch current order book snapshot.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            depth: Order book depth
        
        Returns:
            OrderBookSnapshot
        """
        if not self._market_data_collector:
            raise DataSourceException("Market data collector not set")
        
        depth = depth or self._depth
        
        try:
            data = await self._market_data_collector.fetch_order_book(
                symbol=symbol,
                limit=depth,
                exchange=exchange
            )
            
            # Convert to OrderBookSnapshot
            bids = []
            cumulative = 0.0
            for price, amount in data['bids']:
                cumulative += amount
                bids.append(OrderBookLevel(price=price, amount=amount, total=cumulative))
            
            asks = []
            cumulative = 0.0
            for price, amount in data['asks']:
                cumulative += amount
                asks.append(OrderBookLevel(price=price, amount=amount, total=cumulative))
            
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                exchange=exchange,
                timestamp=data['timestamp'],
                bids=bids,
                asks=asks
            )
            
            # Cache the snapshot
            key = f"{exchange}:{symbol}"
            self._orderbooks[key] = snapshot
            
            # Add to history
            if key not in self._history:
                self._history[key] = []
            self._history[key].append(snapshot)
            
            # Trim history
            if len(self._history[key]) > self._history_size:
                self._history[key] = self._history[key][-self._history_size:]
            
            return snapshot
            
        except Exception as e:
            raise DataSourceException(f"Failed to fetch order book: {e}")
    
    def get_cached_orderbook(self, symbol: str, exchange: str = 'binance') -> Optional[OrderBookSnapshot]:
        """Get cached order book snapshot."""
        key = f"{exchange}:{symbol}"
        return self._orderbooks.get(key)
    
    def calculate_liquidity(
        self,
        symbol: str,
        exchange: str = 'binance',
        depth_pct: float = 0.01
    ) -> Dict[str, float]:
        """
        Calculate liquidity metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            depth_pct: Price range percentage for liquidity calculation
        
        Returns:
            Dictionary with liquidity metrics
        """
        snapshot = self.get_cached_orderbook(symbol, exchange)
        
        if not snapshot or not snapshot.bids or not snapshot.asks:
            return {'bid_liquidity': 0, 'ask_liquidity': 0, 'total_liquidity': 0}
        
        mid_price = snapshot.mid_price
        price_range = mid_price * depth_pct
        
        bid_liquidity = sum(
            b.amount for b in snapshot.bids
            if b.price >= mid_price - price_range
        )
        ask_liquidity = sum(
            a.amount for a in snapshot.asks
            if a.price <= mid_price + price_range
        )
        
        return {
            'bid_liquidity': bid_liquidity,
            'ask_liquidity': ask_liquidity,
            'total_liquidity': bid_liquidity + ask_liquidity,
            'liquidity_imbalance': (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity + 1e-10)
        }
    
    def calculate_price_impact(
        self,
        symbol: str,
        amount: float,
        side: str,
        exchange: str = 'binance'
    ) -> Dict[str, float]:
        """
        Calculate price impact for a given trade size.
        
        Args:
            symbol: Trading symbol
            amount: Trade amount in base currency
            side: 'buy' or 'sell'
            exchange: Exchange name
        
        Returns:
            Dictionary with price impact metrics
        """
        snapshot = self.get_cached_orderbook(symbol, exchange)
        
        if not snapshot:
            return {'impact': 0, 'average_price': 0, 'worst_price': 0}
        
        if side.lower() == 'buy':
            levels = snapshot.asks
        else:
            levels = snapshot.bids
        
        if not levels:
            return {'impact': 0, 'average_price': 0, 'worst_price': 0}
        
        remaining = amount
        total_cost = 0.0
        worst_price = levels[0].price
        best_price = levels[0].price
        
        for level in levels:
            if remaining <= 0:
                break
            
            fill_amount = min(remaining, level.amount)
            total_cost += fill_amount * level.price
            worst_price = level.price
            remaining -= fill_amount
        
        filled = amount - remaining
        avg_price = total_cost / filled if filled > 0 else 0
        
        impact = abs(avg_price - best_price) / best_price if best_price > 0 else 0
        
        return {
            'impact': impact,
            'average_price': avg_price,
            'worst_price': worst_price,
            'filled': filled,
            'remaining': remaining
        }
    
    def get_orderbook_history(
        self,
        symbol: str,
        exchange: str = 'binance',
        limit: int = 10
    ) -> List[OrderBookSnapshot]:
        """Get order book history."""
        key = f"{exchange}:{symbol}"
        history = self._history.get(key, [])
        return history[-limit:]
    
    def analyze_depth(
        self,
        symbol: str,
        exchange: str = 'binance'
    ) -> Dict[str, Any]:
        """
        Analyze order book depth.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
        
        Returns:
            Depth analysis dictionary
        """
        snapshot = self.get_cached_orderbook(symbol, exchange)
        
        if not snapshot:
            return {}
        
        bid_volume = sum(b.amount for b in snapshot.bids)
        ask_volume = sum(a.amount for a in snapshot.asks)
        
        bid_value = sum(b.amount * b.price for b in snapshot.bids)
        ask_value = sum(a.amount * a.price for a in snapshot.asks)
        
        return {
            'symbol': symbol,
            'exchange': exchange,
            'timestamp': snapshot.timestamp.isoformat(),
            'spread': snapshot.spread,
            'spread_pct': snapshot.spread / snapshot.mid_price if snapshot.mid_price > 0 else 0,
            'mid_price': snapshot.mid_price,
            'imbalance': snapshot.imbalance,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bid_value': bid_value,
            'ask_value': ask_value,
            'depth_levels': len(snapshot.bids),
            'top_bid': snapshot.bids[0].price if snapshot.bids else None,
            'top_ask': snapshot.asks[0].price if snapshot.asks else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'healthy': self._initialized,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'cached_symbols': len(self._orderbooks)
        }
