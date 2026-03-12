"""
Liquidity Scanner for market liquidity analysis.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from core.base import BaseModule


class LiquidityScanner(BaseModule):
    """
    Scans and analyzes market liquidity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('liquidity_scanner', config)
        
        self._data_collectors: Dict[str, Any] = {}
        self._liquidity_cache: Dict[str, Dict] = {}
        self._scan_interval = self.config.get('scan_interval', 60)
    
    def register_collector(self, name: str, collector) -> None:
        """Register a data collector."""
        self._data_collectors[name] = collector
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    async def scan_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Scan liquidity for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Liquidity analysis
        """
        liquidity_info = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'bid_liquidity': 0,
            'ask_liquidity': 0,
            'total_liquidity': 0,
            'spread': 0,
            'spread_pct': 0,
            'depth': {},
            'score': 0
        }
        
        # Get orderbook data
        orderbook_collector = self._data_collectors.get('orderbook')
        if orderbook_collector:
            try:
                ob = await orderbook_collector.fetch_orderbook(symbol)
                
                bids = ob.get('bids', [])
                asks = ob.get('asks', [])
                
                if bids and asks:
                    liquidity_info['bid_liquidity'] = sum(b[1] for b in bids[:20])
                    liquidity_info['ask_liquidity'] = sum(a[1] for a in asks[:20])
                    liquidity_info['total_liquidity'] = liquidity_info['bid_liquidity'] + liquidity_info['ask_liquidity']
                    
                    liquidity_info['spread'] = asks[0][0] - bids[0][0]
                    liquidity_info['spread_pct'] = liquidity_info['spread'] / bids[0][0] * 100
                    
                    liquidity_info['depth'] = {
                        'bid_depth': len(bids),
                        'ask_depth': len(asks),
                        'bid_levels': [(b[0], b[1]) for b in bids[:5]],
                        'ask_levels': [(a[0], a[1]) for a in asks[:5]]
                    }
                    
                    # Calculate liquidity score
                    liquidity_info['score'] = self._calculate_liquidity_score(liquidity_info)
                
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
        
        self._liquidity_cache[symbol] = liquidity_info
        return liquidity_info
    
    def _calculate_liquidity_score(self, info: Dict) -> float:
        """Calculate liquidity score (0-100)."""
        score = 0
        
        # Total liquidity contribution (0-40)
        total_liq = info['total_liquidity']
        score += min(total_liq / 100, 40)
        
        # Spread contribution (0-30) - lower spread = higher score
        spread_pct = info['spread_pct']
        if spread_pct < 0.01:
            score += 30
        elif spread_pct < 0.05:
            score += 20
        elif spread_pct < 0.1:
            score += 10
        
        # Balance contribution (0-30)
        if info['total_liquidity'] > 0:
            imbalance = abs(info['bid_liquidity'] - info['ask_liquidity']) / info['total_liquidity']
            score += (1 - imbalance) * 30
        
        return min(score, 100)
    
    async def scan_multiple(self, symbols: List[str]) -> Dict[str, Dict]:
        """Scan multiple symbols."""
        results = {}
        for symbol in symbols:
            results[symbol] = await self.scan_symbol(symbol)
        return results
    
    def get_cached_liquidity(self, symbol: str) -> Optional[Dict]:
        """Get cached liquidity info."""
        return self._liquidity_cache.get(symbol)
    
    def find_liquid_pairs(
        self,
        min_liquidity: float = 1000,
        max_spread_pct: float = 0.1,
        limit: int = 10
    ) -> List[Dict]:
        """Find most liquid trading pairs."""
        liquid = []
        
        for symbol, info in self._liquidity_cache.items():
            if (info['total_liquidity'] >= min_liquidity and
                info['spread_pct'] <= max_spread_pct):
                liquid.append(info)
        
        liquid.sort(key=lambda x: x['score'], reverse=True)
        return liquid[:limit]
    
    def estimate_fill_price(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Dict[str, float]:
        """Estimate fill price for a quantity."""
        info = self._liquidity_cache.get(symbol)
        
        if not info or 'depth' not in info:
            return {'estimated_price': 0, 'impact': 0}
        
        levels = info['depth'].get('bid_levels' if side == 'sell' else 'ask_levels', [])
        
        if not levels:
            return {'estimated_price': 0, 'impact': 0}
        
        remaining = quantity
        total_cost = 0
        worst_price = levels[0][0]
        best_price = levels[0][0]
        
        for price, size in levels:
            if remaining <= 0:
                break
            
            fill = min(remaining, size)
            total_cost += fill * price
            worst_price = price
            remaining -= fill
        
        filled = quantity - remaining
        avg_price = total_cost / filled if filled > 0 else 0
        
        impact = abs(avg_price - best_price) / best_price if best_price > 0 else 0
        
        return {
            'estimated_price': avg_price,
            'worst_price': worst_price,
            'impact': impact,
            'fillable': filled >= quantity
        }
