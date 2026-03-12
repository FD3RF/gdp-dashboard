"""
Smart Order Router for optimal execution.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from core.base import BaseModule


class SmartOrderRouter(BaseModule):
    """
    Smart order routing for best execution.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('smart_router', config)
        
        self._exchanges: Dict[str, Any] = {}
        self._routing_rules = self.config.get('routing_rules', {})
        
        # Scoring weights
        self._price_weight = self.config.get('price_weight', 0.4)
        self._liquidity_weight = self.config.get('liquidity_weight', 0.3)
        self._fee_weight = self.config.get('fee_weight', 0.2)
        self._latency_weight = self.config.get('latency_weight', 0.1)
    
    def register_exchange(self, name: str, adapter) -> None:
        """Register an exchange."""
        self._exchanges[name] = adapter
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    async def find_best_route(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = 'market'
    ) -> Dict[str, Any]:
        """
        Find the best exchange/route for an order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Order type
        
        Returns:
            Best route information
        """
        if not self._exchanges:
            return {'exchange': 'default', 'score': 0}
        
        scores = {}
        
        for name, adapter in self._exchanges.items():
            score = await self._score_exchange(name, adapter, symbol, side, quantity)
            scores[name] = score
        
        # Select best exchange
        best_exchange = max(scores, key=scores.get)
        
        return {
            'exchange': best_exchange,
            'score': scores[best_exchange],
            'all_scores': scores
        }
    
    async def _score_exchange(
        self,
        name: str,
        adapter,
        symbol: str,
        side: str,
        quantity: float
    ) -> float:
        """Score an exchange for routing."""
        score = 0.0
        
        try:
            # Get orderbook
            if hasattr(adapter, 'get_orderbook'):
                ob = await adapter.get_orderbook(symbol)
            else:
                ob = {'bids': [], 'asks': []}
            
            # Price score (best price)
            if side == 'buy':
                best_price = ob['asks'][0][0] if ob['asks'] else 0
            else:
                best_price = ob['bids'][0][0] if ob['bids'] else 0
            
            # Normalize (higher is better for sells, lower for buys)
            price_score = 0.5  # Default
            score += price_score * self._price_weight
            
            # Liquidity score
            liquidity = sum(level[1] for level in (ob['bids'] + ob['asks'])[:5])
            liquidity_score = min(liquidity / quantity / 10, 1.0)
            score += liquidity_score * self._liquidity_weight
            
            # Fee score (lower is better)
            fee = self._get_exchange_fee(name)
            fee_score = 1.0 - min(fee / 0.01, 1.0)  # Normalize around 1%
            score += fee_score * self._fee_weight
            
            # Latency score (lower is better)
            latency = self._get_exchange_latency(name)
            latency_score = 1.0 - min(latency / 500, 1.0)  # Normalize around 500ms
            score += latency_score * self._latency_weight
            
        except Exception as e:
            self.logger.error(f"Error scoring exchange {name}: {e}")
            score = 0.1
        
        return score
    
    def _get_exchange_fee(self, name: str) -> float:
        """Get exchange fee."""
        fees = {
            'binance': 0.001,
            'bybit': 0.001,
            'okx': 0.0015,
            'coinbase': 0.005
        }
        return fees.get(name, 0.002)
    
    def _get_exchange_latency(self, name: str) -> float:
        """Get exchange latency in ms."""
        latencies = {
            'binance': 100,
            'bybit': 150,
            'okx': 200,
            'coinbase': 300
        }
        return latencies.get(name, 250)
    
    async def split_order(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        max_size_per_exchange: float = 10000
    ) -> List[Dict]:
        """
        Split order across multiple exchanges.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_quantity: Total quantity
            max_size_per_exchange: Maximum size per exchange
        
        Returns:
            List of order splits
        """
        routes = await self.find_best_route(symbol, side, total_quantity)
        
        if total_quantity <= max_size_per_exchange:
            return [{
                'exchange': routes['exchange'],
                'quantity': total_quantity
            }]
        
        # Split across exchanges
        splits = []
        remaining = total_quantity
        
        sorted_exchanges = sorted(
            self._exchanges.keys(),
            key=lambda x: self._get_exchange_capacity(x),
            reverse=True
        )
        
        for exchange in sorted_exchanges:
            if remaining <= 0:
                break
            
            fill_amount = min(remaining, max_size_per_exchange)
            splits.append({
                'exchange': exchange,
                'quantity': fill_amount
            })
            remaining -= fill_amount
        
        return splits
    
    def _get_exchange_capacity(self, name: str) -> float:
        """Get exchange capacity estimate."""
        return 10000.0  # Default capacity
