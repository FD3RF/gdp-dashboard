"""
Order Manager for trade execution.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from core.base import BaseModule
from core.constants import OrderSide, OrderType


class OrderStatus(Enum):
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    PARTIAL = 'partial'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class OrderManager(BaseModule):
    """
    Central order management system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('order_manager', config)
        
        self._orders: Dict[str, Dict] = {}
        self._exchange_adapter = None
        self._callbacks: List[Callable] = []
        
        self._max_retries = self.config.get('max_retries', 3)
        self._retry_delay = self.config.get('retry_delay', 1.0)
    
    def set_exchange_adapter(self, adapter) -> None:
        """Set the exchange adapter."""
        self._exchange_adapter = adapter
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Submit a new order."""
        order_id = str(uuid.uuid4())[:12]
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side.value if hasattr(side, 'value') else side,
            'type': order_type.value if hasattr(order_type, 'value') else order_type,
            'quantity': quantity,
            'price': price,
            'stop_price': stop_price,
            'take_profit': take_profit,
            'leverage': leverage,
            'status': OrderStatus.PENDING.value,
            'filled_quantity': 0,
            'filled_price': 0,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'metadata': metadata or {}
        }
        
        self._orders[order_id] = order
        
        # Submit to exchange
        if self._exchange_adapter:
            try:
                result = await self._exchange_adapter.submit_order(order)
                order.update(result)
                order['status'] = OrderStatus.SUBMITTED.value
            except Exception as e:
                order['status'] = OrderStatus.REJECTED.value
                order['error'] = str(e)
        else:
            # Simulate fill
            order['status'] = OrderStatus.FILLED.value
            order['filled_quantity'] = quantity
            order['filled_price'] = price or 50000.0
        
        order['updated_at'] = datetime.now()
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
        
        return order
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        if order_id not in self._orders:
            return {'error': 'Order not found'}
        
        order = self._orders[order_id]
        
        if order['status'] in (OrderStatus.FILLED.value, OrderStatus.CANCELLED.value):
            return {'error': f"Cannot cancel order in {order['status']} status"}
        
        if self._exchange_adapter:
            try:
                await self._exchange_adapter.cancel_order(order_id)
            except Exception as e:
                return {'error': str(e)}
        
        order['status'] = OrderStatus.CANCELLED.value
        order['updated_at'] = datetime.now()
        
        return order
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        if order_id not in self._orders:
            return {'error': 'Order not found'}
        
        order = self._orders[order_id]
        
        # Update from exchange
        if self._exchange_adapter and order['status'] not in (OrderStatus.FILLED.value, OrderStatus.CANCELLED.value):
            try:
                status = await self._exchange_adapter.get_order_status(order_id)
                order.update(status)
                order['updated_at'] = datetime.now()
            except Exception as e:
                self.logger.error(f"Error getting order status: {e}")
        
        return order
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders."""
        open_statuses = [OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value, OrderStatus.PARTIAL.value]
        
        orders = [
            order for order in self._orders.values()
            if order['status'] in open_statuses
        ]
        
        if symbol:
            orders = [o for o in orders if o['symbol'] == symbol]
        
        return orders
    
    def get_order_history(self, limit: int = 100) -> List[Dict]:
        """Get order history."""
        orders = list(self._orders.values())
        orders.sort(key=lambda x: x['created_at'], reverse=True)
        return orders[:limit]
    
    def add_callback(self, callback: Callable) -> None:
        """Add order callback."""
        self._callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get order statistics."""
        orders = list(self._orders.values())
        
        return {
            'total_orders': len(orders),
            'filled': sum(1 for o in orders if o['status'] == OrderStatus.FILLED.value),
            'cancelled': sum(1 for o in orders if o['status'] == OrderStatus.CANCELLED.value),
            'rejected': sum(1 for o in orders if o['status'] == OrderStatus.REJECTED.value),
            'open': len(self.get_open_orders())
        }
