"""
TWAP Engine for time-weighted average price execution.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from core.base import BaseModule


class TWAPEngine(BaseModule):
    """
    Time-Weighted Average Price execution engine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('twap_engine', config)
        
        self._order_manager = None
        self._active_twaps: Dict[str, Dict] = {}
        self._default_duration = self.config.get('default_duration', 300)  # 5 minutes
        self._min_interval = self.config.get('min_interval', 10)  # 10 seconds
    
    def set_order_manager(self, manager) -> None:
        """Set order manager."""
        self._order_manager = manager
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        # Cancel active TWAPs
        for twap_id in list(self._active_twaps.keys()):
            await self.cancel_twap(twap_id)
        return True
    
    async def execute_twap(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        duration: Optional[int] = None,
        num_slices: Optional[int] = None,
        price_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute TWAP order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_quantity: Total quantity
            duration: Duration in seconds
            num_slices: Number of slices
            price_limit: Price limit for limit orders
        
        Returns:
            TWAP execution info
        """
        import uuid
        twap_id = str(uuid.uuid4())[:8]
        
        duration = duration or self._default_duration
        
        if num_slices is None:
            # Calculate slices based on duration and min interval
            num_slices = max(duration // self._min_interval, 2)
        
        slice_quantity = total_quantity / num_slices
        interval = duration / num_slices
        
        twap_info = {
            'twap_id': twap_id,
            'symbol': symbol,
            'side': side,
            'total_quantity': total_quantity,
            'filled_quantity': 0,
            'duration': duration,
            'num_slices': num_slices,
            'slice_quantity': slice_quantity,
            'interval': interval,
            'price_limit': price_limit,
            'slices': [],
            'status': 'active',
            'created_at': datetime.now(),
            'completed_at': None
        }
        
        self._active_twaps[twap_id] = twap_info
        
        # Start execution in background
        asyncio.create_task(self._execute_slices(twap_id))
        
        return twap_info
    
    async def _execute_slices(self, twap_id: str) -> None:
        """Execute TWAP slices."""
        if twap_id not in self._active_twaps:
            return
        
        twap = self._active_twaps[twap_id]
        
        for i in range(twap['num_slices']):
            if not self._running or twap['status'] != 'active':
                break
            
            # Submit slice order
            if self._order_manager:
                try:
                    order = await self._order_manager.submit_order(
                        symbol=twap['symbol'],
                        side=twap['side'],
                        quantity=twap['slice_quantity'],
                        order_type='market',
                        price=twap['price_limit']
                    )
                    
                    twap['slices'].append({
                        'slice_num': i + 1,
                        'order': order,
                        'timestamp': datetime.now()
                    })
                    
                    if order.get('filled_quantity'):
                        twap['filled_quantity'] += order['filled_quantity']
                    
                except Exception as e:
                    self.logger.error(f"TWAP slice error: {e}")
                    twap['slices'].append({
                        'slice_num': i + 1,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
            
            # Wait for next slice
            if i < twap['num_slices'] - 1:
                await asyncio.sleep(twap['interval'])
        
        # Mark as completed
        twap['status'] = 'completed'
        twap['completed_at'] = datetime.now()
    
    async def cancel_twap(self, twap_id: str) -> Dict[str, Any]:
        """Cancel an active TWAP."""
        if twap_id not in self._active_twaps:
            return {'error': 'TWAP not found'}
        
        twap = self._active_twaps[twap_id]
        twap['status'] = 'cancelled'
        twap['completed_at'] = datetime.now()
        
        return {
            'twap_id': twap_id,
            'status': 'cancelled',
            'filled_quantity': twap['filled_quantity'],
            'remaining': twap['total_quantity'] - twap['filled_quantity']
        }
    
    def get_twap_status(self, twap_id: str) -> Optional[Dict]:
        """Get TWAP status."""
        return self._active_twaps.get(twap_id)
    
    def get_active_twaps(self) -> List[Dict]:
        """Get all active TWAPs."""
        return [
            twap for twap in self._active_twaps.values()
            if twap['status'] == 'active'
        ]
