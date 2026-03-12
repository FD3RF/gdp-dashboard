"""
VWAP Engine for volume-weighted average price execution.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
from core.base import BaseModule


class VWAPEngine(BaseModule):
    """
    Volume-Weighted Average Price execution engine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('vwap_engine', config)
        
        self._order_manager = None
        self._data_provider = None
        self._active_vwaps: Dict[str, Dict] = {}
        self._default_duration = self.config.get('default_duration', 300)
    
    def set_order_manager(self, manager) -> None:
        """Set order manager."""
        self._order_manager = manager
    
    def set_data_provider(self, provider) -> None:
        """Set data provider for volume data."""
        self._data_provider = provider
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        for vwap_id in list(self._active_vwaps.keys()):
            await self.cancel_vwap(vwap_id)
        return True
    
    async def execute_vwap(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        duration: Optional[int] = None,
        volume_profile: Optional[List[float]] = None,
        participation_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Execute VWAP order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_quantity: Total quantity
            duration: Duration in seconds
            volume_profile: Historical volume profile
            participation_rate: Max participation rate
        
        Returns:
            VWAP execution info
        """
        import uuid
        vwap_id = str(uuid.uuid4())[:8]
        
        duration = duration or self._default_duration
        
        # Get volume profile
        if volume_profile is None:
            volume_profile = self._get_default_volume_profile(duration)
        
        # Normalize volume profile
        total_volume = sum(volume_profile)
        volume_weights = [v / total_volume for v in volume_profile]
        
        # Calculate slice quantities
        slices = []
        remaining = total_quantity
        num_slices = len(volume_profile)
        interval = duration / num_slices
        
        for i, weight in enumerate(volume_weights):
            if i == num_slices - 1:
                qty = remaining
            else:
                qty = total_quantity * weight
                remaining -= qty
            slices.append({
                'slice_num': i + 1,
                'weight': weight,
                'quantity': qty,
                'status': 'pending'
            })
        
        vwap_info = {
            'vwap_id': vwap_id,
            'symbol': symbol,
            'side': side,
            'total_quantity': total_quantity,
            'filled_quantity': 0,
            'total_value': 0,
            'achieved_vwap': 0,
            'duration': duration,
            'num_slices': num_slices,
            'interval': interval,
            'slices': slices,
            'participation_rate': participation_rate,
            'status': 'active',
            'created_at': datetime.now(),
            'completed_at': None
        }
        
        self._active_vwaps[vwap_id] = vwap_info
        
        # Start execution
        asyncio.create_task(self._execute_slices(vwap_id))
        
        return vwap_info
    
    def _get_default_volume_profile(self, duration: int) -> List[float]:
        """Get default U-shaped volume profile."""
        num_periods = max(duration // 30, 10)  # 30-second periods
        
        # U-shaped profile
        x = np.linspace(0, np.pi, num_periods)
        profile = np.sin(x) + 0.5
        return profile.tolist()
    
    async def _execute_slices(self, vwap_id: str) -> None:
        """Execute VWAP slices."""
        if vwap_id not in self._active_vwaps:
            return
        
        vwap = self._active_vwaps[vwap_id]
        
        for slice_info in vwap['slices']:
            if not self._running or vwap['status'] != 'active':
                break
            
            # Get current market volume
            market_volume = await self._get_market_volume(vwap['symbol'])
            
            # Adjust quantity based on participation rate
            max_qty = market_volume * vwap['participation_rate']
            adjusted_qty = min(slice_info['quantity'], max_qty)
            
            # Submit order
            if self._order_manager and adjusted_qty > 0:
                try:
                    order = await self._order_manager.submit_order(
                        symbol=vwap['symbol'],
                        side=vwap['side'],
                        quantity=adjusted_qty,
                        order_type='market'
                    )
                    
                    slice_info['order'] = order
                    slice_info['status'] = 'filled'
                    slice_info['filled_quantity'] = order.get('filled_quantity', adjusted_qty)
                    slice_info['filled_price'] = order.get('filled_price', 0)
                    
                    # Update VWAP tracking
                    if slice_info['filled_price'] and slice_info['filled_quantity']:
                        vwap['filled_quantity'] += slice_info['filled_quantity']
                        vwap['total_value'] += slice_info['filled_quantity'] * slice_info['filled_price']
                        vwap['achieved_vwap'] = vwap['total_value'] / vwap['filled_quantity']
                    
                except Exception as e:
                    self.logger.error(f"VWAP slice error: {e}")
                    slice_info['status'] = 'error'
                    slice_info['error'] = str(e)
            
            await asyncio.sleep(vwap['interval'])
        
        vwap['status'] = 'completed'
        vwap['completed_at'] = datetime.now()
    
    async def _get_market_volume(self, symbol: str) -> float:
        """Get current market volume."""
        # Default volume
        return 100.0
    
    async def cancel_vwap(self, vwap_id: str) -> Dict[str, Any]:
        """Cancel an active VWAP."""
        if vwap_id not in self._active_vwaps:
            return {'error': 'VWAP not found'}
        
        vwap = self._active_vwaps[vwap_id]
        vwap['status'] = 'cancelled'
        vwap['completed_at'] = datetime.now()
        
        return {
            'vwap_id': vwap_id,
            'status': 'cancelled',
            'filled_quantity': vwap['filled_quantity'],
            'achieved_vwap': vwap['achieved_vwap']
        }
    
    def get_vwap_status(self, vwap_id: str) -> Optional[Dict]:
        """Get VWAP status."""
        return self._active_vwaps.get(vwap_id)
    
    def get_active_vwaps(self) -> List[Dict]:
        """Get all active VWAPs."""
        return [v for v in self._active_vwaps.values() if v['status'] == 'active']
