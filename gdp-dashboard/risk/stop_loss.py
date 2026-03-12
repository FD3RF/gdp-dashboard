"""
Stop Loss Engine for risk management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from core.base import BaseModule


class StopLossEngine(BaseModule):
    """
    Stop loss management engine supporting multiple stop types.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('stop_loss_engine', config)
        
        self._default_stop_pct = self.config.get('default_stop_pct', 0.05)
        self._trailing_enabled = self.config.get('trailing_enabled', True)
        self._trailing_pct = self.config.get('trailing_pct', 0.03)
        
        # Active stops: position_id -> stop info
        self._active_stops: Dict[str, Dict] = {}
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    def set_stop(
        self,
        position_id: str,
        entry_price: float,
        position_type: str,
        stop_type: str = 'fixed',
        stop_value: Optional[float] = None,
        trailing_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Set a stop loss for a position.
        
        Args:
            position_id: Unique position identifier
            entry_price: Entry price
            position_type: 'long' or 'short'
            stop_type: 'fixed', 'trailing', 'atr', 'percentage'
            stop_value: Stop price or percentage
            trailing_pct: Trailing percentage for trailing stops
        
        Returns:
            Stop loss configuration
        """
        stop_info = {
            'position_id': position_id,
            'entry_price': entry_price,
            'position_type': position_type,
            'stop_type': stop_type,
            'created_at': datetime.now(),
            'triggered': False
        }
        
        if stop_type == 'fixed':
            if stop_value is None:
                stop_value = entry_price * (1 - self._default_stop_pct) if position_type == 'long' else \
                            entry_price * (1 + self._default_stop_pct)
            stop_info['stop_price'] = stop_value
            stop_info['initial_stop'] = stop_value
        
        elif stop_type == 'trailing':
            trailing_pct = trailing_pct or self._trailing_pct
            initial_stop = entry_price * (1 - trailing_pct) if position_type == 'long' else \
                          entry_price * (1 + trailing_pct)
            stop_info['stop_price'] = initial_stop
            stop_info['initial_stop'] = initial_stop
            stop_info['trailing_pct'] = trailing_pct
            stop_info['highest_price'] = entry_price
        
        elif stop_type == 'percentage':
            pct = stop_value or self._default_stop_pct
            stop_info['stop_price'] = entry_price * (1 - pct) if position_type == 'long' else \
                                     entry_price * (1 + pct)
            stop_info['initial_stop'] = stop_info['stop_price']
            stop_info['stop_pct'] = pct
        
        elif stop_type == 'atr':
            # ATR-based stop
            stop_info['stop_price'] = stop_value or entry_price
            stop_info['atr_stop'] = True
        
        self._active_stops[position_id] = stop_info
        return stop_info
    
    def update_stop(
        self,
        position_id: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Update stop loss (mainly for trailing stops).
        
        Args:
            position_id: Position identifier
            current_price: Current market price
        
        Returns:
            Updated stop info
        """
        if position_id not in self._active_stops:
            return {'error': 'Position not found'}
        
        stop_info = self._active_stops[position_id]
        
        if stop_info['stop_type'] == 'trailing':
            position_type = stop_info['position_type']
            trailing_pct = stop_info['trailing_pct']
            
            if position_type == 'long':
                # For longs, trail up on higher prices
                if current_price > stop_info['highest_price']:
                    stop_info['highest_price'] = current_price
                    new_stop = current_price * (1 - trailing_pct)
                    stop_info['stop_price'] = max(stop_info['stop_price'], new_stop)
            else:
                # For shorts, trail down on lower prices
                if current_price < stop_info.get('lowest_price', current_price):
                    stop_info['lowest_price'] = current_price
                    new_stop = current_price * (1 + trailing_pct)
                    stop_info['stop_price'] = min(stop_info['stop_price'], new_stop)
        
        return stop_info
    
    def check_stop(
        self,
        position_id: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Check if stop loss is triggered.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
        
        Returns:
            Stop check result
        """
        if position_id not in self._active_stops:
            return {'triggered': False, 'error': 'Position not found'}
        
        stop_info = self._active_stops[position_id]
        
        triggered = False
        if stop_info['position_type'] == 'long':
            triggered = current_price <= stop_info['stop_price']
        else:
            triggered = current_price >= stop_info['stop_price']
        
        if triggered:
            stop_info['triggered'] = True
            stop_info['triggered_at'] = datetime.now()
            stop_info['trigger_price'] = current_price
        
        return {
            'triggered': triggered,
            'stop_price': stop_info['stop_price'],
            'current_price': current_price,
            'position_id': position_id
        }
    
    def remove_stop(self, position_id: str) -> bool:
        """Remove stop loss for a position."""
        if position_id in self._active_stops:
            del self._active_stops[position_id]
            return True
        return False
    
    def get_stop(self, position_id: str) -> Optional[Dict]:
        """Get stop info for a position."""
        return self._active_stops.get(position_id)
    
    def get_all_stops(self) -> Dict[str, Dict]:
        """Get all active stops."""
        return self._active_stops.copy()
    
    def calculate_risk_amount(
        self,
        entry_price: float,
        stop_price: float,
        position_size: float
    ) -> float:
        """Calculate risk amount for a position."""
        return abs(entry_price - stop_price) * position_size
