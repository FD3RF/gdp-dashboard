"""
Drawdown Protection for risk management.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
from core.base import BaseModule


class DrawdownProtection(BaseModule):
    """
    Drawdown protection and monitoring system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('drawdown_protection', config)
        
        self._max_drawdown = self.config.get('max_drawdown', 0.15)  # 15%
        self._warning_threshold = self.config.get('warning_threshold', 0.10)  # 10%
        self._halt_threshold = self.config.get('halt_threshold', 0.20)  # 20%
        
        self._equity_peak = 0
        self._current_drawdown = 0
        self._drawdown_history: List[Dict] = []
        self._halted = False
    
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
    
    def update(
        self,
        current_equity: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update drawdown calculation.
        
        Args:
            current_equity: Current portfolio equity
            timestamp: Optional timestamp
        
        Returns:
            Drawdown status
        """
        timestamp = timestamp or datetime.now()
        
        # Update peak
        if current_equity > self._equity_peak:
            self._equity_peak = current_equity
        
        # Calculate drawdown
        if self._equity_peak > 0:
            self._current_drawdown = (self._equity_peak - current_equity) / self._equity_peak
        else:
            self._current_drawdown = 0
        
        # Record history
        self._drawdown_history.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'peak': self._equity_peak,
            'drawdown': self._current_drawdown
        })
        
        # Trim history
        if len(self._drawdown_history) > 1000:
            self._drawdown_history = self._drawdown_history[-1000:]
        
        # Check thresholds
        status = self._check_thresholds()
        
        return {
            'current_equity': current_equity,
            'peak_equity': self._equity_peak,
            'drawdown': self._current_drawdown,
            'drawdown_pct': self._current_drawdown * 100,
            **status
        }
    
    def _check_thresholds(self) -> Dict[str, Any]:
        """Check drawdown thresholds and determine action."""
        status = {
            'warning': False,
            'halt': False,
            'level': 'normal'
        }
        
        if self._current_drawdown >= self._halt_threshold:
            status['halt'] = True
            status['level'] = 'critical'
            self._halted = True
        elif self._current_drawdown >= self._max_drawdown:
            status['warning'] = True
            status['level'] = 'danger'
        elif self._current_drawdown >= self._warning_threshold:
            status['warning'] = True
            status['level'] = 'warning'
        
        return status
    
    def reset_peak(self) -> None:
        """Reset equity peak (e.g., after a new capital injection)."""
        self._equity_peak = 0
    
    def get_max_drawdown(self, days: int = 30) -> float:
        """Get maximum drawdown over a period."""
        cutoff = datetime.now() - timedelta(days=days)
        relevant = [d for d in self._drawdown_history if d['timestamp'] >= cutoff]
        
        if not relevant:
            return 0
        
        return max(d['drawdown'] for d in relevant)
    
    def get_avg_drawdown(self, days: int = 30) -> float:
        """Get average drawdown over a period."""
        cutoff = datetime.now() - timedelta(days=days)
        relevant = [d for d in self._drawdown_history if d['timestamp'] >= cutoff]
        
        if not relevant:
            return 0
        
        return np.mean([d['drawdown'] for d in relevant])
    
    def get_drawdown_duration(self) -> timedelta:
        """Get current drawdown duration."""
        # Find when current drawdown started
        for i, record in enumerate(reversed(self._drawdown_history)):
            if record['drawdown'] == 0:
                return datetime.now() - record['timestamp']
        
        return timedelta(0)
    
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self._halted
    
    def resume(self) -> None:
        """Resume trading after halt."""
        self._halted = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current drawdown status."""
        return {
            'current_drawdown': self._current_drawdown,
            'peak_equity': self._equity_peak,
            'max_drawdown_limit': self._max_drawdown,
            'warning_threshold': self._warning_threshold,
            'halt_threshold': self._halt_threshold,
            'halted': self._halted
        }
