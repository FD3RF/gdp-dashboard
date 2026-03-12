"""
Volatility Filter for risk management.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from core.base import BaseModule


class VolatilityFilter(BaseModule):
    """
    Volatility filtering and monitoring system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('volatility_filter', config)
        
        self._lookback = self.config.get('lookback', 20)
        self._high_vol_threshold = self.config.get('high_vol_threshold', 0.05)
        self._low_vol_threshold = self.config.get('low_vol_threshold', 0.01)
        self._extreme_vol_threshold = self.config.get('extreme_vol_threshold', 0.10)
        
        self._volatility_history: Dict[str, List[float]] = {}
        self._current_state: Dict[str, str] = {}
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    def calculate_volatility(
        self,
        prices: pd.Series,
        method: str = 'returns'
    ) -> float:
        """
        Calculate volatility from price series.
        
        Args:
            prices: Price series
            method: 'returns', 'parkinson', 'garman_klass'
        
        Returns:
            Annualized volatility
        """
        if len(prices) < 2:
            return 0
        
        if method == 'returns':
            returns = prices.pct_change().dropna()
            vol = returns.std() * np.sqrt(252 * 24)  # Annualized for hourly
            return vol
        
        elif method == 'parkinson':
            # Parkinson volatility (uses high-low)
            if 'high' not in prices or 'low' not in prices:
                return self.calculate_volatility(prices, 'returns')
            
            hl_ratio = np.log(prices['high'] / prices['low'])
            vol = np.sqrt((hl_ratio ** 2).mean() / (4 * np.log(2)))
            return vol * np.sqrt(252 * 24)
        
        return 0
    
    def update(
        self,
        symbol: str,
        prices: pd.Series,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update volatility for a symbol.
        
        Args:
            symbol: Trading symbol
            prices: Recent price series
            timestamp: Optional timestamp
        
        Returns:
            Volatility status
        """
        timestamp = timestamp or datetime.now()
        
        # Calculate current volatility
        vol = self.calculate_volatility(prices.tail(self._lookback))
        
        # Store history
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = []
        self._volatility_history[symbol].append(vol)
        
        # Trim history
        if len(self._volatility_history[symbol]) > 100:
            self._volatility_history[symbol] = self._volatility_history[symbol][-100:]
        
        # Determine state
        state = self._determine_state(vol)
        self._current_state[symbol] = state
        
        return {
            'symbol': symbol,
            'volatility': vol,
            'state': state,
            'timestamp': timestamp,
            'high_threshold': self._high_vol_threshold,
            'low_threshold': self._low_vol_threshold
        }
    
    def _determine_state(self, vol: float) -> str:
        """Determine volatility state."""
        if vol >= self._extreme_vol_threshold:
            return 'extreme'
        elif vol >= self._high_vol_threshold:
            return 'high'
        elif vol <= self._low_vol_threshold:
            return 'low'
        else:
            return 'normal'
    
    def should_reduce_position(self, symbol: str) -> bool:
        """Check if position should be reduced due to volatility."""
        state = self._current_state.get(symbol, 'normal')
        return state in ('high', 'extreme')
    
    def should_avoid_trading(self, symbol: str) -> bool:
        """Check if trading should be avoided."""
        state = self._current_state.get(symbol, 'normal')
        return state == 'extreme'
    
    def get_position_size_adjustment(self, symbol: str) -> float:
        """Get position size adjustment factor based on volatility."""
        state = self._current_state.get(symbol, 'normal')
        
        if state == 'extreme':
            return 0.25  # Reduce to 25%
        elif state == 'high':
            return 0.5   # Reduce to 50%
        elif state == 'low':
            return 1.25  # Increase by 25%
        else:
            return 1.0   # No adjustment
    
    def get_volatility_rank(self, symbol: str) -> float:
        """Get percentile rank of current volatility."""
        if symbol not in self._volatility_history:
            return 0.5
        
        history = self._volatility_history[symbol]
        if len(history) < 5:
            return 0.5
        
        current = history[-1]
        rank = sum(1 for v in history if v < current) / len(history)
        return rank
    
    def get_status(self) -> Dict[str, Any]:
        """Get current volatility status for all symbols."""
        return {
            'states': self._current_state.copy(),
            'high_vol_symbols': [s for s, state in self._current_state.items() if state in ('high', 'extreme')],
            'thresholds': {
                'high': self._high_vol_threshold,
                'low': self._low_vol_threshold,
                'extreme': self._extreme_vol_threshold
            }
        }
