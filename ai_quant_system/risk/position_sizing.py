"""
Position Sizing for risk management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from core.base import BaseModule


class PositionSizing(BaseModule):
    """
    Position sizing module for calculating appropriate trade sizes.
    Supports Kelly Criterion, Fixed Fractional, and Volatility-based sizing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('position_sizing', config)
        
        self._method = self.config.get('method', 'fixed_fractional')
        self._risk_per_trade = self.config.get('risk_per_trade', 0.01)  # 1%
        self._max_position_pct = self.config.get('max_position_pct', 0.05)  # 5%
        self._kelly_fraction = self.config.get('kelly_fraction', 0.5)  # Half Kelly
    
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
    
    def calculate_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate position size based on configured method.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price
            stop_loss_price: Stop loss price
            volatility: Asset volatility
            win_rate: Historical win rate
            avg_win_loss_ratio: Average win/loss ratio
        
        Returns:
            Position sizing information
        """
        if self._method == 'fixed_fractional':
            size, risk_amount = self._fixed_fractional(
                portfolio_value, entry_price, stop_loss_price
            )
        elif self._method == 'kelly':
            size, risk_amount = self._kelly_criterion(
                portfolio_value, entry_price, win_rate, avg_win_loss_ratio
            )
        elif self._method == 'volatility':
            size, risk_amount = self._volatility_based(
                portfolio_value, entry_price, volatility
            )
        else:
            size, risk_amount = self._fixed_fractional(
                portfolio_value, entry_price, stop_loss_price
            )
        
        # Apply max position constraint
        max_size = (portfolio_value * self._max_position_pct) / entry_price
        size = min(size, max_size)
        
        return {
            'size': size,
            'value': size * entry_price,
            'risk_amount': risk_amount,
            'risk_pct': risk_amount / portfolio_value if portfolio_value > 0 else 0,
            'method': self._method
        }
    
    def _fixed_fractional(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: Optional[float]
    ) -> tuple:
        """Fixed fractional position sizing."""
        risk_amount = portfolio_value * self._risk_per_trade
        
        if stop_loss_price and entry_price != stop_loss_price:
            risk_per_share = abs(entry_price - stop_loss_price)
            size = risk_amount / risk_per_share
        else:
            # Default to 2% price movement as stop
            risk_per_share = entry_price * 0.02
            size = risk_amount / risk_per_share
        
        return size, risk_amount
    
    def _kelly_criterion(
        self,
        portfolio_value: float,
        entry_price: float,
        win_rate: Optional[float],
        avg_win_loss_ratio: Optional[float]
    ) -> tuple:
        """Kelly Criterion position sizing."""
        if win_rate is None:
            win_rate = 0.5
        if avg_win_loss_ratio is None:
            avg_win_loss_ratio = 1.0
        
        # Kelly formula: f = (bp - q) / b
        # b = avg_win_loss_ratio, p = win_rate, q = 1 - win_rate
        kelly_pct = (avg_win_loss_ratio * win_rate - (1 - win_rate)) / avg_win_loss_ratio
        
        # Apply Kelly fraction (typically half Kelly for safety)
        kelly_pct = max(0, kelly_pct * self._kelly_fraction)
        
        # Limit to max position
        kelly_pct = min(kelly_pct, self._max_position_pct)
        
        position_value = portfolio_value * kelly_pct
        size = position_value / entry_price
        risk_amount = position_value * self._risk_per_trade
        
        return size, risk_amount
    
    def _volatility_based(
        self,
        portfolio_value: float,
        entry_price: float,
        volatility: Optional[float]
    ) -> tuple:
        """Volatility-based position sizing."""
        if volatility is None:
            volatility = 0.02  # Default 2% volatility
        
        # Target volatility contribution
        target_vol = portfolio_value * self._risk_per_trade
        
        # Position size based on volatility
        # Higher volatility = smaller position
        size = target_vol / (entry_price * volatility)
        risk_amount = size * entry_price * volatility
        
        return size, risk_amount
    
    def calculate_leverage_adjusted_size(
        self,
        base_size: float,
        leverage: float,
        max_leverage: float = 3.0
    ) -> Dict[str, Any]:
        """Calculate leverage-adjusted position size."""
        leverage = min(leverage, max_leverage)
        
        return {
            'size': base_size * leverage,
            'effective_leverage': leverage,
            'margin_required': base_size / leverage
        }
