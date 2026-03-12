"""
Exposure Control for risk management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from core.base import BaseModule


class ExposureControl(BaseModule):
    """
    Exposure and leverage control system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('exposure_control', config)
        
        self._max_leverage = self.config.get('max_leverage', 3.0)
        self._max_single_exposure = self.config.get('max_single_exposure', 0.2)
        self._max_sector_exposure = self.config.get('max_sector_exposure', 0.4)
        self._max_total_exposure = self.config.get('max_total_exposure', 1.0)
        
        self._positions: Dict[str, Dict] = {}
        self._sectors: Dict[str, List[str]] = {}
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    def set_sector_mapping(self, sectors: Dict[str, List[str]]) -> None:
        """Set symbol to sector mapping."""
        self._sectors = sectors
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        value: float,
        leverage: float = 1.0
    ) -> None:
        """Update position exposure."""
        self._positions[symbol] = {
            'quantity': quantity,
            'value': value,
            'leverage': leverage,
            'updated_at': datetime.now()
        }
    
    def remove_position(self, symbol: str) -> None:
        """Remove position."""
        if symbol in self._positions:
            del self._positions[symbol]
    
    def check_exposure(
        self,
        symbol: Optional[str] = None,
        additional_value: Optional[float] = None,
        portfolio_value: float = 1.0
    ) -> Dict[str, Any]:
        """
        Check if proposed trade would violate exposure limits.
        
        Args:
            symbol: Symbol to trade
            additional_value: Additional exposure value
            portfolio_value: Total portfolio value
        
        Returns:
            Exposure check result
        """
        result = {
            'allowed': True,
            'violations': [],
            'current_exposure': self._calculate_total_exposure(portfolio_value),
            'current_leverage': self._calculate_leverage()
        }
        
        # Check total exposure
        total_exp = result['current_exposure']
        if additional_value:
            new_total = total_exp + additional_value / portfolio_value
            if new_total > self._max_total_exposure:
                result['allowed'] = False
                result['violations'].append('max_total_exposure')
        
        # Check leverage
        if result['current_leverage'] > self._max_leverage:
            result['allowed'] = False
            result['violations'].append('max_leverage')
        
        # Check single exposure
        if symbol and additional_value:
            current_single = self._get_symbol_exposure(symbol, portfolio_value)
            new_single = current_single + additional_value / portfolio_value
            if new_single > self._max_single_exposure:
                result['allowed'] = False
                result['violations'].append('max_single_exposure')
        
        return result
    
    def _calculate_total_exposure(self, portfolio_value: float) -> float:
        """Calculate total exposure as % of portfolio."""
        if portfolio_value <= 0:
            return 0
        
        total = sum(abs(p['value']) for p in self._positions.values())
        return total / portfolio_value
    
    def _calculate_leverage(self) -> float:
        """Calculate portfolio leverage."""
        total_value = sum(abs(p['value']) for p in self._positions.values())
        margin_used = sum(abs(p['value']) / p.get('leverage', 1) for p in self._positions.values())
        
        if margin_used <= 0:
            return 0
        
        return total_value / margin_used
    
    def _get_symbol_exposure(self, symbol: str, portfolio_value: float) -> float:
        """Get exposure for a specific symbol."""
        if symbol not in self._positions:
            return 0
        
        return abs(self._positions[symbol]['value']) / portfolio_value
    
    def get_sector_exposure(self, portfolio_value: float) -> Dict[str, float]:
        """Calculate exposure by sector."""
        sector_exp = {}
        
        for symbol, pos in self._positions.items():
            sector = self._get_symbol_sector(symbol)
            if sector not in sector_exp:
                sector_exp[sector] = 0
            sector_exp[sector] += abs(pos['value']) / portfolio_value
        
        return sector_exp
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        for sector, symbols in self._sectors.items():
            if symbol in symbols:
                return sector
        return 'unknown'
    
    def get_exposure_summary(self, portfolio_value: float) -> Dict[str, Any]:
        """Get full exposure summary."""
        return {
            'total_exposure': self._calculate_total_exposure(portfolio_value),
            'leverage': self._calculate_leverage(),
            'positions_count': len(self._positions),
            'top_exposures': self._get_top_exposures(portfolio_value, 5),
            'sector_exposure': self.get_sector_exposure(portfolio_value),
            'limits': {
                'max_leverage': self._max_leverage,
                'max_single': self._max_single_exposure,
                'max_total': self._max_total_exposure
            }
        }
    
    def _get_top_exposures(self, portfolio_value: float, n: int) -> List[Dict]:
        """Get top N exposures."""
        exposures = [
            {
                'symbol': symbol,
                'value': pos['value'],
                'exposure_pct': abs(pos['value']) / portfolio_value
            }
            for symbol, pos in self._positions.items()
        ]
        
        return sorted(exposures, key=lambda x: x['exposure_pct'], reverse=True)[:n]
