"""
Slippage Model for backtesting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from core.constants import OrderSide


class SlippageModel(ABC):
    """Base class for slippage models."""
    
    @abstractmethod
    def apply(self, price: float, side: OrderSide) -> float:
        """Apply slippage to price."""
        pass


class PercentageSlippage(SlippageModel):
    """Percentage-based slippage model."""
    
    def __init__(self, slippage_pct: float = 0.0005):
        self.slippage_pct = slippage_pct
    
    def apply(self, price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY or side.value == 'buy':
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)


class VolumeBasedSlippage(SlippageModel):
    """Volume-based slippage model."""
    
    def __init__(self, base_slippage: float = 0.0002, volume_factor: float = 0.0001):
        self.base_slippage = base_slippage
        self.volume_factor = volume_factor
    
    def apply(self, price: float, side: OrderSide, volume: float = 1000) -> float:
        # Higher volume = more slippage
        slippage = self.base_slippage + self.volume_factor * np.log10(volume)
        
        if side == OrderSide.BUY or side.value == 'buy':
            return price * (1 + slippage)
        else:
            return price * (1 - slippage)


class FixedSlippage(SlippageModel):
    """Fixed absolute slippage model."""
    
    def __init__(self, slippage_amount: float = 0.01):
        self.slippage_amount = slippage_amount
    
    def apply(self, price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY or side.value == 'buy':
            return price + self.slippage_amount
        else:
            return price - self.slippage_amount


class RandomSlippage(SlippageModel):
    """Random slippage model."""
    
    def __init__(self, min_slippage: float = 0.0001, max_slippage: float = 0.001):
        self.min_slippage = min_slippage
        self.max_slippage = max_slippage
    
    def apply(self, price: float, side: OrderSide) -> float:
        slippage = np.random.uniform(self.min_slippage, self.max_slippage)
        
        if side == OrderSide.BUY or side.value == 'buy':
            return price * (1 + slippage)
        else:
            return price * (1 - slippage)


def SlippageModel(model_type: str = 'percentage', params: Dict[str, Any] = None):
    """Factory function for slippage models."""
    params = params or {}
    
    models = {
        'percentage': PercentageSlippage,
        'volume': VolumeBasedSlippage,
        'fixed': FixedSlippage,
        'random': RandomSlippage
    }
    
    model_class = models.get(model_type, PercentageSlippage)
    return model_class(**params)
