"""
Fee Model for backtesting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class FeeModelBase(ABC):
    """Base class for fee models."""
    
    @abstractmethod
    def calculate(self, trade_value: float) -> float:
        """Calculate fee for a trade."""
        pass


class PercentageFee(FeeModelBase):
    """Percentage-based fee model."""
    
    def __init__(self, fee_pct: float = 0.001):
        self.fee_pct = fee_pct
    
    def calculate(self, trade_value: float) -> float:
        return trade_value * self.fee_pct


class TieredFee(FeeModelBase):
    """Tiered fee model based on volume."""
    
    def __init__(self, tiers: list = None):
        # Default tiers: [(volume_threshold, fee_pct), ...]
        self.tiers = tiers or [
            (0, 0.002),      # 0-10k: 0.2%
            (10000, 0.0015), # 10k-50k: 0.15%
            (50000, 0.001),  # 50k+: 0.1%
        ]
    
    def calculate(self, trade_value: float) -> float:
        fee_pct = self.tiers[-1][1]  # Default to lowest tier
        
        for threshold, pct in reversed(self.tiers):
            if trade_value >= threshold:
                fee_pct = pct
                break
        
        return trade_value * fee_pct


class FixedFee(FeeModelBase):
    """Fixed fee per trade."""
    
    def __init__(self, fee_amount: float = 1.0):
        self.fee_amount = fee_amount
    
    def calculate(self, trade_value: float) -> float:
        return self.fee_amount


class MakerTakerFee(FeeModelBase):
    """Maker-taker fee model."""
    
    def __init__(
        self,
        maker_fee: float = 0.001,
        taker_fee: float = 0.0015,
        is_maker: bool = False
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.is_maker = is_maker
    
    def calculate(self, trade_value: float, is_maker: bool = None) -> float:
        if is_maker is None:
            is_maker = self.is_maker
        
        fee_pct = self.maker_fee if is_maker else self.taker_fee
        return trade_value * fee_pct


class CombinedFee(FeeModelBase):
    """Combined fee model (exchange + network)."""
    
    def __init__(
        self,
        exchange_fee: float = 0.001,
        network_fee: float = 0.0001
    ):
        self.exchange_fee = exchange_fee
        self.network_fee = network_fee
    
    def calculate(self, trade_value: float) -> float:
        return trade_value * (self.exchange_fee + self.network_fee)


def FeeModel(model_type: str = 'percentage', params: Dict[str, Any] = None):
    """Factory function for fee models."""
    params = params or {}
    
    models = {
        'percentage': PercentageFee,
        'tiered': TieredFee,
        'fixed': FixedFee,
        'maker_taker': MakerTakerFee,
        'combined': CombinedFee
    }
    
    model_class = models.get(model_type, PercentageFee)
    return model_class(**params)
