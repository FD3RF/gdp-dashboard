"""
Market Making Strategy.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig, Signal
from core.constants import OrderSide


class MarketMakingStrategy(BaseStrategy):
    """
    Market Making Strategy providing liquidity around the mid price.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        data_provider=None,
        order_executor=None,
        orderbook_collector=None
    ):
        super().__init__(config, data_provider, order_executor)
        
        self._orderbook_collector = orderbook_collector
        
        # Market making parameters
        self._spread_pct = self._config.parameters.get('spread_pct', 0.002)  # 0.2%
        self._order_size = self._config.parameters.get('order_size', 0.1)
        self._max_inventory = self._config.parameters.get('max_inventory', 1.0)
        self._inventory_skew = self._config.parameters.get('inventory_skew', 0.5)
        self._min_profit = self._config.parameters.get('min_profit', 0.001)
        self._volatility_adjust = self._config.parameters.get('volatility_adjust', True)
        
        # State
        self._inventory: Dict[str, float] = {}
        self._quotes: Dict[str, Dict] = {}
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate market making quotes/signals."""
        signals = []
        
        latest = data.iloc[-1] if len(data) > 0 else None
        if latest is None:
            return signals
        
        for symbol in self._config.symbols:
            try:
                # Get orderbook data
                mid_price = latest['close']
                
                if self._orderbook_collector:
                    ob = self._orderbook_collector.get_cached_orderbook(symbol)
                    if ob:
                        mid_price = ob.mid_price
                
                # Calculate volatility adjustment
                vol_adj = 1.0
                if self._volatility_adjust and len(data) > 20:
                    returns = data['close'].pct_change()
                    volatility = returns.rolling(20).std().iloc[-1]
                    vol_adj = 1 + (volatility * 10)  # Scale volatility
                
                # Get current inventory
                inventory = self._inventory.get(symbol, 0)
                
                # Adjust spread based on inventory
                inventory_skew = self._calculate_inventory_skew(inventory)
                
                # Calculate bid/ask prices
                half_spread = mid_price * self._spread_pct * vol_adj
                
                bid_price = mid_price - half_spread * (1 - inventory_skew)
                ask_price = mid_price + half_spread * (1 + inventory_skew)
                
                # Calculate order sizes
                bid_size = self._calculate_order_size(inventory, 'buy')
                ask_size = self._calculate_order_size(inventory, 'sell')
                
                # Store quotes
                self._quotes[symbol] = {
                    'bid': bid_price,
                    'ask': ask_price,
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'mid_price': mid_price,
                    'spread': ask_price - bid_price,
                    'spread_pct': (ask_price - bid_price) / mid_price,
                    'timestamp': datetime.now()
                }
                
                # Generate signals for both sides
                if abs(inventory) < self._max_inventory:
                    # Bid signal
                    if bid_size > 0:
                        signals.append(Signal(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            signal_type='market_making_bid',
                            strength=0.5,
                            price=bid_price,
                            quantity=bid_size,
                            metadata={
                                'quote_type': 'bid',
                                'spread_pct': self._quotes[symbol]['spread_pct'],
                                'inventory': inventory
                            }
                        ))
                    
                    # Ask signal
                    if ask_size > 0:
                        signals.append(Signal(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            signal_type='market_making_ask',
                            strength=0.5,
                            price=ask_price,
                            quantity=ask_size,
                            metadata={
                                'quote_type': 'ask',
                                'spread_pct': self._quotes[symbol]['spread_pct'],
                                'inventory': inventory
                            }
                        ))
                
            except Exception as e:
                self.logger.error(f"Error generating MM signals for {symbol}: {e}")
        
        return signals
    
    def _calculate_inventory_skew(self, inventory: float) -> float:
        """Calculate inventory skew for pricing."""
        if abs(inventory) >= self._max_inventory:
            return np.sign(inventory) * self._inventory_skew
        
        skew = (inventory / self._max_inventory) * self._inventory_skew
        return skew
    
    def _calculate_order_size(self, inventory: float, side: str) -> float:
        """Calculate order size based on inventory."""
        base_size = self._order_size
        
        # Reduce size if approaching inventory limit
        if side == 'buy':
            if inventory > 0:
                # Already long, reduce buy size
                reduction = inventory / self._max_inventory
                return base_size * (1 - reduction)
            else:
                # Short, increase buy size to balance
                return base_size * (1 + abs(inventory) / self._max_inventory)
        else:  # sell
            if inventory < 0:
                # Already short, reduce sell size
                reduction = abs(inventory) / self._max_inventory
                return base_size * (1 - reduction)
            else:
                # Long, increase sell size to balance
                return base_size * (1 + inventory / self._max_inventory)
    
    def update_inventory(self, symbol: str, quantity: float) -> None:
        """Update inventory after a fill."""
        if symbol not in self._inventory:
            self._inventory[symbol] = 0
        
        self._inventory[symbol] += quantity
    
    def get_quotes(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get current quotes."""
        if symbol:
            return self._quotes.get(symbol, {})
        return self._quotes
    
    def calculate_expected_profit(
        self,
        symbol: str,
        fills: int = 100
    ) -> Dict[str, float]:
        """Calculate expected profit from market making."""
        quote = self._quotes.get(symbol)
        if not quote:
            return {}
        
        spread = quote['spread']
        avg_size = self._order_size
        
        # Expected profit per fill (half spread)
        profit_per_fill = spread * avg_size / 2
        
        # Total expected profit
        total_profit = profit_per_fill * fills
        
        return {
            'spread': spread,
            'spread_pct': quote['spread_pct'],
            'profit_per_fill': profit_per_fill,
            'expected_profit': total_profit,
            'estimated_fills': fills
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        status = super().get_status()
        status['inventory'] = self._inventory
        status['quotes'] = {
            k: {**v, 'timestamp': v['timestamp'].isoformat()}
            for k, v in self._quotes.items()
        }
        return status
