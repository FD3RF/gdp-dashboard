"""
Funding Rate Arbitrage Strategy.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig, Signal
from core.constants import OrderSide


class FundingRateArbitrageStrategy(BaseStrategy):
    """
    Funding Rate Arbitrage Strategy for perpetual futures.
    Captures funding payments while hedging price exposure.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        data_provider=None,
        order_executor=None,
        funding_rate_collector=None
    ):
        super().__init__(config, data_provider, order_executor)
        
        self._funding_rate_collector = funding_rate_collector
        
        # Funding arb parameters
        self._min_funding_rate = self._config.parameters.get('min_funding_rate', 0.0005)
        self._entry_threshold = self._config.parameters.get('entry_threshold', 0.001)
        self._exit_threshold = self._config.parameters.get('exit_threshold', 0.0001)
        self._position_duration_hours = self._config.parameters.get('position_duration', 8)
        
        # Track funding positions
        self._funding_positions: Dict[str, Dict] = {}
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate funding rate arbitrage signals."""
        signals = []
        
        if not self._funding_rate_collector:
            return signals
        
        for symbol in self._config.symbols:
            try:
                # Get current funding rate
                funding_data = await self._funding_rate_collector.fetch_funding_rate(symbol)
                funding_rate = funding_data.get('funding_rate', 0)
                
                current_position = self._positions.get(symbol, {}).get('quantity', 0)
                
                # Annualized rate for comparison
                annualized = funding_rate * 3 * 365  # 3 funding periods per day
                
                # Entry logic
                if abs(funding_rate) >= self._min_funding_rate and current_position == 0:
                    if funding_rate > 0:
                        # Positive funding - short futures, long spot (delta neutral)
                        # For simplicity, just short futures
                        signals.append(Signal(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            signal_type='entry',
                            strength=min(abs(funding_rate) * 100, 1.0),
                            price=funding_data.get('mark_price'),
                            metadata={
                                'strategy': 'funding_arb',
                                'funding_rate': funding_rate,
                                'annualized_rate': annualized,
                                'direction': 'short_perp',
                                'expected_return': funding_rate * 3 * 100  # Daily %
                            }
                        ))
                    else:
                        # Negative funding - long futures, short spot
                        signals.append(Signal(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            signal_type='entry',
                            strength=min(abs(funding_rate) * 100, 1.0),
                            price=funding_data.get('mark_price'),
                            metadata={
                                'strategy': 'funding_arb',
                                'funding_rate': funding_rate,
                                'annualized_rate': annualized,
                                'direction': 'long_perp',
                                'expected_return': abs(funding_rate) * 3 * 100
                            }
                        ))
                
                # Exit logic
                elif current_position != 0:
                    position = self._funding_positions.get(symbol, {})
                    entry_time = position.get('entry_time')
                    
                    should_exit = False
                    exit_reason = ''
                    
                    # Exit if funding rate drops below threshold
                    if abs(funding_rate) < self._exit_threshold:
                        should_exit = True
                        exit_reason = 'funding_rate_normalized'
                    
                    # Exit after position duration
                    elif entry_time:
                        duration = datetime.now() - entry_time
                        if duration > timedelta(hours=self._position_duration_hours):
                            should_exit = True
                            exit_reason = 'duration_expired'
                    
                    if should_exit:
                        exit_side = OrderSide.BUY if current_position < 0 else OrderSide.SELL
                        signals.append(Signal(
                            symbol=symbol,
                            side=exit_side,
                            signal_type='exit',
                            strength=1.0,
                            price=funding_data.get('mark_price'),
                            metadata={'reason': exit_reason}
                        ))
            
            except Exception as e:
                self.logger.error(f"Error processing funding arb for {symbol}: {e}")
        
        return signals
    
    def update_funding_position(
        self,
        symbol: str,
        quantity: float,
        funding_rate: float
    ) -> None:
        """Update funding position tracking."""
        if quantity != 0:
            self._funding_positions[symbol] = {
                'quantity': quantity,
                'entry_time': datetime.now(),
                'entry_funding_rate': funding_rate,
                'payments_collected': 0
            }
        else:
            if symbol in self._funding_positions:
                del self._funding_positions[symbol]
    
    async def calculate_expected_pnl(
        self,
        symbol: str,
        position_size: float,
        duration_hours: int = 8
    ) -> Dict[str, float]:
        """Calculate expected PnL from funding arbitrage."""
        if not self._funding_rate_collector:
            return {}
        
        try:
            funding_data = await self._funding_rate_collector.fetch_funding_rate(symbol)
            funding_rate = funding_data.get('funding_rate', 0)
            mark_price = funding_data.get('mark_price', 0)
            
            # Number of funding periods
            periods = duration_hours / 8
            
            # Expected funding payment
            position_value = position_size * mark_price
            expected_funding = position_value * funding_rate * periods
            
            return {
                'funding_rate': funding_rate,
                'annualized_rate': funding_rate * 3 * 365,
                'expected_funding': expected_funding,
                'periods': periods,
                'daily_return_pct': funding_rate * 3 * 100
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating funding PnL: {e}")
            return {}
    
    def get_funding_opportunities(
        self,
        symbols: List[str],
        min_rate: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Scan for funding arbitrage opportunities."""
        opportunities = []
        min_rate = min_rate or self._min_funding_rate
        
        # This would be called by the strategy manager
        # with actual funding rate data
        
        return opportunities
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status including funding positions."""
        status = super().get_status()
        status['funding_positions'] = {
            k: {
                'quantity': v['quantity'],
                'entry_time': v['entry_time'].isoformat(),
                'entry_funding_rate': v['entry_funding_rate']
            }
            for k, v in self._funding_positions.items()
        }
        return status
