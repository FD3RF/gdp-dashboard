"""
Statistical Arbitrage Strategy.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig, Signal
from core.constants import OrderSide


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage Strategy using cointegration and pairs trading.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        data_provider=None,
        order_executor=None
    ):
        super().__init__(config, data_provider, order_executor)
        
        # Stat arb parameters
        self._lookback = self._config.parameters.get('lookback', 100)
        self._z_threshold = self._config.parameters.get('z_threshold', 2.0)
        self._exit_z = self._config.parameters.get('exit_z', 0.5)
        self._hedge_ratio = self._config.parameters.get('hedge_ratio', {})
        self._pairs = self._config.parameters.get('pairs', [])
        
        # Store spread history
        self._spread_history: Dict[str, List[float]] = {}
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate statistical arbitrage signals."""
        signals = []
        
        if len(data) < self._lookback:
            return signals
        
        # Process each pair
        for pair in self._pairs:
            if len(pair) != 2:
                continue
            
            symbol_a, symbol_b = pair
            pair_key = f"{symbol_a}_{symbol_b}"
            
            # Get prices for both symbols
            if 'symbol' in data.columns:
                price_a = data[data['symbol'] == symbol_a]['close']
                price_b = data[data['symbol'] == symbol_b]['close']
            else:
                # Assume multi-column format
                price_a = data.get(f'{symbol_a}_close', data['close'])
                price_b = data.get(f'{symbol_b}_close', data['close'])
            
            if len(price_a) < self._lookback or len(price_b) < self._lookback:
                continue
            
            # Calculate spread
            spread = self._calculate_spread(price_a, price_b, pair_key)
            
            if spread is None:
                continue
            
            # Calculate z-score
            zscore = self._calculate_zscore(spread)
            
            current_z = zscore.iloc[-1]
            prev_z = zscore.iloc[-2]
            
            # Get current positions
            pos_a = self._positions.get(symbol_a, {}).get('quantity', 0)
            pos_b = self._positions.get(symbol_b, {}).get('quantity', 0)
            
            # Trading logic
            if current_z > self._z_threshold and pos_a == 0 and pos_b == 0:
                # Spread is too wide - short A, long B
                hedge_ratio = self._hedge_ratio.get(pair_key, 1.0)
                
                signals.append(Signal(
                    symbol=symbol_a,
                    side=OrderSide.SELL,
                    signal_type='entry',
                    strength=min(abs(current_z) / 4, 1.0),
                    metadata={'pair': pair_key, 'zscore': current_z, 'direction': 'short_spread'}
                ))
                
                signals.append(Signal(
                    symbol=symbol_b,
                    side=OrderSide.BUY,
                    signal_type='entry',
                    strength=min(abs(current_z) / 4, 1.0),
                    quantity_ratio=hedge_ratio,
                    metadata={'pair': pair_key, 'zscore': current_z, 'direction': 'short_spread'}
                ))
            
            elif current_z < -self._z_threshold and pos_a == 0 and pos_b == 0:
                # Spread is too narrow - long A, short B
                hedge_ratio = self._hedge_ratio.get(pair_key, 1.0)
                
                signals.append(Signal(
                    symbol=symbol_a,
                    side=OrderSide.BUY,
                    signal_type='entry',
                    strength=min(abs(current_z) / 4, 1.0),
                    metadata={'pair': pair_key, 'zscore': current_z, 'direction': 'long_spread'}
                ))
                
                signals.append(Signal(
                    symbol=symbol_b,
                    side=OrderSide.SELL,
                    signal_type='entry',
                    strength=min(abs(current_z) / 4, 1.0),
                    quantity_ratio=hedge_ratio,
                    metadata={'pair': pair_key, 'zscore': current_z, 'direction': 'long_spread'}
                ))
            
            # Exit signals
            elif abs(current_z) < self._exit_z:
                if pos_a != 0 or pos_b != 0:
                    # Close positions
                    if pos_a > 0:
                        signals.append(Signal(
                            symbol=symbol_a,
                            side=OrderSide.SELL,
                            signal_type='exit',
                            strength=1.0,
                            metadata={'reason': 'spread_normalized'}
                        ))
                    elif pos_a < 0:
                        signals.append(Signal(
                            symbol=symbol_a,
                            side=OrderSide.BUY,
                            signal_type='exit',
                            strength=1.0,
                            metadata={'reason': 'spread_normalized'}
                        ))
                    
                    if pos_b > 0:
                        signals.append(Signal(
                            symbol=symbol_b,
                            side=OrderSide.SELL,
                            signal_type='exit',
                            strength=1.0,
                            metadata={'reason': 'spread_normalized'}
                        ))
                    elif pos_b < 0:
                        signals.append(Signal(
                            symbol=symbol_b,
                            side=OrderSide.BUY,
                            signal_type='exit',
                            strength=1.0,
                            metadata={'reason': 'spread_normalized'}
                        ))
        
        return signals
    
    def _calculate_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        pair_key: str
    ) -> Optional[pd.Series]:
        """Calculate the spread between two price series."""
        try:
            # Use OLS to find hedge ratio if not specified
            if pair_key not in self._hedge_ratio:
                # Simple OLS regression
                X = np.vstack([price_b.values, np.ones(len(price_b))]).T
                y = price_a.values
                
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    self._hedge_ratio[pair_key] = beta[0]
                except:
                    self._hedge_ratio[pair_key] = 1.0
            
            hedge_ratio = self._hedge_ratio[pair_key]
            
            # Calculate spread
            spread = price_a - hedge_ratio * price_b
            
            # Store spread history
            if pair_key not in self._spread_history:
                self._spread_history[pair_key] = []
            self._spread_history[pair_key].extend(spread.values[-10:].tolist())
            
            return spread
            
        except Exception as e:
            self.logger.error(f"Error calculating spread: {e}")
            return None
    
    def _calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate rolling z-score of the spread."""
        rolling_mean = spread.rolling(window=self._lookback).mean()
        rolling_std = spread.rolling(window=self._lookback).std()
        
        zscore = (spread - rolling_mean) / (rolling_std + 0.0001)
        
        return zscore
    
    def test_cointegration(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Dict[str, Any]:
        """Test for cointegration between two price series."""
        try:
            from statsmodels.tsa.stattools import coint, adfuller
            
            # Engle-Granger cointegration test
            score, pvalue, _ = coint(price_a, price_b)
            
            # ADF test on spread
            spread = price_a - price_b
            adf_result = adfuller(spread.dropna())
            
            return {
                'cointegrated': pvalue < 0.05,
                'pvalue': pvalue,
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'hedge_ratio': self._hedge_ratio.get(
                    f"{price_a.name}_{price_b.name}", 1.0
                )
            }
        except ImportError:
            return {'cointegrated': False, 'error': 'statsmodels not available'}
        except Exception as e:
            return {'cointegrated': False, 'error': str(e)}
    
    def get_indicator_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get current indicator values for all pairs."""
        values = {}
        
        for pair in self._pairs:
            if len(pair) != 2:
                continue
            
            symbol_a, symbol_b = pair
            pair_key = f"{symbol_a}_{symbol_b}"
            
            if pair_key in self._spread_history:
                spread_series = pd.Series(self._spread_history[pair_key])
                zscore = self._calculate_zscore(spread_series)
                
                values[pair_key] = {
                    'spread': spread_series.iloc[-1] if len(spread_series) > 0 else None,
                    'zscore': zscore.iloc[-1] if len(zscore) > 0 else None,
                    'hedge_ratio': self._hedge_ratio.get(pair_key)
                }
        
        return values
