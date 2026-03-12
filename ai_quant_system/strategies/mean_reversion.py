"""
Mean Reversion Strategy.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig, Signal
from core.constants import OrderSide


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands and RSI.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        data_provider=None,
        order_executor=None
    ):
        super().__init__(config, data_provider, order_executor)
        
        # Mean reversion parameters
        self._bb_period = self._config.parameters.get('bb_period', 20)
        self._bb_std = self._config.parameters.get('bb_std', 2.0)
        self._rsi_period = self._config.parameters.get('rsi_period', 14)
        self._rsi_oversold = self._config.parameters.get('rsi_oversold', 30)
        self._rsi_overbought = self._config.parameters.get('rsi_overbought', 70)
        self._lookback = self._config.parameters.get('lookback', 50)
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate mean reversion signals."""
        signals = []
        
        if len(data) < max(self._bb_period, self._rsi_period, self._lookback) + 1:
            return signals
        
        # Calculate indicators
        data = self._calculate_indicators(data)
        
        latest = data.iloc[-1]
        
        for symbol in self._config.symbols:
            # Get symbol-specific data
            if 'symbol' in data.columns:
                symbol_data = data[data['symbol'] == symbol]
                if len(symbol_data) < 2:
                    continue
                latest = symbol_data.iloc[-1]
            
            current_position = self._positions.get(symbol, {}).get('quantity', 0)
            
            # Get indicator values
            bb_upper = latest.get('bb_upper')
            bb_lower = latest.get('bb_lower')
            bb_middle = latest.get('bb_middle')
            rsi = latest.get('rsi')
            close = latest['close']
            
            # Check for oversold condition (buy signal)
            if (close <= bb_lower and rsi <= self._rsi_oversold and current_position <= 0):
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    signal_type='entry',
                    strength=self._calculate_strength(latest, 'oversold'),
                    price=close,
                    stop_loss=close * (1 - self._config.stop_loss_pct),
                    take_profit=bb_middle,
                    metadata={
                        'bb_position': (close - bb_lower) / (bb_upper - bb_lower),
                        'rsi': rsi,
                        'signal_type': 'oversold_reversion'
                    }
                )
                signals.append(signal)
            
            # Check for overbought condition (sell signal)
            elif (close >= bb_upper and rsi >= self._rsi_overbought and current_position >= 0):
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    signal_type='entry',
                    strength=self._calculate_strength(latest, 'overbought'),
                    price=close,
                    stop_loss=close * (1 + self._config.stop_loss_pct),
                    take_profit=bb_middle,
                    metadata={
                        'bb_position': (close - bb_lower) / (bb_upper - bb_lower),
                        'rsi': rsi,
                        'signal_type': 'overbought_reversion'
                    }
                )
                signals.append(signal)
            
            # Exit signals for existing positions
            elif current_position > 0:
                # Exit long if price returns to mean or becomes overbought
                if close >= bb_middle or rsi >= self._rsi_overbought:
                    signals.append(Signal(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        signal_type='exit',
                        strength=1.0,
                        price=close,
                        metadata={'reason': 'mean_reversion_complete'}
                    ))
            
            elif current_position < 0:
                # Exit short if price returns to mean or becomes oversold
                if close <= bb_middle or rsi <= self._rsi_oversold:
                    signals.append(Signal(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        signal_type='exit',
                        strength=1.0,
                        price=close,
                        metadata={'reason': 'mean_reversion_complete'}
                    ))
        
        return signals
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators."""
        df = data.copy()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self._bb_period).mean()
        bb_std = df['close'].rolling(window=self._bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self._bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self._bb_std)
        
        # BB Width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # BB Position (0-1 where price is relative to bands)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self._rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self._rsi_period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Z-Score
        df['zscore'] = (df['close'] - df['close'].rolling(window=self._lookback).mean()) / \
                       df['close'].rolling(window=self._lookback).std()
        
        # Mean
        df['sma'] = df['close'].rolling(window=self._lookback).mean()
        
        return df
    
    def _calculate_strength(self, row: pd.Series, condition: str) -> float:
        """Calculate signal strength based on deviation from mean."""
        bb_position = row.get('bb_position', 0.5)
        rsi = row.get('rsi', 50)
        
        if condition == 'oversold':
            # Stronger signal when more oversold
            rsi_strength = (self._rsi_oversold - rsi) / self._rsi_oversold
            bb_strength = 1 - bb_position
        else:  # overbought
            rsi_strength = (rsi - self._rsi_overbought) / (100 - self._rsi_overbought)
            bb_strength = bb_position
        
        strength = (rsi_strength + bb_strength) / 2
        return min(max(strength, 0), 1)
    
    def get_indicator_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get current indicator values."""
        data = self._calculate_indicators(data)
        latest = data.iloc[-1]
        
        return {
            'bb_upper': latest.get('bb_upper'),
            'bb_middle': latest.get('bb_middle'),
            'bb_lower': latest.get('bb_lower'),
            'bb_width': latest.get('bb_width'),
            'bb_position': latest.get('bb_position'),
            'rsi': latest.get('rsi'),
            'zscore': latest.get('zscore'),
            'condition': self._get_condition(latest)
        }
    
    def _get_condition(self, row: pd.Series) -> str:
        """Determine current market condition."""
        rsi = row.get('rsi', 50)
        bb_position = row.get('bb_position', 0.5)
        
        if rsi <= self._rsi_oversold or bb_position <= 0.1:
            return 'oversold'
        elif rsi >= self._rsi_overbought or bb_position >= 0.9:
            return 'overbought'
        else:
            return 'neutral'
