"""
Trend Following Strategy.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig, Signal
from core.constants import OrderSide


class TrendStrategy(BaseStrategy):
    """
    Trend Following Strategy using moving averages and trend indicators.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        data_provider=None,
        order_executor=None
    ):
        super().__init__(config, data_provider, order_executor)
        
        # Trend parameters
        self._fast_period = self._config.parameters.get('fast_period', 10)
        self._slow_period = self._config.parameters.get('slow_period', 30)
        self._trend_strength_threshold = self._config.parameters.get('trend_strength', 0.02)
        self._atr_period = self._config.parameters.get('atr_period', 14)
        self._atr_multiplier = self._config.parameters.get('atr_multiplier', 2.0)
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trend following signals."""
        signals = []
        
        if len(data) < self._slow_period + 1:
            return signals
        
        # Calculate indicators
        data = self._calculate_indicators(data)
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        for symbol in self._config.symbols:
            # Get symbol-specific data if multi-symbol
            if 'symbol' in data.columns:
                symbol_data = data[data['symbol'] == symbol]
                if len(symbol_data) < 2:
                    continue
                latest = symbol_data.iloc[-1]
                prev = symbol_data.iloc[-2]
            
            # Trend direction
            trend_up = latest[f'sma_{self._fast_period}'] > latest[f'sma_{self._slow_period}']
            prev_trend_up = prev[f'sma_{self._fast_period}'] > prev[f'sma_{self._slow_period}']
            
            # Check for crossover
            crossover_up = trend_up and not prev_trend_up
            crossover_down = not trend_up and prev_trend_up
            
            # Check position
            current_position = self._positions.get(symbol, {}).get('quantity', 0)
            
            if crossover_up and current_position <= 0:
                # Buy signal
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    signal_type='entry',
                    strength=self._calculate_signal_strength(latest),
                    price=latest['close'],
                    stop_loss=self._calculate_trailing_stop(latest, 'long'),
                    take_profit=None,  # Let profits run
                    metadata={
                        'fast_sma': latest[f'sma_{self._fast_period}'],
                        'slow_sma': latest[f'sma_{self._slow_period}'],
                        'trend': 'up'
                    }
                )
                signals.append(signal)
            
            elif crossover_down and current_position >= 0:
                # Sell signal
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    signal_type='entry',
                    strength=self._calculate_signal_strength(latest),
                    price=latest['close'],
                    stop_loss=self._calculate_trailing_stop(latest, 'short'),
                    take_profit=None,
                    metadata={
                        'fast_sma': latest[f'sma_{self._fast_period}'],
                        'slow_sma': latest[f'sma_{self._slow_period}'],
                        'trend': 'down'
                    }
                )
                signals.append(signal)
            
            # Check for exit signals on existing positions
            elif current_position > 0 and not trend_up:
                # Exit long position
                signals.append(Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    signal_type='exit',
                    strength=1.0,
                    price=latest['close'],
                    metadata={'reason': 'trend_reversal'}
                ))
            
            elif current_position < 0 and trend_up:
                # Exit short position
                signals.append(Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    signal_type='exit',
                    strength=1.0,
                    price=latest['close'],
                    metadata={'reason': 'trend_reversal'}
                ))
        
        return signals
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators."""
        df = data.copy()
        
        # Moving averages
        df[f'sma_{self._fast_period}'] = df['close'].rolling(window=self._fast_period).mean()
        df[f'sma_{self._slow_period}'] = df['close'].rolling(window=self._slow_period).mean()
        df[f'ema_{self._fast_period}'] = df['close'].ewm(span=self._fast_period, adjust=False).mean()
        df[f'ema_{self._slow_period}'] = df['close'].ewm(span=self._slow_period, adjust=False).mean()
        
        # ATR for trailing stops
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self._atr_period).mean()
        
        # ADX for trend strength
        df['adx'] = self._calculate_adx(df)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX indicator."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_signal_strength(self, row: pd.Series) -> float:
        """Calculate signal strength based on trend strength."""
        adx = row.get('adx', 25)
        
        # Normalize ADX (typically 0-100, strong trend > 25)
        strength = min(adx / 50, 1.0)
        
        return strength
    
    def _calculate_trailing_stop(self, row: pd.Series, direction: str) -> float:
        """Calculate trailing stop price."""
        atr = row.get('atr', row['close'] * 0.02)
        
        if direction == 'long':
            return row['close'] - atr * self._atr_multiplier
        else:
            return row['close'] + atr * self._atr_multiplier
    
    def get_indicator_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get current indicator values."""
        data = self._calculate_indicators(data)
        latest = data.iloc[-1]
        
        return {
            'fast_sma': latest.get(f'sma_{self._fast_period}'),
            'slow_sma': latest.get(f'sma_{self._slow_period}'),
            'fast_ema': latest.get(f'ema_{self._fast_period}'),
            'slow_ema': latest.get(f'ema_{self._slow_period}'),
            'atr': latest.get('atr'),
            'adx': latest.get('adx'),
            'trend': 'up' if latest.get(f'sma_{self._fast_period}', 0) > latest.get(f'sma_{self._slow_period}', 0) else 'down'
        }
