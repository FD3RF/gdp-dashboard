"""
Momentum Strategy.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig, Signal
from core.constants import OrderSide


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy using rate of change and momentum indicators.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        data_provider=None,
        order_executor=None
    ):
        super().__init__(config, data_provider, order_executor)
        
        # Momentum parameters
        self._momentum_period = self._config.parameters.get('momentum_period', 14)
        self._roc_period = self._config.parameters.get('roc_period', 10)
        self._volume_threshold = self._config.parameters.get('volume_threshold', 1.5)
        self._momentum_threshold = self._config.parameters.get('momentum_threshold', 0)
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate momentum signals."""
        signals = []
        
        if len(data) < max(self._momentum_period, self._roc_period) + 20:
            return signals
        
        # Calculate indicators
        data = self._calculate_indicators(data)
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        for symbol in self._config.symbols:
            if 'symbol' in data.columns:
                symbol_data = data[data['symbol'] == symbol]
                if len(symbol_data) < 2:
                    continue
                latest = symbol_data.iloc[-1]
                prev = symbol_data.iloc[-2]
            
            current_position = self._positions.get(symbol, {}).get('quantity', 0)
            
            momentum = latest.get('momentum', 0)
            roc = latest.get('roc', 0)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Strong positive momentum with volume confirmation
            if (momentum > self._momentum_threshold and 
                roc > 0 and 
                volume_ratio > self._volume_threshold and
                current_position <= 0):
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    signal_type='entry',
                    strength=self._calculate_strength(latest),
                    price=latest['close'],
                    stop_loss=latest['close'] * (1 - self._config.stop_loss_pct),
                    take_profit=latest['close'] * (1 + self._config.take_profit_pct),
                    metadata={
                        'momentum': momentum,
                        'roc': roc,
                        'volume_ratio': volume_ratio,
                        'signal': 'bullish_momentum'
                    }
                )
                signals.append(signal)
            
            # Strong negative momentum
            elif (momentum < -self._momentum_threshold and 
                  roc < 0 and 
                  volume_ratio > self._volume_threshold and
                  current_position >= 0):
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    signal_type='entry',
                    strength=self._calculate_strength(latest),
                    price=latest['close'],
                    stop_loss=latest['close'] * (1 + self._config.stop_loss_pct),
                    take_profit=latest['close'] * (1 - self._config.take_profit_pct),
                    metadata={
                        'momentum': momentum,
                        'roc': roc,
                        'volume_ratio': volume_ratio,
                        'signal': 'bearish_momentum'
                    }
                )
                signals.append(signal)
            
            # Exit signals
            if current_position > 0:
                # Exit if momentum turns negative
                if momentum < 0 or roc < -2:
                    signals.append(Signal(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        signal_type='exit',
                        strength=1.0,
                        price=latest['close'],
                        metadata={'reason': 'momentum_loss'}
                    ))
            
            elif current_position < 0:
                # Exit if momentum turns positive
                if momentum > 0 or roc > 2:
                    signals.append(Signal(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        signal_type='exit',
                        strength=1.0,
                        price=latest['close'],
                        metadata={'reason': 'momentum_loss'}
                    ))
        
        return signals
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        df = data.copy()
        
        # Momentum (current price - price N periods ago)
        df['momentum'] = df['close'] - df['close'].shift(self._momentum_period)
        
        # Rate of Change
        df['roc'] = ((df['close'] - df['close'].shift(self._roc_period)) / 
                     df['close'].shift(self._roc_period)) * 100
        
        # Volume ratio
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Relative Strength (vs previous periods)
        df['price_change'] = df['close'].diff()
        df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: -x if x < 0 else 0)
        
        df['avg_gain'] = df['gain'].rolling(window=self._momentum_period).mean()
        df['avg_loss'] = df['loss'].rolling(window=self._momentum_period).mean()
        
        df['rs'] = df['avg_gain'] / (df['avg_loss'] + 0.0001)
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Momentum Score (composite)
        df['momentum_score'] = (
            (df['momentum'] / df['close']) * 0.3 +
            (df['roc'] / 10) * 0.3 +
            ((df['rsi'] - 50) / 50) * 0.2 +
            (df['macd_hist'] / df['close'] * 100) * 0.2
        )
        
        return df
    
    def _calculate_strength(self, row: pd.Series) -> float:
        """Calculate signal strength."""
        momentum_score = abs(row.get('momentum_score', 0))
        volume_ratio = row.get('volume_ratio', 1)
        
        # Normalize strength
        strength = min((momentum_score * 2 + (volume_ratio - 1)) / 2, 1.0)
        
        return max(strength, 0.1)
    
    def get_indicator_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get current indicator values."""
        data = self._calculate_indicators(data)
        latest = data.iloc[-1]
        
        return {
            'momentum': latest.get('momentum'),
            'roc': latest.get('roc'),
            'volume_ratio': latest.get('volume_ratio'),
            'rsi': latest.get('rsi'),
            'macd': latest.get('macd'),
            'macd_signal': latest.get('macd_signal'),
            'macd_hist': latest.get('macd_hist'),
            'momentum_score': latest.get('momentum_score'),
            'trend': 'bullish' if latest.get('momentum', 0) > 0 else 'bearish'
        }
