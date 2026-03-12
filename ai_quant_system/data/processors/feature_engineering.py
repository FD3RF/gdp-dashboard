"""
Feature Engineering for generating trading features.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from core.base import BaseModule


class FeatureEngineering(BaseModule):
    """
    Generates technical indicators and features for trading strategies.
    Supports trend, momentum, volatility, and volume indicators.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('feature_engineering', config)
        self._default_periods = self.config.get('default_periods', {
            'sma_fast': 10,
            'sma_slow': 20,
            'ema_fast': 12,
            'ema_slow': 26,
            'rsi': 14,
            'macd_signal': 9,
            'bollinger': 20,
            'atr': 14
        })
    
    async def initialize(self) -> bool:
        """Initialize feature engineering."""
        self.logger.info("Initializing feature engineering...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start feature engineering."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop feature engineering."""
        self._running = False
        return True
    
    def generate_all_features(
        self,
        df: pd.DataFrame,
        include_volume: bool = True,
        include_volatility: bool = True
    ) -> pd.DataFrame:
        """
        Generate all common trading features.
        
        Args:
            df: OHLCV DataFrame
            include_volume: Include volume features
            include_volatility: Include volatility features
        
        Returns:
            DataFrame with all features
        """
        df_features = df.copy()
        
        # Trend indicators
        df_features = self.add_sma(df_features)
        df_features = self.add_ema(df_features)
        
        # Momentum indicators
        df_features = self.add_rsi(df_features)
        df_features = self.add_macd(df_features)
        df_features = self.add_stochastic(df_features)
        
        # Volatility indicators
        if include_volatility:
            df_features = self.add_bollinger_bands(df_features)
            df_features = self.add_atr(df_features)
        
        # Volume indicators
        if include_volume and 'volume' in df.columns:
            df_features = self.add_volume_features(df_features)
        
        # Price features
        df_features = self.add_price_features(df_features)
        
        return df_features
    
    def add_sma(
        self,
        df: pd.DataFrame,
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        df_result = df.copy()
        
        if periods is None:
            periods = [10, 20, 50, 100, 200]
        
        for period in periods:
            df_result[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Add crossover signals
        if 10 in periods and 20 in periods:
            df_result['sma_cross'] = np.where(
                df_result['sma_10'] > df_result['sma_20'], 1, -1
            )
        
        return df_result
    
    def add_ema(
        self,
        df: pd.DataFrame,
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        df_result = df.copy()
        
        if periods is None:
            periods = [12, 26, 50, 100]
        
        for period in periods:
            df_result[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        return df_result
    
    def add_rsi(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """Add Relative Strength Index."""
        df_result = df.copy()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df_result['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        df_result['rsi_overbought'] = np.where(df_result['rsi'] > 70, 1, 0)
        df_result['rsi_oversold'] = np.where(df_result['rsi'] < 30, 1, 0)
        
        return df_result
    
    def add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Add MACD indicator."""
        df_result = df.copy()
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df_result['macd'] = ema_fast - ema_slow
        df_result['macd_signal'] = df_result['macd'].ewm(span=signal, adjust=False).mean()
        df_result['macd_histogram'] = df_result['macd'] - df_result['macd_signal']
        
        # MACD crossover signals
        df_result['macd_cross'] = np.where(
            df_result['macd'] > df_result['macd_signal'], 1, -1
        )
        
        return df_result
    
    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df_result = df.copy()
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df_result['bb_upper'] = sma + (std * std_dev)
        df_result['bb_middle'] = sma
        df_result['bb_lower'] = sma - (std * std_dev)
        df_result['bb_width'] = (df_result['bb_upper'] - df_result['bb_lower']) / sma
        
        # Bollinger position (0-1, where price is relative to bands)
        df_result['bb_position'] = (
            df['close'] - df_result['bb_lower']
        ) / (df_result['bb_upper'] - df_result['bb_lower'])
        
        return df_result
    
    def add_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """Add Average True Range."""
        df_result = df.copy()
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_result['atr'] = tr.rolling(window=period).mean()
        
        # ATR as percentage of price
        df_result['atr_pct'] = df_result['atr'] / df['close'] * 100
        
        return df_result
    
    def add_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        df_result = df.copy()
        
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df_result['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df_result['stoch_d'] = df_result['stoch_k'].rolling(window=d_period).mean()
        
        return df_result
    
    def add_volume_features(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> pd.DataFrame:
        """Add volume-based features."""
        df_result = df.copy()
        
        # Volume SMA
        df_result['volume_sma'] = df['volume'].rolling(window=period).mean()
        
        # Volume ratio
        df_result['volume_ratio'] = df['volume'] / df_result['volume_sma']
        
        # OBV (On-Balance Volume)
        df_result['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume Profile (simplified)
        df_result['volume_price_trend'] = df['volume'] * df['close'].pct_change()
        
        # VWAP
        df_result['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df_result
    
    def add_price_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add price-based features."""
        df_result = df.copy()
        
        # Returns
        df_result['returns'] = df['close'].pct_change()
        df_result['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price position in range
        df_result['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Candle features
        df_result['body_size'] = abs(df['close'] - df['open']) / df['close']
        df_result['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df_result['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Candle direction
        df_result['is_bullish'] = np.where(df['close'] > df['open'], 1, 0)
        
        # Momentum
        df_result['momentum'] = df['close'] - df['close'].shift(10)
        df_result['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        
        return df_result
    
    def add_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """Add support and resistance levels."""
        df_result = df.copy()
        
        # Rolling high/low
        df_result['resistance'] = df['high'].rolling(window=window).max()
        df_result['support'] = df['low'].rolling(window=window).min()
        
        # Distance from support/resistance
        df_result['dist_resistance'] = (df_result['resistance'] - df['close']) / df['close']
        df_result['dist_support'] = (df['close'] - df_result['support']) / df['close']
        
        return df_result
    
    def add_lagged_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lags: List[int] = None
    ) -> pd.DataFrame:
        """Add lagged features."""
        df_result = df.copy()
        
        if columns is None:
            columns = ['close', 'volume', 'returns']
        
        if lags is None:
            lags = [1, 2, 3, 5, 10]
        
        for col in columns:
            if col not in df.columns:
                continue
            for lag in lags:
                df_result[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df_result
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """Add rolling statistics features."""
        df_result = df.copy()
        
        if windows is None:
            windows = [5, 10, 20]
        
        for window in windows:
            df_result[f'returns_mean_{window}'] = df['close'].pct_change().rolling(window=window).mean()
            df_result[f'returns_std_{window}'] = df['close'].pct_change().rolling(window=window).std()
            df_result[f'returns_skew_{window}'] = df['close'].pct_change().rolling(window=window).skew()
        
        return df_result
    
    def detect_patterns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Detect common candlestick patterns."""
        df_result = df.copy()
        
        # Doji
        body = abs(df['close'] - df['open'])
        total = df['high'] - df['low']
        df_result['doji'] = np.where(body / (total + 1e-10) < 0.1, 1, 0)
        
        # Hammer (bullish)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        df_result['hammer'] = np.where(
            (lower_wick > 2 * body) & (upper_wick < body), 1, 0
        )
        
        # Engulfing
        prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
        df_result['bullish_engulfing'] = np.where(
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (body > prev_body), 1, 0
        )
        
        return df_result
