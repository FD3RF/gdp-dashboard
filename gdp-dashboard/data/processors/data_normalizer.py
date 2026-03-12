"""
Data Normalizer for data normalization and standardization.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    PowerTransformer
)
from core.base import BaseModule


class DataNormalizer(BaseModule):
    """
    Normalizes and standardizes market data.
    Supports various normalization methods for ML-ready data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('data_normalizer', config)
        self._scalers: Dict[str, Any] = {}
        self._default_method = self.config.get('default_method', 'minmax')
        self._feature_ranges = self.config.get('feature_ranges', (-1, 1))
    
    async def initialize(self) -> bool:
        """Initialize the data normalizer."""
        self.logger.info("Initializing data normalizer...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the normalizer."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the normalizer."""
        self._running = False
        return True
    
    def normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize DataFrame columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: Normalization method ('minmax', 'zscore', 'robust', 'log')
            fit: Whether to fit the scaler
        
        Returns:
            Tuple of (normalized_df, scaler_info)
        """
        method = method or self._default_method
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_norm = df.copy()
        scaler_info = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            scaler_key = f"{col}_{method}"
            
            if method == 'minmax':
                if fit or scaler_key not in self._scalers:
                    scaler = MinMaxScaler(feature_range=self._feature_ranges)
                    df_norm[col] = scaler.fit_transform(df[[col]])
                    self._scalers[scaler_key] = scaler
                else:
                    scaler = self._scalers[scaler_key]
                    df_norm[col] = scaler.transform(df[[col]])
                
                scaler_info[col] = {
                    'method': 'minmax',
                    'min': scaler.data_min_[0],
                    'max': scaler.data_max_[0]
                }
            
            elif method == 'zscore':
                if fit or scaler_key not in self._scalers:
                    scaler = StandardScaler()
                    df_norm[col] = scaler.fit_transform(df[[col]])
                    self._scalers[scaler_key] = scaler
                else:
                    scaler = self._scalers[scaler_key]
                    df_norm[col] = scaler.transform(df[[col]])
                
                scaler_info[col] = {
                    'method': 'zscore',
                    'mean': scaler.mean_[0],
                    'std': scaler.scale_[0]
                }
            
            elif method == 'robust':
                if fit or scaler_key not in self._scalers:
                    scaler = RobustScaler()
                    df_norm[col] = scaler.fit_transform(df[[col]])
                    self._scalers[scaler_key] = scaler
                else:
                    scaler = self._scalers[scaler_key]
                    df_norm[col] = scaler.transform(df[[col]])
                
                scaler_info[col] = {
                    'method': 'robust',
                    'center': scaler.center_[0],
                    'scale': scaler.scale_[0]
                }
            
            elif method == 'log':
                # Log transformation
                min_val = df[col].min()
                if min_val <= 0:
                    df_norm[col] = np.log1p(df[col] - min_val + 1)
                else:
                    df_norm[col] = np.log(df[col])
                
                scaler_info[col] = {'method': 'log', 'min_original': min_val}
            
            elif method == 'percentile':
                # Percentile rank normalization
                df_norm[col] = df[col].rank(pct=True) * 2 - 1
                scaler_info[col] = {'method': 'percentile'}
        
        return df_norm, scaler_info
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        scaler_info: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Inverse transform normalized data.
        
        Args:
            df: Normalized DataFrame
            scaler_info: Scaler information from normalize()
        
        Returns:
            Original scale DataFrame
        """
        df_orig = df.copy()
        
        for col, info in scaler_info.items():
            if col not in df.columns:
                continue
            
            method = info['method']
            scaler_key = f"{col}_{method}"
            
            if method == 'minmax' and scaler_key in self._scalers:
                scaler = self._scalers[scaler_key]
                df_orig[col] = scaler.inverse_transform(df[[col]])
            
            elif method == 'zscore' and scaler_key in self._scalers:
                scaler = self._scalers[scaler_key]
                df_orig[col] = scaler.inverse_transform(df[[col]])
            
            elif method == 'log':
                min_val = info.get('min_original', 0)
                if min_val <= 0:
                    df_orig[col] = np.expm1(df[col]) + min_val - 1
                else:
                    df_orig[col] = np.exp(df[col])
        
        return df_orig
    
    def normalize_by_price(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Normalize all columns by price.
        Useful for comparing assets with different prices.
        
        Args:
            df: Input DataFrame
            price_col: Column to use for normalization
        
        Returns:
            Normalized DataFrame
        """
        df_norm = df.copy()
        
        price = df[price_col]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != price_col:
                df_norm[col] = df[col] / price
        
        return df_norm
    
    def normalize_by_volume(
        self,
        df: pd.DataFrame,
        volume_col: str = 'volume'
    ) -> pd.DataFrame:
        """
        Normalize columns by volume.
        
        Args:
            df: Input DataFrame
            volume_col: Volume column name
        
        Returns:
            Normalized DataFrame
        """
        df_norm = df.copy()
        
        volume = df[volume_col]
        if volume.mean() > 0:
            df_norm[volume_col] = volume / volume.mean()
        
        return df_norm
    
    def normalize_returns(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        method: str = 'log'
    ) -> pd.DataFrame:
        """
        Calculate and normalize returns.
        
        Args:
            df: Input DataFrame
            price_col: Price column
            method: 'log' or 'simple'
        
        Returns:
            DataFrame with returns column
        """
        df_result = df.copy()
        
        if method == 'log':
            df_result['returns'] = np.log(df[price_col] / df[price_col].shift(1))
        else:
            df_result['returns'] = df[price_col].pct_change()
        
        return df_result
    
    def normalize_cross_sectional(
        self,
        df: pd.DataFrame,
        group_col: str = 'symbol'
    ) -> pd.DataFrame:
        """
        Cross-sectional normalization (z-score within each group).
        
        Args:
            df: Input DataFrame
            group_col: Column to group by
        
        Returns:
            Cross-sectionally normalized DataFrame
        """
        df_norm = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == group_col:
                continue
            
            df_norm[col] = df.groupby(group_col)[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )
        
        return df_norm
    
    def normalize_rolling(
        self,
        df: pd.DataFrame,
        window: int = 20,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Rolling window normalization.
        
        Args:
            df: Input DataFrame
            window: Rolling window size
            method: Normalization method
        
        Returns:
            Rolling normalized DataFrame
        """
        df_norm = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            
            if method == 'zscore':
                df_norm[col] = (df[col] - rolling_mean) / (rolling_std + 1e-10)
            elif method == 'minmax':
                rolling_min = df[col].rolling(window=window).min()
                rolling_max = df[col].rolling(window=window).max()
                df_norm[col] = (df[col] - rolling_min) / (rolling_max - rolling_min + 1e-10)
        
        return df_norm
    
    def get_scaler(self, name: str):
        """Get a fitted scaler by name."""
        return self._scalers.get(name)
    
    def clear_scalers(self) -> None:
        """Clear all stored scalers."""
        self._scalers.clear()
    
    def save_scalers(self, path: str) -> bool:
        """Save scalers to file."""
        import pickle
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(self._scalers, f)
            return True
        except Exception as e:
            self.logger.error(f"Error saving scalers: {e}")
            return False
    
    def load_scalers(self, path: str) -> bool:
        """Load scalers from file."""
        import pickle
        
        try:
            with open(path, 'rb') as f:
                self._scalers = pickle.load(f)
            return True
        except Exception as e:
            self.logger.error(f"Error loading scalers: {e}")
            return False
