"""
Data Cleaner for data validation and cleaning.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from core.base import BaseModule
from core.exceptions import DataValidationException


class DataCleaner(BaseModule):
    """
    Cleans and validates market data.
    Handles missing values, outliers, and data quality issues.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('data_cleaner', config)
        self._max_missing_pct = self.config.get('max_missing_pct', 0.1)
        self._outlier_method = self.config.get('outlier_method', 'iqr')
        self._outlier_threshold = self.config.get('outlier_threshold', 3.0)
    
    async def initialize(self) -> bool:
        """Initialize the data cleaner."""
        self.logger.info("Initializing data cleaner...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the cleaner."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the cleaner."""
        self._running = False
        return True
    
    def clean_ohlcv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Tuple of (cleaned_df, report)
        """
        report = {
            'original_rows': len(df),
            'original_cols': len(df.columns),
            'issues': [],
            'fixes': []
        }
        
        if df.empty:
            report['issues'].append('Empty dataframe')
            return df, report
        
        df_clean = df.copy()
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df_clean.columns]
        if missing_cols:
            raise DataValidationException(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        missing = df_clean.isnull().sum()
        missing_pct = missing / len(df_clean)
        
        for col, pct in missing_pct.items():
            if pct > 0:
                report['issues'].append(f"Column '{col}' has {pct:.2%} missing values")
                
                if pct > self._max_missing_pct:
                    report['fixes'].append(f"Dropped column '{col}' due to high missing rate")
                    df_clean.drop(columns=[col], inplace=True)
                else:
                    # Forward fill for price data
                    if col in ['open', 'high', 'low', 'close']:
                        df_clean[col].fillna(method='ffill', inplace=True)
                        report['fixes'].append(f"Forward filled missing values in '{col}'")
                    elif col == 'volume':
                        df_clean[col].fillna(0, inplace=True)
                        report['fixes'].append(f"Filled missing volume with 0")
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (df_clean['high'] < df_clean['low']) |
            (df_clean['high'] < df_clean['open']) |
            (df_clean['high'] < df_clean['close']) |
            (df_clean['low'] > df_clean['open']) |
            (df_clean['low'] > df_clean['close'])
        )
        
        invalid_count = invalid_ohlc.sum()
        if invalid_count > 0:
            report['issues'].append(f"Found {invalid_count} rows with invalid OHLC relationships")
            # Fix by adjusting high/low
            df_clean.loc[invalid_ohlc, 'high'] = df_clean.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].max(axis=1)
            df_clean.loc[invalid_ohlc, 'low'] = df_clean.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].min(axis=1)
            report['fixes'].append(f"Fixed OHLC relationships in {invalid_count} rows")
        
        # Check for zero/negative prices
        zero_prices = (df_clean[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if zero_prices.sum() > 0:
            report['issues'].append(f"Found {zero_prices.sum()} rows with zero or negative prices")
            df_clean = df_clean[~zero_prices]
            report['fixes'].append(f"Removed {zero_prices.sum()} rows with invalid prices")
        
        # Check for negative volume
        if (df_clean['volume'] < 0).any():
            report['issues'].append("Found negative volume values")
            df_clean.loc[df_clean['volume'] < 0, 'volume'] = 0
            report['fixes'].append("Set negative volumes to 0")
        
        # Remove duplicates
        duplicates = df_clean.index.duplicated()
        if duplicates.any():
            report['issues'].append(f"Found {duplicates.sum()} duplicate timestamps")
            df_clean = df_clean[~duplicates]
            report['fixes'].append("Removed duplicate rows")
        
        report['final_rows'] = len(df_clean)
        report['final_cols'] = len(df_clean.columns)
        report['rows_removed'] = report['original_rows'] - report['final_rows']
        
        return df_clean, report
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detect outliers in the data.
        
        Args:
            df: Input DataFrame
            columns: Columns to check
        
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if self._outlier_method == 'iqr':
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self._outlier_threshold * iqr
                upper = q3 + self._outlier_threshold * iqr
                result[f'{col}_outlier'] = (df[col] < lower) | (df[col] > upper)
            
            elif self._outlier_method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                result[f'{col}_outlier'] = z_scores > self._outlier_threshold
        
        return result
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'remove'
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Remove or cap outliers.
        
        Args:
            df: Input DataFrame
            columns: Columns to process
            method: 'remove', 'cap', or 'winsorize'
        
        Returns:
            Tuple of (processed_df, outlier_counts)
        """
        if columns is None:
            columns = ['close', 'volume']
        
        df_result = df.copy()
        outlier_counts = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self._outlier_threshold * iqr
            upper = q3 + self._outlier_threshold * iqr
            
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            outlier_counts[col] = outlier_mask.sum()
            
            if method == 'remove':
                df_result = df_result[~outlier_mask]
            elif method == 'cap':
                df_result.loc[df_result[col] < lower, col] = lower
                df_result.loc[df_result[col] > upper, col] = upper
            elif method == 'winsorize':
                df_result[col] = df[col].clip(lower=lower, upper=upper)
        
        return df_result, outlier_counts
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate overall data quality.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Quality report
        """
        report = {
            'timestamp': datetime.now(),
            'rows': len(df),
            'columns': len(df.columns),
            'quality_score': 100.0,
            'issues': []
        }
        
        if df.empty:
            report['quality_score'] = 0
            report['issues'].append('Empty dataset')
            return report
        
        # Check missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0:
            report['issues'].append(f"Missing values: {missing_pct:.2%}")
            report['quality_score'] -= missing_pct * 20
        
        # Check for gaps in timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            if hasattr(df, 'attrs') and 'timeframe' in df.attrs:
                expected_freq = df.attrs['timeframe']
            else:
                expected_freq = '1h'
            
            try:
                expected = pd.date_range(
                    start=df.index.min(),
                    end=df.index.max(),
                    freq=expected_freq
                )
                missing_timestamps = len(expected) - len(df)
                if missing_timestamps > 0:
                    gap_pct = missing_timestamps / len(expected)
                    report['issues'].append(f"Timestamp gaps: {missing_timestamps} ({gap_pct:.2%})")
                    report['quality_score'] -= gap_pct * 30
            except Exception:
                pass
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].std() == 0:
                report['issues'].append(f"Column '{col}' has zero variance")
                report['quality_score'] -= 5
        
        report['quality_score'] = max(0, report['quality_score'])
        
        return report
    
    def fill_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str = '1h'
    ) -> pd.DataFrame:
        """
        Fill gaps in time series data.
        
        Args:
            df: Input DataFrame with DatetimeIndex
            timeframe: Expected timeframe
        
        Returns:
            DataFrame with gaps filled
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Create complete timestamp range
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=timeframe
        )
        
        # Reindex and fill
        df_filled = df.reindex(full_range)
        
        # Forward fill for prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df_filled.columns:
                df_filled[col].fillna(method='ffill', inplace=True)
        
        # Fill volume with 0
        if 'volume' in df_filled.columns:
            df_filled['volume'].fillna(0, inplace=True)
        
        return df_filled
