"""
Utility functions for the AI Quant Trading System.
"""

import asyncio
import functools
import logging
import re
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, Coroutine
)
import pandas as pd
import numpy as np


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with optional file handler.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        format_string: Optional custom format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying async functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay after each attempt
        exceptions: Tuple of exceptions to catch
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    """
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        logger = logging.getLogger(func.__module__)
        logger.debug(
            f"{func.__name__} executed in {end_time - start_time:.4f} seconds"
        )
        return result
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        
        logger = logging.getLogger(func.__module__)
        logger.debug(
            f"{func.__name__} executed in {end_time - start_time:.4f} seconds"
        )
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
    
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Standard format: BASE/QUOTE
    pattern = r'^[A-Z]{2,10}/[A-Z]{2,10}$'
    return bool(re.match(pattern, symbol))


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
    fees: float = 0.0
) -> Dict[str, float]:
    """
    Calculate PnL for a trade.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Position quantity
        side: 'long' or 'short'
        fees: Total fees paid
    
    Returns:
        Dictionary with PnL metrics
    """
    if side.lower() in ('long', 'buy'):
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity
    
    net_pnl = gross_pnl - fees
    pnl_pct = (net_pnl / (entry_price * quantity)) * 100
    
    return {
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'pnl_pct': pnl_pct,
        'fees': fees,
        'return_on_investment': pnl_pct / 100
    }


def round_to_precision(
    value: float,
    precision: int,
    rounding_mode: str = 'down'
) -> float:
    """
    Round a value to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        rounding_mode: 'down', 'up', or 'half_even'
    
    Returns:
        Rounded value
    """
    if precision < 0:
        return value
    
    decimal_value = Decimal(str(value))
    
    rounding_modes = {
        'down': ROUND_DOWN,
        'up': 'ROUND_UP',
        'half_even': 'ROUND_HALF_EVEN'
    }
    
    mode = rounding_modes.get(rounding_mode, ROUND_DOWN)
    rounded = decimal_value.quantize(
        Decimal(10) ** -precision,
        rounding=mode
    )
    
    return float(rounded)


def calculate_position_size(
    account_balance: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: float = 1.0
) -> Dict[str, float]:
    """
    Calculate appropriate position size based on risk parameters.
    
    Args:
        account_balance: Current account balance
        risk_per_trade_pct: Risk per trade as percentage
        entry_price: Entry price
        stop_loss_price: Stop loss price
        leverage: Position leverage
    
    Returns:
        Dictionary with position sizing information
    """
    risk_amount = account_balance * (risk_per_trade_pct / 100)
    price_diff = abs(entry_price - stop_loss_price)
    
    if price_diff == 0:
        return {
            'position_size': 0,
            'position_value': 0,
            'risk_amount': risk_amount,
            'error': 'Invalid stop loss price'
        }
    
    # Position size in base currency
    position_size = (risk_amount / price_diff) * leverage
    position_value = position_size * entry_price
    
    return {
        'position_size': position_size,
        'position_value': position_value,
        'risk_amount': risk_amount,
        'leverage': leverage,
        'effective_risk_pct': (risk_amount / account_balance) * 100
    }


def generate_order_id(prefix: str = 'ORD') -> str:
    """Generate a unique order ID."""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    random_suffix = np.random.randint(1000, 9999)
    return f"{prefix}_{timestamp}_{random_suffix}"


def time_to_next_candle(timeframe: str) -> int:
    """
    Calculate seconds until next candle close.
    
    Args:
        timeframe: Timeframe string (e.g., '1h', '5m')
    
    Returns:
        Seconds until next candle
    """
    now = datetime.now()
    
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        next_candle = now.replace(second=0, microsecond=0)
        next_candle += timedelta(minutes=minutes - (now.minute % minutes))
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        next_candle = now.replace(minute=0, second=0, microsecond=0)
        next_candle += timedelta(hours=hours - (now.hour % hours))
    elif timeframe.endswith('d'):
        next_candle = now.replace(hour=0, minute=0, second=0, microsecond=0)
        next_candle += timedelta(days=1)
    else:
        return 60  # Default to 1 minute
    
    return max(0, int((next_candle - now).total_seconds()))


def safe_divide(a: Union[float, np.ndarray],
                b: Union[float, np.ndarray],
                fill_value: float = 0.0) -> Union[float, np.ndarray]:
    """Safe division that handles division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result = np.where(np.isfinite(result), result, fill_value)
    return result


def normalize_data(
    data: pd.DataFrame,
    method: str = 'minmax',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize DataFrame columns.
    
    Args:
        data: Input DataFrame
        method: Normalization method ('minmax', 'zscore', 'robust')
        columns: Columns to normalize
    
    Returns:
        Normalized DataFrame
    """
    df = data.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'minmax':
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        elif method == 'zscore':
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        elif method == 'robust':
            median = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            df[col] = (df[col] - median) / iqr
    
    return df


def detect_outliers(
    data: pd.Series,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in a Series.
    
    Args:
        data: Input Series
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean Series indicating outliers
    """
    if method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (data < lower) | (data > upper)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    return pd.Series(False, index=data.index)


async def run_in_executor(func: Callable, *args, **kwargs) -> Any:
    """Run a synchronous function in an async context."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(func, *args, **kwargs)
    )
