# data/kline_builder.py
"""
K线技术指标计算
================
计算 MA、EMA、RSI、MACD、布林带等技术指标
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_ma(df: pd.DataFrame, periods: list = [5, 10, 20, 60]) -> pd.DataFrame:
    """计算移动平均线"""
    for period in periods:
        df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    return df


def calculate_ema(df: pd.DataFrame, periods: list = [12, 26]) -> pd.DataFrame:
    """计算指数移动平均线"""
    for period in periods:
        df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算相对强弱指数 RSI"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-8)
    df[f'rsi{period}'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """计算 MACD 指标"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    return df


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
    """计算布林带"""
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    
    df['bb_upper'] = df['bb_middle'] + std_dev * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - std_dev * df['bb_std']
    
    # 布林带宽度
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # 价格位置 (0-1)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算平均真实波幅 ATR"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f'atr{period}'] = tr.rolling(window=period).mean()
    
    return df


def calculate_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
    """计算成交量指标"""
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    return df


def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """计算动量指标"""
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['roc'] = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-8) * 100
    
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标
    
    Args:
        df: 原始K线数据
        
    Returns:
        添加指标后的 DataFrame
    """
    if df is None or len(df) < 60:
        return df
    
    df = df.copy()
    
    # 均线
    df = calculate_ma(df, [5, 10, 20, 60])
    df = calculate_ema(df, [12, 26])
    
    # RSI
    df = calculate_rsi(df, 14)
    
    # MACD
    df = calculate_macd(df)
    
    # 布林带
    df = calculate_bollinger_bands(df)
    
    # ATR
    df = calculate_atr(df, 14)
    
    # 成交量
    df = calculate_volume_profile(df)
    
    # 动量
    df = calculate_momentum(df)
    
    # 价格变化率
    df['price_change'] = df['close'].pct_change()
    df['price_change_5m'] = df['close'].pct_change(5)
    
    # 波动率
    df['volatility'] = df['price_change'].rolling(window=20).std() * np.sqrt(288)  # 年化
    
    return df


def get_latest_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    获取最新的指标值
    
    Args:
        df: 带指标的 DataFrame
        
    Returns:
        最新指标字典
    """
    if df is None or len(df) == 0:
        return {}
    
    latest = df.iloc[-1]
    
    return {
        'close': latest.get('close', 0),
        'volume': latest.get('volume', 0),
        'ma5': latest.get('ma5', 0),
        'ma10': latest.get('ma10', 0),
        'ma20': latest.get('ma20', 0),
        'ma60': latest.get('ma60', 0),
        'rsi14': latest.get('rsi14', 50),
        'macd': latest.get('macd', 0),
        'macd_signal': latest.get('macd_signal', 0),
        'macd_histogram': latest.get('macd_histogram', 0),
        'bb_upper': latest.get('bb_upper', 0),
        'bb_lower': latest.get('bb_lower', 0),
        'bb_position': latest.get('bb_position', 0.5),
        'atr14': latest.get('atr14', 0),
        'volume_ratio': latest.get('volume_ratio', 1),
        'momentum': latest.get('momentum', 0),
        'volatility': latest.get('volatility', 0),
    }


# 测试
if __name__ == "__main__":
    from data.market_stream import get_realtime_eth_data
    
    df, price = get_realtime_eth_data()
    
    if df is not None:
        df = calculate_indicators(df)
        indicators = get_latest_indicators(df)
        
        print("最新技术指标:")
        for key, value in indicators.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
