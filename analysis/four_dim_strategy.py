"""
ETH 5分钟四维共振策略模块
============================
六种进场场景：
1. 标准多头 - 放量起涨，突破阴线，MACD翻红，绿三角现
2. 标准空头 - 放量下跌，跌破阳线，MACD翻绿，紫三角现
3. 诱空陷阱做多 - 放量暴跌后收回，假跌真买
4. 诱多陷阱做空 - 放量暴涨后回落，假涨真空
5. 平台突破做多 - 缩量横盘后放量突破
6. 平台跌破做空 - 缩量横盘后放量跌破
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    STANDARD_BULL = "标准多头"
    STANDARD_BEAR = "标准空头"
    TRAP_BULL = "诱空陷阱做多"
    TRAP_BEAR = "诱多陷阱做空"
    PLATFORM_BULL = "平台突破做多"
    PLATFORM_BEAR = "平台跌破做空"
    NONE = "无信号"


@dataclass
class FourDimSignal:
    """四维共振信号"""
    timestamp: datetime
    signal_type: SignalType
    direction: str  # 'long' or 'short'
    price: float
    stop_loss: float
    take_profit: float
    atr: float
    reason: str
    confidence: float = 1.0
    
    # 四维指标状态
    volume_signal: str = ""
    macd_signal: str = ""
    sar_signal: str = ""
    support_resistance: str = ""


class FourDimConfig:
    """四维共振策略配置"""
    # 成交量参数
    VOL_LOOKBACK = 5
    VOL_SHRINK_RATIO = 0.6
    VOL_EXPAND_RATIO = 1.5
    VOL_PANIC_RATIO = 3.0

    # MACD参数
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    # SAR参数
    SAR_START = 0.02
    SAR_INC = 0.02
    SAR_MAX = 0.2

    # 关键点位参数
    SUPPORT_LOOKBACK = 20
    RANGE_PERCENT = 0.3
    CONSOLIDATION_BARS = 5

    # 止损止盈参数
    SL_ATR_MULT = 1.5
    TP_ATR_MULT = 3.0
    USE_TRAILING = True

    # 过滤器
    USE_EMA_FILTER = True
    USE_VOLATILITY_FILTER = True
    USE_RSI_FILTER = True
    EMA50_LEN = 50
    RSI_LEN = 14


class FourDimIndicators:
    """四维指标计算器"""
    
    @staticmethod
    def calc_volume_signals(df: pd.DataFrame, config: FourDimConfig) -> pd.DataFrame:
        """计算成交量信号"""
        df = df.copy()
        df['vol_sma'] = df['volume'].rolling(window=config.VOL_LOOKBACK).mean()
        df['is_shrink'] = df['volume'] < df['vol_sma'] * config.VOL_SHRINK_RATIO
        df['is_expand'] = df['volume'] > df['vol_sma'] * config.VOL_EXPAND_RATIO
        df['is_panic'] = df['volume'] > df['vol_sma'] * config.VOL_PANIC_RATIO
        return df
    
    @staticmethod
    def calc_macd(df: pd.DataFrame, config: FourDimConfig) -> pd.DataFrame:
        """计算MACD"""
        df = df.copy()
        ema_fast = df['close'].ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=config.MACD_SLOW, adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow
        df['signal_line'] = df['macd_line'].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        df['hist'] = df['macd_line'] - df['signal_line']
        df['hist_rising'] = df['hist'] > df['hist'].shift(1)
        df['hist_falling'] = df['hist'] < df['hist'].shift(1)
        df['hist_cross_up'] = (df['hist'] > 0) & (df['hist'].shift(1) <= 0)
        df['hist_cross_down'] = (df['hist'] < 0) & (df['hist'].shift(1) >= 0)
        return df
    
    @staticmethod
    def calc_sar(df: pd.DataFrame, config: FourDimConfig) -> pd.DataFrame:
        """计算SAR抛物线"""
        df = df.copy()
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        sar = np.zeros(len(df))
        ep = 0.0
        af = config.SAR_START
        is_long = True
        sar[0] = low[0]
        
        for i in range(1, len(df)):
            if is_long:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                if low[i] < sar[i]:
                    is_long = False
                    sar[i] = ep
                    ep = low[i]
                    af = config.SAR_START
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + config.SAR_INC, config.SAR_MAX)
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                if high[i] > sar[i]:
                    is_long = True
                    sar[i] = ep
                    ep = high[i]
                    af = config.SAR_START
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + config.SAR_INC, config.SAR_MAX)
        
        df['sar'] = sar
        df['is_green_triangle'] = (close > sar) & (df['close'].shift(1) <= np.roll(sar, 1))
        df['is_purple_triangle'] = (close < sar) & (df['close'].shift(1) >= np.roll(sar, 1))
        return df
    
    @staticmethod
    def calc_support_resistance(df: pd.DataFrame, config: FourDimConfig) -> pd.DataFrame:
        """计算支撑阻力"""
        df = df.copy()
        df['support_level'] = df['low'].rolling(window=config.SUPPORT_LOOKBACK).min()
        df['resistance_level'] = df['high'].rolling(window=config.SUPPORT_LOOKBACK).max()
        return df
    
    @staticmethod
    def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算ATR"""
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        df['volatility_low'] = df['atr'] < df['atr_ma']
        return df
    
    @staticmethod
    def calc_ema(df: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """计算EMA"""
        df = df.copy()
        df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算RSI"""
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def detect_consolidation(df: pd.DataFrame, config: FourDimConfig) -> pd.DataFrame:
        """检测横盘"""
        df = df.copy()
        df['highest_range'] = df['high'].rolling(window=config.CONSOLIDATION_BARS).max()
        df['lowest_range'] = df['low'].rolling(window=config.CONSOLIDATION_BARS).min()
        df['range_width'] = (df['highest_range'] - df['lowest_range']) / df['lowest_range'] * 100
        df['is_consolidation'] = df['range_width'] < config.RANGE_PERCENT
        df['all_shrink'] = df['volume'].rolling(window=config.CONSOLIDATION_BARS).mean() < df['vol_sma'] * config.VOL_SHRINK_RATIO
        return df
    
    @staticmethod
    def detect_divergence(df: pd.DataFrame, config: FourDimConfig) -> pd.DataFrame:
        """检测RSI背离"""
        df = df.copy()
        df = FourDimIndicators.calc_rsi(df, config.RSI_LEN)
        
        lookback = 20
        df['price_ll'] = df['low'].rolling(window=lookback).min()
        df['rsi_ll'] = df['rsi'].rolling(window=lookback).min()
        df['bull_div'] = (df['low'] == df['price_ll']) & (df['rsi'] > df['rsi_ll'].shift(1))
        
        df['price_hh'] = df['high'].rolling(window=lookback).max()
        df['rsi_hh'] = df['rsi'].rolling(window=lookback).max()
        df['bear_div'] = (df['high'] == df['price_hh']) & (df['rsi'] < df['rsi_hh'].shift(1))
        return df


class FourDimStrategy:
    """
    四维共振策略引擎
    
    四维：成交量 + MACD + SAR + 支撑阻力
    """
    
    def __init__(self, config: FourDimConfig = None):
        self.config = config or FourDimConfig()
        self.signal_history: List[FourDimSignal] = []
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有指标"""
        df = FourDimIndicators.calc_volume_signals(df, self.config)
        df = FourDimIndicators.calc_macd(df, self.config)
        df = FourDimIndicators.calc_sar(df, self.config)
        df = FourDimIndicators.calc_support_resistance(df, self.config)
        df = FourDimIndicators.calc_atr(df, 14)
        df = FourDimIndicators.calc_ema(df, self.config.EMA50_LEN)
        df = FourDimIndicators.detect_consolidation(df, self.config)
        df = FourDimIndicators.detect_divergence(df, self.config)
        
        # 识别明显阴线/阳线
        df['bearish_candle'] = (df['close'] < df['open']) & \
                               ((df['open'] - df['close']) / df['open'] > 0.002)
        df['bullish_candle'] = (df['close'] > df['open']) & \
                               ((df['close'] - df['open']) / df['open'] > 0.002)
        
        # 最近明显阴线高点
        df['last_bear_high'] = df['high'].where(df['bearish_candle']).ffill()
        # 最近明显阳线低点
        df['last_bull_low'] = df['low'].where(df['bullish_candle']).ffill()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = self.calculate_all_indicators(df)
        df = df.copy()
        
        df['long_signal'] = False
        df['short_signal'] = False
        df['signal_type'] = SignalType.NONE.value
        df['signal_reason'] = ""
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # ========== 标准多头 ==========
        near_support = (df['close'] >= df['support_level']) & \
                       (df['close'] < df['support_level'] * 1.01)
        shrink_at_support = near_support & df['is_shrink'] & \
                           (df['low'] > df['support_level'])
        breakout_bull = df['is_expand'] & (df['close'] > df['last_bear_high'].shift(1))
        
        standard_bull = (shrink_at_support.shift(1) | df['is_consolidation'].shift(1)) & \
                       breakout_bull & df['hist_cross_up'] & df['is_green_triangle']
        
        # ========== 标准空头 ==========
        near_resistance = (df['close'] <= df['resistance_level']) & \
                         (df['close'] > df['resistance_level'] * 0.99)
        shrink_at_resistance = near_resistance & df['is_shrink'] & \
                              (df['high'] < df['resistance_level'])
        breakdown_bear = df['is_expand'] & (df['close'] < df['last_bull_low'].shift(1))
        
        standard_bear = (shrink_at_resistance.shift(1) | df['is_consolidation'].shift(1)) & \
                       breakdown_bear & df['hist_cross_down'] & df['is_purple_triangle']
        
        # ========== 诱空陷阱 (做多) ==========
        panic_down = df['is_panic'] & (df['low'] < df['support_level']) & \
                    (df['close'] > df['support_level'])
        confirm_up = df['close'] > (df['high'].shift(1) * 0.5 + df['low'].shift(1) * 0.5)
        
        trap_bull = panic_down.shift(1) & confirm_up & df['is_green_triangle'] & \
                   (df['hist'].shift(1) < 0) & (df['hist'] > df['hist'].shift(1))
        if self.config.USE_RSI_FILTER:
            trap_bull = trap_bull & df['bull_div']
        
        # ========== 诱多陷阱 (做空) ==========
        panic_up = df['is_panic'] & (df['high'] > df['resistance_level']) & \
                  (df['close'] < df['resistance_level'])
        confirm_down = df['close'] < (df['high'].shift(1) + df['low'].shift(1)) / 2
        
        trap_bear = panic_up.shift(1) & confirm_down & df['is_purple_triangle'] & \
                   (df['hist'].shift(1) > 0) & (df['hist'] < df['hist'].shift(1))
        if self.config.USE_RSI_FILTER:
            trap_bear = trap_bear & df['bear_div']
        
        # ========== 平台突破 (做多) ==========
        consolidation_zone = df['is_consolidation'] & df['all_shrink']
        breakout_up = (df['close'] > df['highest_range'].shift(1)) & df['is_expand']
        
        platform_bull = consolidation_zone.shift(1) & breakout_up & \
                       df['hist_cross_up'] & df['is_green_triangle']
        if self.config.USE_VOLATILITY_FILTER:
            platform_bull = platform_bull & ~df['volatility_low']
        
        # ========== 平台跌破 (做空) ==========
        breakdown_down = (df['close'] < df['lowest_range'].shift(1)) & df['is_expand']
        
        platform_bear = consolidation_zone.shift(1) & breakdown_down & \
                       df['hist_cross_down'] & df['is_purple_triangle']
        if self.config.USE_VOLATILITY_FILTER:
            platform_bear = platform_bear & ~df['volatility_low']
        
        # ========== 综合信号 ==========
        long_base = (standard_bull | trap_bull | platform_bull)
        short_base = (standard_bear | trap_bear | platform_bear)
        
        # EMA趋势过滤
        if self.config.USE_EMA_FILTER:
            ema_col = f'ema{self.config.EMA50_LEN}'
            long_base = long_base & (df['close'] > df[ema_col])
            short_base = short_base & (df['close'] < df[ema_col])
        
        df['long_signal'] = long_base
        df['short_signal'] = short_base
        
        # 信号类型和原因
        df.loc[standard_bull, 'signal_type'] = SignalType.STANDARD_BULL.value
        df.loc[standard_bull, 'signal_reason'] = "放量起涨，突破阴线，MACD翻红，绿三角现 → 直接开多"
        
        df.loc[standard_bear, 'signal_type'] = SignalType.STANDARD_BEAR.value
        df.loc[standard_bear, 'signal_reason'] = "放量下跌，跌破阳线，MACD翻绿，紫三角现 → 直接开空"
        
        df.loc[trap_bull, 'signal_type'] = SignalType.TRAP_BULL.value
        df.loc[trap_bull, 'signal_reason'] = "放量暴跌，低点不破，MACD缩短，绿三角现 → 假跌真买"
        
        df.loc[trap_bear, 'signal_type'] = SignalType.TRAP_BEAR.value
        df.loc[trap_bear, 'signal_reason'] = "放量暴涨，高点不破，MACD缩短，紫三角现 → 假涨真空"
        
        df.loc[platform_bull, 'signal_type'] = SignalType.PLATFORM_BULL.value
        df.loc[platform_bull, 'signal_reason'] = "缩量横盘，低点托住，MACD金叉，绿三角现 → 埋伏等涨"
        
        df.loc[platform_bear, 'signal_type'] = SignalType.PLATFORM_BEAR.value
        df.loc[platform_bear, 'signal_reason'] = "缩量横盘，高点压住，MACD死叉，紫三角现 → 埋伏等跌"
        
        # 计算止损止盈
        df.loc[df['long_signal'], 'stop_loss'] = df.loc[df['long_signal'], 'low'] - \
                                                   df.loc[df['long_signal'], 'atr'] * self.config.SL_ATR_MULT
        df.loc[df['long_signal'], 'take_profit'] = df.loc[df['long_signal'], 'close'] + \
                                                    df.loc[df['long_signal'], 'atr'] * self.config.TP_ATR_MULT
        
        df.loc[df['short_signal'], 'stop_loss'] = df.loc[df['short_signal'], 'high'] + \
                                                   df.loc[df['short_signal'], 'atr'] * self.config.SL_ATR_MULT
        df.loc[df['short_signal'], 'take_profit'] = df.loc[df['short_signal'], 'close'] - \
                                                     df.loc[df['short_signal'], 'atr'] * self.config.TP_ATR_MULT
        
        return df
    
    def get_latest_signal(self, df: pd.DataFrame) -> Optional[FourDimSignal]:
        """获取最新信号"""
        df = self.generate_signals(df)
        
        if len(df) == 0:
            return None
        
        last = df.iloc[-1]
        
        if last['long_signal'] or last['short_signal']:
            signal = FourDimSignal(
                timestamp=last.name if isinstance(last.name, datetime) else datetime.now(),
                signal_type=SignalType(last['signal_type']),
                direction='long' if last['long_signal'] else 'short',
                price=last['close'],
                stop_loss=last['stop_loss'],
                take_profit=last['take_profit'],
                atr=last['atr'],
                reason=last['signal_reason'],
                volume_signal="放量" if last['is_expand'] else ("缩量" if last['is_shrink'] else "正常"),
                macd_signal="金叉" if last['hist_cross_up'] else ("死叉" if last['hist_cross_down'] else "运行中"),
                sar_signal="绿三角" if last['is_green_triangle'] else ("紫三角" if last['is_purple_triangle'] else "跟随"),
                support_resistance=f"支撑{last['support_level']:.2f}/阻力{last['resistance_level']:.2f}"
            )
            self.signal_history.append(signal)
            return signal
        
        return None
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取信号摘要"""
        signal = self.get_latest_signal(df)
        
        if signal is None:
            return {
                "status": "no_signal",
                "message": "当前无交易信号"
            }
        
        return {
            "status": "signal",
            "timestamp": signal.timestamp.isoformat(),
            "signal_type": signal.signal_type.value,
            "direction": signal.direction,
            "price": round(signal.price, 2),
            "stop_loss": round(signal.stop_loss, 2),
            "take_profit": round(signal.take_profit, 2),
            "risk_reward": round(abs(signal.take_profit - signal.price) / abs(signal.price - signal.stop_loss), 2),
            "reason": signal.reason,
            "indicators": {
                "volume": signal.volume_signal,
                "macd": signal.macd_signal,
                "sar": signal.sar_signal,
                "support_resistance": signal.support_resistance
            }
        }


# 便捷函数
def generate_four_dim_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    生成四维共振信号
    
    Args:
        df: 包含 open, high, low, close, volume 的 DataFrame
        
    Returns:
        信号摘要字典
    """
    strategy = FourDimStrategy()
    return strategy.get_signal_summary(df)
