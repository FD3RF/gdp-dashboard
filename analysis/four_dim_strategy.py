"""
ETH 5分钟四维共振策略模块
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    STANDARD_BULL = "标准多头"
    STANDARD_BEAR = "标准空头"
    TRAP_BULL = "诱空陷阱做多"
    TRAP_BEAR = "诱多陷阱做空"
    PLATFORM_BULL = "平台突破做多"
    PLATFORM_BEAR = "平台跌破做空"
    NONE = "无信号"


@dataclass
class FourDimSignal:
    timestamp: datetime
    signal_type: SignalType
    direction: str
    price: float
    stop_loss: float
    take_profit: float
    atr: float
    reason: str
    volume_signal: str = ""
    macd_signal: str = ""
    sar_signal: str = ""
    support_resistance: str = ""


class FourDimConfig:
    VOL_LOOKBACK = 5
    VOL_SHRINK_RATIO = 0.6
    VOL_EXPAND_RATIO = 1.5
    VOL_PANIC_RATIO = 3.0
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    SAR_START = 0.02
    SAR_INC = 0.02
    SAR_MAX = 0.2
    SUPPORT_LOOKBACK = 20
    RANGE_PERCENT = 0.3
    CONSOLIDATION_BARS = 5
    SL_ATR_MULT = 1.5
    TP_ATR_MULT = 3.0
    USE_EMA_FILTER = True
    USE_VOLATILITY_FILTER = True
    USE_RSI_FILTER = True
    EMA50_LEN = 50
    RSI_LEN = 14


class FourDimStrategy:
    def __init__(self, config: FourDimConfig = None):
        self.config = config or FourDimConfig()
    
    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 成交量
        df['vol_sma'] = df['volume'].rolling(window=self.config.VOL_LOOKBACK).mean()
        df['is_shrink'] = df['volume'] < df['vol_sma'] * self.config.VOL_SHRINK_RATIO
        df['is_expand'] = df['volume'] > df['vol_sma'] * self.config.VOL_EXPAND_RATIO
        df['is_panic'] = df['volume'] > df['vol_sma'] * self.config.VOL_PANIC_RATIO
        
        # MACD
        ema_fast = df['close'].ewm(span=self.config.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.config.MACD_SLOW, adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow
        df['signal_line'] = df['macd_line'].ewm(span=self.config.MACD_SIGNAL, adjust=False).mean()
        df['hist'] = df['macd_line'] - df['signal_line']
        df['hist_rising'] = df['hist'] > df['hist'].shift(1)
        df['hist_falling'] = df['hist'] < df['hist'].shift(1)
        df['hist_cross_up'] = (df['hist'] > 0) & (df['hist'].shift(1) <= 0)
        df['hist_cross_down'] = (df['hist'] < 0) & (df['hist'].shift(1) >= 0)
        
        # SAR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        sar = np.zeros(len(df))
        ep, af, is_long = 0.0, self.config.SAR_START, True
        sar[0] = low[0]
        
        for i in range(1, len(df)):
            if is_long:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                if low[i] < sar[i]:
                    is_long, sar[i], ep, af = False, ep, low[i], self.config.SAR_START
                elif high[i] > ep:
                    ep, af = high[i], min(af + self.config.SAR_INC, self.config.SAR_MAX)
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                if high[i] > sar[i]:
                    is_long, sar[i], ep, af = True, ep, high[i], self.config.SAR_START
                elif low[i] < ep:
                    ep, af = low[i], min(af + self.config.SAR_INC, self.config.SAR_MAX)
        
        df['sar'] = sar
        df['is_green_triangle'] = (close > sar) & (df['close'].shift(1) <= np.roll(sar, 1))
        df['is_purple_triangle'] = (close < sar) & (df['close'].shift(1) >= np.roll(sar, 1))
        
        # 支撑阻力
        df['support_level'] = df['low'].rolling(window=self.config.SUPPORT_LOOKBACK).min()
        df['resistance_level'] = df['high'].rolling(window=self.config.SUPPORT_LOOKBACK).max()
        
        # ATR
        tr = pd.concat([df['high'] - df['low'], 
                       abs(df['high'] - df['close'].shift(1)),
                       abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        df['volatility_low'] = df['atr'] < df['atr_ma']
        
        # EMA
        df['ema50'] = df['close'].ewm(span=self.config.EMA50_LEN, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.RSI_LEN).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.RSI_LEN).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # 横盘
        df['highest_range'] = df['high'].rolling(window=self.config.CONSOLIDATION_BARS).max()
        df['lowest_range'] = df['low'].rolling(window=self.config.CONSOLIDATION_BARS).min()
        df['range_width'] = (df['highest_range'] - df['lowest_range']) / df['lowest_range'] * 100
        df['is_consolidation'] = df['range_width'] < self.config.RANGE_PERCENT
        df['all_shrink'] = df['volume'].rolling(window=self.config.CONSOLIDATION_BARS).mean() < df['vol_sma'] * self.config.VOL_SHRINK_RATIO
        
        # 背离
        df['price_ll'] = df['low'].rolling(window=20).min()
        df['rsi_ll'] = df['rsi'].rolling(window=20).min()
        df['bull_div'] = (df['low'] == df['price_ll']) & (df['rsi'] > df['rsi_ll'].shift(1))
        df['price_hh'] = df['high'].rolling(window=20).max()
        df['rsi_hh'] = df['rsi'].rolling(window=20).max()
        df['bear_div'] = (df['high'] == df['price_hh']) & (df['rsi'] < df['rsi_hh'].shift(1))
        
        # K线形态
        df['bearish_candle'] = (df['close'] < df['open']) & ((df['open'] - df['close']) / df['open'] > 0.002)
        df['bullish_candle'] = (df['close'] > df['open']) & ((df['close'] - df['open']) / df['open'] > 0.002)
        df['last_bear_high'] = df['high'].where(df['bearish_candle']).ffill()
        df['last_bull_low'] = df['low'].where(df['bullish_candle']).ffill()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._calc_indicators(df)
        
        df['long_signal'] = False
        df['short_signal'] = False
        df['signal_type'] = SignalType.NONE.value
        df['signal_reason'] = ""
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # 标准多头
        near_support = (df['close'] >= df['support_level']) & (df['close'] < df['support_level'] * 1.01)
        shrink_at_support = near_support & df['is_shrink'] & (df['low'] > df['support_level'])
        breakout_bull = df['is_expand'] & (df['close'] > df['last_bear_high'].shift(1))
        standard_bull = (shrink_at_support.shift(1) | df['is_consolidation'].shift(1)) & breakout_bull & df['hist_cross_up'] & df['is_green_triangle']
        
        # 标准空头
        near_resistance = (df['close'] <= df['resistance_level']) & (df['close'] > df['resistance_level'] * 0.99)
        shrink_at_resistance = near_resistance & df['is_shrink'] & (df['high'] < df['resistance_level'])
        breakdown_bear = df['is_expand'] & (df['close'] < df['last_bull_low'].shift(1))
        standard_bear = (shrink_at_resistance.shift(1) | df['is_consolidation'].shift(1)) & breakdown_bear & df['hist_cross_down'] & df['is_purple_triangle']
        
        # 诱空陷阱
        panic_down = df['is_panic'] & (df['low'] < df['support_level']) & (df['close'] > df['support_level'])
        confirm_up = df['close'] > (df['high'].shift(1) * 0.5 + df['low'].shift(1) * 0.5)
        trap_bull = panic_down.shift(1) & confirm_up & df['is_green_triangle'] & (df['hist'].shift(1) < 0) & (df['hist'] > df['hist'].shift(1))
        if self.config.USE_RSI_FILTER:
            trap_bull = trap_bull & df['bull_div']
        
        # 诱多陷阱
        panic_up = df['is_panic'] & (df['high'] > df['resistance_level']) & (df['close'] < df['resistance_level'])
        confirm_down = df['close'] < (df['high'].shift(1) + df['low'].shift(1)) / 2
        trap_bear = panic_up.shift(1) & confirm_down & df['is_purple_triangle'] & (df['hist'].shift(1) > 0) & (df['hist'] < df['hist'].shift(1))
        if self.config.USE_RSI_FILTER:
            trap_bear = trap_bear & df['bear_div']
        
        # 平台突破
        consolidation_zone = df['is_consolidation'] & df['all_shrink']
        breakout_up = (df['close'] > df['highest_range'].shift(1)) & df['is_expand']
        platform_bull = consolidation_zone.shift(1) & breakout_up & df['hist_cross_up'] & df['is_green_triangle']
        if self.config.USE_VOLATILITY_FILTER:
            platform_bull = platform_bull & ~df['volatility_low']
        
        # 平台跌破
        breakdown_down = (df['close'] < df['lowest_range'].shift(1)) & df['is_expand']
        platform_bear = consolidation_zone.shift(1) & breakdown_down & df['hist_cross_down'] & df['is_purple_triangle']
        if self.config.USE_VOLATILITY_FILTER:
            platform_bear = platform_bear & ~df['volatility_low']
        
        # 综合
        long_base = standard_bull | trap_bull | platform_bull
        short_base = standard_bear | trap_bear | platform_bear
        
        if self.config.USE_EMA_FILTER:
            long_base = long_base & (df['close'] > df['ema50'])
            short_base = short_base & (df['close'] < df['ema50'])
        
        df['long_signal'] = long_base
        df['short_signal'] = short_base
        
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
        
        # 止损止盈
        df.loc[df['long_signal'], 'stop_loss'] = df.loc[df['long_signal'], 'low'] - df.loc[df['long_signal'], 'atr'] * self.config.SL_ATR_MULT
        df.loc[df['long_signal'], 'take_profit'] = df.loc[df['long_signal'], 'close'] + df.loc[df['long_signal'], 'atr'] * self.config.TP_ATR_MULT
        df.loc[df['short_signal'], 'stop_loss'] = df.loc[df['short_signal'], 'high'] + df.loc[df['short_signal'], 'atr'] * self.config.SL_ATR_MULT
        df.loc[df['short_signal'], 'take_profit'] = df.loc[df['short_signal'], 'close'] - df.loc[df['short_signal'], 'atr'] * self.config.TP_ATR_MULT
        
        return df
    
    def get_latest_signal(self, df: pd.DataFrame) -> Optional[FourDimSignal]:
        df = self.generate_signals(df)
        if len(df) == 0:
            return None
        last = df.iloc[-1]
        if last['long_signal'] or last['short_signal']:
            return FourDimSignal(
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
                support_resistance=f"支撑{last['support_level']:.0f}/阻力{last['resistance_level']:.0f}"
            )
        return None
