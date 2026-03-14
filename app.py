"""
ETH 5分钟四维共振策略 - 完整版
实时数据 + 优化参数 + 回测统计
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import time

# ==================== 实时数据获取 ====================

@st.cache_data(ttl=60)
def fetch_binance_klines(symbol: str = "ETHUSDT", interval: str = "5m", limit: int = 500) -> pd.DataFrame:
    """从币安API获取实时K线数据"""
    import urllib.request
    import json
    
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.warning(f"获取实时数据失败: {e}，使用模拟数据")
        return None


def generate_sample_data(n_bars: int = 500, seed: int = None) -> pd.DataFrame:
    """生成模拟数据（趋势+波动）"""
    if seed is not None:
        np.random.seed(seed)
    
    price = 2000.0
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='5min')
    opens, highs, lows, closes, volumes = [], [], [], [], []
    
    # 创建趋势和波动
    trend = np.cumsum(np.random.randn(n_bars) * 0.001)
    
    for i in range(n_bars):
        base_change = trend[i] + np.random.normal(0, 0.003)
        open_price = price
        close_price = price * (1 + base_change)
        vol = abs(np.random.normal(0.003, 0.002))
        high_price = max(open_price, close_price) * (1 + vol)
        low_price = min(open_price, close_price) * (1 - vol)
        
        # 添加一些极端成交量
        if np.random.random() < 0.05:
            volume = 1000 * np.random.uniform(2.5, 4.0)  # 极端放量
        elif np.random.random() < 0.15:
            volume = 1000 * np.random.uniform(0.3, 0.5)  # 缩量
        else:
            volume = 1000 * np.random.uniform(0.8, 1.5)  # 正常
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)
        price = close_price
    
    return pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes}, index=dates)


# ==================== 策略核心 ====================

class SignalType(Enum):
    STANDARD_BULL = "标准多头"
    STANDARD_BEAR = "标准空头"
    TRAP_BULL = "诱空陷阱做多"
    TRAP_BEAR = "诱多陷阱做空"
    PLATFORM_BULL = "平台突破做多"
    PLATFORM_BEAR = "平台跌破做空"
    NONE = "无信号"


@dataclass
class TradeSignal:
    signal_type: SignalType
    direction: str
    price: float
    stop_loss: float
    take_profit: float
    atr: float
    reason: str
    timestamp: datetime = None
    volume_signal: str = ""
    macd_signal: str = ""
    sar_signal: str = ""
    support_resistance: str = ""


class FourDimStrategy:
    """四维共振策略 - 成交量 + MACD + SAR + 支撑阻力"""
    
    # 优化后的默认参数（增加信号频率）
    DEFAULT_CONFIG = {
        'vol_lookback': 5,
        'vol_shrink_ratio': 0.5,      # 降低，更容易识别缩量
        'vol_expand_ratio': 1.3,       # 降低，更容易识别放量
        'vol_panic_ratio': 2.5,        # 降低，更容易识别极端放量
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'sar_start': 0.02,
        'sar_inc': 0.02,
        'sar_max': 0.2,
        'support_lookback': 20,
        'range_percent': 0.5,          # 提高，更容易识别横盘
        'consolidation_bars': 5,
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 3.0,
        'use_ema_filter': False,       # 默认关闭，增加信号
        'use_volatility_filter': False, # 默认关闭，增加信号
        'use_rsi_filter': False,        # 默认关闭，增加信号
        'ema50_len': 50,
        'rsi_len': 14,
    }
    
    def __init__(self, config=None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()
        cfg = self.config
        
        # 成交量
        df['vol_sma'] = df['volume'].rolling(window=cfg['vol_lookback']).mean()
        df['is_shrink'] = df['volume'] < df['vol_sma'] * cfg['vol_shrink_ratio']
        df['is_expand'] = df['volume'] > df['vol_sma'] * cfg['vol_expand_ratio']
        df['is_panic'] = df['volume'] > df['vol_sma'] * cfg['vol_panic_ratio']
        
        # MACD
        ema_fast = df['close'].ewm(span=cfg['macd_fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=cfg['macd_slow'], adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow
        df['signal_line'] = df['macd_line'].ewm(span=cfg['macd_signal'], adjust=False).mean()
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
        ep, af, is_long = 0.0, cfg['sar_start'], True
        sar[0] = low[0]
        
        for i in range(1, len(df)):
            if is_long:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                if low[i] < sar[i]:
                    is_long, sar[i], ep, af = False, ep, low[i], cfg['sar_start']
                elif high[i] > ep:
                    ep, af = high[i], min(af + cfg['sar_inc'], cfg['sar_max'])
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                if high[i] > sar[i]:
                    is_long, sar[i], ep, af = True, ep, high[i], cfg['sar_start']
                elif low[i] < ep:
                    ep, af = low[i], min(af + cfg['sar_inc'], cfg['sar_max'])
        
        df['sar'] = sar
        df['is_green_triangle'] = (close > sar) & (df['close'].shift(1) <= np.roll(sar, 1))
        df['is_purple_triangle'] = (close < sar) & (df['close'].shift(1) >= np.roll(sar, 1))
        
        # 支撑阻力
        df['support_level'] = df['low'].rolling(window=cfg['support_lookback']).min()
        df['resistance_level'] = df['high'].rolling(window=cfg['support_lookback']).max()
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        df['volatility_low'] = df['atr'] < df['atr_ma']
        
        # EMA50
        df['ema50'] = df['close'].ewm(span=cfg['ema50_len'], adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=cfg['rsi_len']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=cfg['rsi_len']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 背离检测
        df['price_ll'] = df['low'].rolling(window=20).min()
        df['rsi_ll'] = df['rsi'].rolling(window=20).min()
        df['bull_div'] = (df['low'] == df['price_ll']) & (df['rsi'] > df['rsi_ll'].shift(1))
        df['price_hh'] = df['high'].rolling(window=20).max()
        df['rsi_hh'] = df['rsi'].rolling(window=20).max()
        df['bear_div'] = (df['high'] == df['price_hh']) & (df['rsi'] < df['rsi_hh'].shift(1))
        
        # 横盘检测
        df['highest_range'] = df['high'].rolling(window=cfg['consolidation_bars']).max()
        df['lowest_range'] = df['low'].rolling(window=cfg['consolidation_bars']).min()
        df['range_width'] = (df['highest_range'] - df['lowest_range']) / df['lowest_range'] * 100
        df['is_consolidation'] = df['range_width'] < cfg['range_percent']
        df['all_shrink'] = df['volume'].rolling(window=cfg['consolidation_bars']).mean() < df['vol_sma'] * cfg['vol_shrink_ratio']
        
        # K线形态
        df['bearish_candle'] = (df['close'] < df['open']) & ((df['open'] - df['close']) / df['open'] > 0.002)
        df['bullish_candle'] = (df['close'] > df['open']) & ((df['close'] - df['open']) / df['open'] > 0.002)
        df['last_bear_high'] = df['high'].where(df['bearish_candle']).ffill()
        df['last_bull_low'] = df['low'].where(df['bullish_candle']).ffill()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = self.calculate_indicators(df)
        cfg = self.config
        
        df['long_signal'] = False
        df['short_signal'] = False
        df['signal_type'] = SignalType.NONE.value
        df['signal_reason'] = ""
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # 1. 标准多头
        near_support = (df['close'] >= df['support_level']) & (df['close'] < df['support_level'] * 1.02)
        shrink_at_support = near_support & df['is_shrink'] & (df['low'] > df['support_level'] * 0.998)
        breakout_bull = df['is_expand'] & (df['close'] > df['last_bear_high'].shift(1))
        standard_bull = (shrink_at_support.shift(1) | df['is_consolidation'].shift(1)) & breakout_bull & df['hist_cross_up'] & df['is_green_triangle']
        
        # 2. 标准空头
        near_resistance = (df['close'] <= df['resistance_level']) & (df['close'] > df['resistance_level'] * 0.98)
        shrink_at_resistance = near_resistance & df['is_shrink'] & (df['high'] < df['resistance_level'] * 1.002)
        breakdown_bear = df['is_expand'] & (df['close'] < df['last_bull_low'].shift(1))
        standard_bear = (shrink_at_resistance.shift(1) | df['is_consolidation'].shift(1)) & breakdown_bear & df['hist_cross_down'] & df['is_purple_triangle']
        
        # 3. 诱空陷阱做多
        panic_down = df['is_panic'] & (df['low'] < df['support_level'] * 1.005) & (df['close'] > df['support_level'])
        confirm_up = df['close'] > (df['high'].shift(1) * 0.5 + df['low'].shift(1) * 0.5)
        trap_bull = panic_down.shift(1) & confirm_up & df['is_green_triangle'] & (df['hist'].shift(1) < 0) & (df['hist'] > df['hist'].shift(1))
        if cfg['use_rsi_filter']:
            trap_bull = trap_bull & df['bull_div']
        
        # 4. 诱多陷阱做空
        panic_up = df['is_panic'] & (df['high'] > df['resistance_level'] * 0.995) & (df['close'] < df['resistance_level'])
        confirm_down = df['close'] < (df['high'].shift(1) + df['low'].shift(1)) / 2
        trap_bear = panic_up.shift(1) & confirm_down & df['is_purple_triangle'] & (df['hist'].shift(1) > 0) & (df['hist'] < df['hist'].shift(1))
        if cfg['use_rsi_filter']:
            trap_bear = trap_bear & df['bear_div']
        
        # 5. 平台突破做多
        consolidation_zone = df['is_consolidation'] & df['all_shrink']
        breakout_up = (df['close'] > df['highest_range'].shift(1)) & df['is_expand']
        platform_bull = consolidation_zone.shift(1) & breakout_up & df['hist_cross_up'] & df['is_green_triangle']
        if cfg['use_volatility_filter']:
            platform_bull = platform_bull & ~df['volatility_low']
        
        # 6. 平台跌破做空
        breakdown_down = (df['close'] < df['lowest_range'].shift(1)) & df['is_expand']
        platform_bear = consolidation_zone.shift(1) & breakdown_down & df['hist_cross_down'] & df['is_purple_triangle']
        if cfg['use_volatility_filter']:
            platform_bear = platform_bear & ~df['volatility_low']
        
        # 综合信号
        long_base = standard_bull | trap_bull | platform_bull
        short_base = standard_bear | trap_bear | platform_bear
        
        if cfg['use_ema_filter']:
            long_base = long_base & (df['close'] > df['ema50'])
            short_base = short_base & (df['close'] < df['ema50'])
        
        df['long_signal'] = long_base
        df['short_signal'] = short_base
        
        # 信号类型
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
        df.loc[df['long_signal'], 'stop_loss'] = df.loc[df['long_signal'], 'low'] - df.loc[df['long_signal'], 'atr'] * cfg['sl_atr_mult']
        df.loc[df['long_signal'], 'take_profit'] = df.loc[df['long_signal'], 'close'] + df.loc[df['long_signal'], 'atr'] * cfg['tp_atr_mult']
        df.loc[df['short_signal'], 'stop_loss'] = df.loc[df['short_signal'], 'high'] + df.loc[df['short_signal'], 'atr'] * cfg['sl_atr_mult']
        df.loc[df['short_signal'], 'take_profit'] = df.loc[df['short_signal'], 'close'] - df.loc[df['short_signal'], 'atr'] * cfg['tp_atr_mult']
        
        return df
    
    def get_all_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        """获取所有历史信号"""
        signals = []
        signal_rows = df[(df['long_signal'] == True) | (df['short_signal'] == True)]
        
        for idx, row in signal_rows.iterrows():
            signals.append(TradeSignal(
                signal_type=SignalType(row['signal_type']),
                direction='long' if row['long_signal'] else 'short',
                price=row['close'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                atr=row['atr'],
                reason=row['signal_reason'],
                timestamp=idx,
                volume_signal="放量" if row['is_expand'] else ("缩量" if row['is_shrink'] else "正常"),
                macd_signal="金叉" if row['hist_cross_up'] else ("死叉" if row['hist_cross_down'] else "运行中"),
                sar_signal="绿三角" if row['is_green_triangle'] else ("紫三角" if row['is_purple_triangle'] else "跟随"),
                support_resistance=f"支撑{row['support_level']:.0f}/阻力{row['resistance_level']:.0f}"
            ))
        
        return signals
    
    def get_latest_signal(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        """获取最新信号"""
        if len(df) == 0:
            return None
        last = df.iloc[-1]
        if last['long_signal'] or last['short_signal']:
            return TradeSignal(
                signal_type=SignalType(last['signal_type']),
                direction='long' if last['long_signal'] else 'short',
                price=last['close'],
                stop_loss=last['stop_loss'],
                take_profit=last['take_profit'],
                atr=last['atr'],
                reason=last['signal_reason'],
                timestamp=df.index[-1],
                volume_signal="放量" if last['is_expand'] else ("缩量" if last['is_shrink'] else "正常"),
                macd_signal="金叉" if last['hist_cross_up'] else ("死叉" if last['hist_cross_down'] else "运行中"),
                sar_signal="绿三角" if last['is_green_triangle'] else ("紫三角" if last['is_purple_triangle'] else "跟随"),
                support_resistance=f"支撑{last['support_level']:.0f}/阻力{last['resistance_level']:.0f}"
            )
        return None


# ==================== 回测引擎 ====================

def run_backtest(df: pd.DataFrame, initial_capital: float = 10000, position_size: float = 0.02) -> dict:
    """简单回测"""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [initial_capital]
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['long_signal'] and position == 0:
            position = capital * position_size / row['close']
            entry_price = row['close']
            stop_loss = row['stop_loss']
            take_profit = row['take_profit']
            capital -= position * entry_price
            trades.append({'type': 'BUY', 'price': entry_price, 'time': idx})
            
        elif row['short_signal'] and position == 0:
            position = -capital * position_size / row['close']
            entry_price = row['close']
            stop_loss = row['stop_loss']
            take_profit = row['take_profit']
            capital -= abs(position) * entry_price
            trades.append({'type': 'SELL', 'price': entry_price, 'time': idx})
            
        # 止损止盈检查
        elif position > 0:
            if row['low'] <= stop_loss:
                capital += position * stop_loss
                trades.append({'type': 'STOP_LOSS', 'price': stop_loss, 'time': idx})
                position = 0
            elif row['high'] >= take_profit:
                capital += position * take_profit
                trades.append({'type': 'TAKE_PROFIT', 'price': take_profit, 'time': idx})
                position = 0
                
        elif position < 0:
            if row['high'] >= stop_loss:
                capital += abs(position) * stop_loss
                trades.append({'type': 'STOP_LOSS', 'price': stop_loss, 'time': idx})
                position = 0
            elif row['low'] <= take_profit:
                capital += abs(position) * take_profit
                trades.append({'type': 'TAKE_PROFIT', 'price': take_profit, 'time': idx})
                position = 0
        
        # 计算权益
        if position != 0:
            equity = capital + position * row['close']
        else:
            equity = capital
        equity_curve.append(equity)
    
    # 平仓
    if position != 0:
        capital += position * df.iloc[-1]['close']
    
    # 计算统计
    win_trades = [t for t in trades if t['type'] in ['TAKE_PROFIT']]
    loss_trades = [t for t in trades if t['type'] in ['STOP_LOSS']]
    
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'total_trades': len([t for t in trades if t['type'] in ['BUY', 'SELL']]),
        'win_trades': len(win_trades),
        'loss_trades': len(loss_trades),
        'win_rate': len(win_trades) / max(1, len(win_trades) + len(loss_trades)) * 100,
        'equity_curve': equity_curve,
        'trades': trades
    }


# ==================== 图表 ====================

def create_chart(df: pd.DataFrame, show_signals: bool = True) -> go.Figure:
    """创建K线图"""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15], subplot_titles=('K线图', '成交量', 'MACD', 'RSI'))
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='K线',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    if 'sar' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sar'], mode='markers',
            marker=dict(size=3, color='blue'), name='SAR'), row=1, col=1)
    
    if 'support_level' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['support_level'], mode='lines',
            line=dict(color='green', dash='dash'), name='支撑'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['resistance_level'], mode='lines',
            line=dict(color='red', dash='dash'), name='阻力'), row=1, col=1)
    
    if 'ema50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema50'], mode='lines',
            line=dict(color='orange', width=1), name='EMA50'), row=1, col=1)
    
    if show_signals:
        long_signals = df[df['long_signal'] == True]
        if len(long_signals) > 0:
            fig.add_trace(go.Scatter(x=long_signals.index, y=long_signals['low'] * 0.998,
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='做多'), row=1, col=1)
        
        short_signals = df[df['short_signal'] == True]
        if len(short_signals) > 0:
            fig.add_trace(go.Scatter(x=short_signals.index, y=short_signals['high'] * 1.002,
                mode='markers', marker=dict(symbol='triangle-down', size=12, color='purple'), name='做空'), row=1, col=1)
    
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量', opacity=0.7), row=2, col=1)
    
    if 'macd_line' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_line'], mode='lines',
            line=dict(color='blue'), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['signal_line'], mode='lines',
            line=dict(color='orange'), name='Signal'), row=3, col=1)
        hist_colors = ['#26a69a' if h >= 0 else '#ef5350' for h in df['hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['hist'], marker_color=hist_colors, name='Hist', opacity=0.7), row=3, col=1)
    
    if 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], mode='lines',
            line=dict(color='purple'), name='RSI'), row=4, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='red', row=4, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', row=4, col=1)
    
    fig.update_layout(height=800, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis_rangeslider_visible=False, template='plotly_dark')
    return fig


# ==================== 主应用 ====================

def main():
    st.set_page_config(page_title="ETH 5分钟四维共振策略", page_icon="📊", layout="wide")
    
    st.title("📊 ETH 5分钟四维共振策略")
    st.markdown("**六种进场场景 | 实时数据 | 成交量+MACD+SAR+支撑阻力**")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 参数设置")
        
        # 数据源
        st.subheader("📡 数据源")
        data_source = st.radio("选择数据源", ["实时数据 (币安)", "模拟数据"])
        auto_refresh = st.checkbox("自动刷新 (60秒)", value=False) if data_source == "实时数据 (币安)" else False
        
        st.markdown("---")
        
        # 成交量参数
        st.subheader("📊 成交量参数")
        vol_lookback = st.number_input("均量周期", value=5, min_value=1, max_value=20)
        vol_shrink = st.number_input("缩量阈值", value=0.5, step=0.1, min_value=0.1, max_value=1.0)
        vol_expand = st.number_input("放量阈值", value=1.3, step=0.1, min_value=1.0, max_value=3.0)
        
        # MACD参数
        st.subheader("📈 MACD参数")
        macd_fast = st.number_input("MACD快线", value=12, min_value=5, max_value=30)
        macd_slow = st.number_input("MACD慢线", value=26, min_value=10, max_value=50)
        
        # 止损止盈
        st.subheader("💰 止损止盈")
        sl_mult = st.number_input("止损ATR倍数", value=1.5, step=0.1, min_value=0.5, max_value=3.0)
        tp_mult = st.number_input("止盈ATR倍数", value=3.0, step=0.5, min_value=1.0, max_value=5.0)
        
        # 过滤器
        st.subheader("🔍 过滤器")
        use_ema = st.checkbox("EMA趋势过滤", value=False)
        use_vol = st.checkbox("波动率过滤", value=False)
        use_rsi = st.checkbox("RSI背离过滤", value=False)
        
        st.markdown("---")
        
        # 按钮
        run_btn = st.button("🚀 运行分析", type="primary")
        
        if auto_refresh:
            st.info("🔄 自动刷新已开启")
            time.sleep(60)
            st.rerun()
    
    # 运行分析
    if run_btn or 'df' in st.session_state:
        with st.spinner("加载数据并计算中..."):
            # 获取数据
            if data_source == "实时数据 (币安)":
                df = fetch_binance_klines("ETHUSDT", "5m", 500)
                if df is None:
                    df = generate_sample_data(500)
                    data_source = "模拟数据 (备用)"
            else:
                df = generate_sample_data(500)
            
            # 配置策略
            config = {
                'vol_lookback': vol_lookback,
                'vol_shrink_ratio': vol_shrink,
                'vol_expand_ratio': vol_expand,
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'sl_atr_mult': sl_mult,
                'tp_atr_mult': tp_mult,
                'use_ema_filter': use_ema,
                'use_volatility_filter': use_vol,
                'use_rsi_filter': use_rsi,
            }
            
            strategy = FourDimStrategy(config)
            df = strategy.generate_signals(df)
            st.session_state.df = df
            
            # 回测
            backtest = run_backtest(df)
        
        # 数据信息
        st.markdown(f"**数据源:** {data_source} | **最新价格:** ${df['close'].iloc[-1]:,.2f} | **时间:** {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}") 
        
        # 信号统计
        long_count = int(df['long_signal'].sum())
        short_count = int(df['short_signal'].sum())
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🟢 做多信号", long_count)
        col2.metric("🔴 做空信号", short_count)
        col3.metric("📊 总信号", long_count + short_count)
        col4.metric("📈 K线数", len(df))
        col5.metric("💰 总收益", f"{backtest['total_return']:.1f}%")
        
        st.markdown("---")
        
        # 最新信号
        st.subheader("🔔 最新信号")
        signal = strategy.get_latest_signal(df)
        
        if signal:
            emoji = "🟢" if signal.direction == 'long' else "🔴"
            bg = '#1b5e20' if signal.direction == 'long' else '#b71c1c'
            st.markdown(f"""
            <div style="background-color: {bg}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{emoji} {signal.signal_type.value}</h2>
                <p style="color: #ccc; margin: 10px 0 0 0;">{signal.reason}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("入场价", f"${signal.price:,.2f}")
            col2.metric("止损", f"${signal.stop_loss:,.2f}")
            col3.metric("止盈", f"${signal.take_profit:,.2f}")
            rr = abs(signal.take_profit - signal.price) / max(0.01, abs(signal.price - signal.stop_loss))
            col4.metric("盈亏比", f"1:{rr:.1f}")
        else:
            last = df.iloc[-1]
            st.info(f"当前无交易信号 | 支撑: ${last['support_level']:,.2f} | 阻力: ${last['resistance_level']:,.2f} | ATR: ${last['atr']:.2f}")
        
        # K线图
        st.markdown("---")
        st.subheader("📈 价格走势与信号")
        fig = create_chart(df)
        st.plotly_chart(fig, width='stretch')
        
        # 回测统计
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 回测统计")
            st.metric("初始资金", f"${backtest['initial_capital']:,.0f}")
            st.metric("最终资金", f"${backtest['final_capital']:,.0f}")
            st.metric("胜率", f"{backtest['win_rate']:.1f}%")
            st.metric("交易次数", backtest['total_trades'])
        
        with col2:
            st.subheader("📊 四维指标状态")
            last = df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("成交量", "放量" if last['is_expand'] else ("缩量" if last['is_shrink'] else "正常"))
            c2.metric("MACD", "金叉" if last['hist_cross_up'] else ("死叉" if last['hist_cross_down'] else "运行中"))
            c3.metric("SAR", "绿三角" if last['is_green_triangle'] else ("紫三角" if last['is_purple_triangle'] else "跟随"))
            c4.metric("RSI", f"{last['rsi']:.1f}")
        
        # 信号历史
        with st.expander("📋 信号历史"):
            signals = strategy.get_all_signals(df)
            if signals:
                signal_df = pd.DataFrame([{
                    '时间': s.timestamp.strftime('%Y-%m-%d %H:%M'),
                    '类型': s.signal_type.value,
                    '方向': '做多' if s.direction == 'long' else '做空',
                    '价格': f"${s.price:,.2f}",
                    '止损': f"${s.stop_loss:,.2f}",
                    '止盈': f"${s.take_profit:,.2f}",
                    '原因': s.reason[:30] + '...'
                } for s in signals[-20:]])
                st.dataframe(signal_df, use_container_width=True)
            else:
                st.info("暂无历史信号")
    
    else:
        st.info("👈 在侧边栏配置参数后点击 '运行分析' 开始")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📖 策略说明", expanded=True):
                st.markdown("""
                ### 六种进场场景
                
                | 类型 | 描述 |
                |------|------|
                | 🟢 标准多头 | 放量起涨，突破阴线，MACD翻红，绿三角现 |
                | 🔴 标准空头 | 放量下跌，跌破阳线，MACD翻绿，紫三角现 |
                | 🟢 诱空陷阱 | 放量暴跌后收回，假跌真买 |
                | 🔴 诱多陷阱 | 放量暴涨后回落，假涨真空 |
                | 🟢 平台突破 | 缩量横盘后放量突破 |
                | 🔴 平台跌破 | 缩量横盘后放量跌破 |
                
                **四维指标：** 成交量 + MACD + SAR + 支撑阻力
                """)
        
        with col2:
            with st.expander("🎯 参数优化建议", expanded=True):
                st.markdown("""
                ### 增加信号频率
                - 降低 **缩量阈值** (0.4-0.5)
                - 降低 **放量阈值** (1.2-1.3)
                - 关闭过滤器 (EMA/波动率/RSI)
                
                ### 提高信号质量
                - 提高 **缩量阈值** (0.6-0.7)
                - 提高 **放量阈值** (1.5-2.0)
                - 开启过滤器
                
                ### 止损止盈
                - 短线: 止损1.0 ATR, 止盈2.0 ATR
                - 中线: 止损1.5 ATR, 止盈3.0 ATR
                - 长线: 止损2.0 ATR, 止盈4.0 ATR
                """)


if __name__ == "__main__":
    main()
