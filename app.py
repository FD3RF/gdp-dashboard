"""
ETH 5分钟四维共振策略 - 简化版
实时数据 + 成交量+MACD+SAR+支撑阻力
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import time
import urllib.request
import json


# ==================== 实时数据获取 ====================

@st.cache_data(ttl=60)
def fetch_binance_klines(symbol: str = "ETHUSDT", interval: str = "5m", limit: int = 500) -> pd.DataFrame:
    """从币安API获取实时K线数据"""
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
    """生成模拟数据"""
    if seed is not None:
        np.random.seed(seed)
    
    price = 2000.0
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='5min')
    opens, highs, lows, closes, volumes = [], [], [], [], []
    
    trend = np.cumsum(np.random.randn(n_bars) * 0.001)
    
    for i in range(n_bars):
        base_change = trend[i] + np.random.normal(0, 0.003)
        open_price = price
        close_price = price * (1 + base_change)
        vol = abs(np.random.normal(0.003, 0.002))
        high_price = max(open_price, close_price) * (1 + vol)
        low_price = min(open_price, close_price) * (1 - vol)
        
        if np.random.random() < 0.05:
            volume = 1000 * np.random.uniform(2.5, 4.0)
        elif np.random.random() < 0.15:
            volume = 1000 * np.random.uniform(0.3, 0.5)
        else:
            volume = 1000 * np.random.uniform(0.8, 1.5)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)
        price = close_price
    
    return pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes}, index=dates)


# ==================== 策略核心 ====================

class SignalType(Enum):
    LONG = "做多"
    SHORT = "做空"
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


class FourDimStrategy:
    """四维共振策略 - 成交量 + MACD + SAR + 支撑阻力"""
    
    def __init__(self, config=None):
        self.config = config or {
            'vol_period': 5,           # 均量周期
            'vol_shrink': 0.60,        # 缩量阈值
            'vol_expand': 1.50,        # 放量阈值
            'macd_fast': 12,           # MACD快线
            'macd_slow': 26,           # MACD慢线
            'macd_signal': 9,          # MACD信号线
            'atr_period': 14,          # ATR周期
            'stop_atr': 1.5,           # 止损ATR倍数
            'take_atr': 3.0,           # 止盈ATR倍数
            'support_lookback': 20,    # 支撑阻力回溯周期
            'sar_start': 0.02,         # SAR起始步长
            'sar_max': 0.2,            # SAR最大值
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()
        cfg = self.config
        
        # 成交量均线
        df['vol_ma'] = df['volume'].rolling(window=cfg['vol_period']).mean()
        df['is_expand'] = df['volume'] > df['vol_ma'] * cfg['vol_expand']
        df['is_shrink'] = df['volume'] < df['vol_ma'] * cfg['vol_shrink']
        
        # MACD
        ema_fast = df['close'].ewm(span=cfg['macd_fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=cfg['macd_slow'], adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow
        df['signal_line'] = df['macd_line'].ewm(span=cfg['macd_signal'], adjust=False).mean()
        df['hist'] = df['macd_line'] - df['signal_line']
        df['macd_cross_up'] = (df['macd_line'] > df['signal_line']) & (df['macd_line'].shift(1) <= df['signal_line'].shift(1))
        df['macd_cross_down'] = (df['macd_line'] < df['signal_line']) & (df['macd_line'].shift(1) >= df['signal_line'].shift(1))
        
        # SAR (抛物线转向)
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
                    ep, af = high[i], min(af + 0.02, cfg['sar_max'])
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                if high[i] > sar[i]:
                    is_long, sar[i], ep, af = True, ep, high[i], cfg['sar_start']
                elif low[i] < ep:
                    ep, af = low[i], min(af + 0.02, cfg['sar_max'])
        
        df['sar'] = sar
        df['above_sar'] = close > sar
        df['below_sar'] = close < sar
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=cfg['atr_period']).mean()
        
        # 支撑阻力
        df['resistance'] = df['high'].rolling(window=cfg['support_lookback']).max()
        df['support'] = df['low'].rolling(window=cfg['support_lookback']).min()
        
        # 接近阈值
        df['touch_threshold'] = df['atr'] * 0.3
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = self.calculate_indicators(df)
        cfg = self.config
        
        df['long_signal'] = False
        df['short_signal'] = False
        df['signal_type'] = SignalType.NONE.value
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # 多头条件：
        # 1. 放量 (volume > vol_ma * vol_expand)
        # 2. MACD > 信号线 (金叉或柱线为正)
        # 3. 价格在SAR之上 (SAR位于K线下方)
        # 4. 价格接近支撑位 (close < support + touch_threshold)
        long_condition = (
            df['is_expand'] & 
            (df['macd_line'] > df['signal_line']) & 
            df['above_sar'] & 
            (df['close'] < df['support'] + df['touch_threshold'])
        )
        
        # 空头条件：
        # 1. 放量下跌
        # 2. MACD < 信号线
        # 3. 价格在SAR之下
        # 4. 价格接近阻力位 (close > resistance - touch_threshold)
        short_condition = (
            df['is_expand'] & 
            (df['macd_line'] < df['signal_line']) & 
            df['below_sar'] & 
            (df['close'] > df['resistance'] - df['touch_threshold'])
        )
        
        df['long_signal'] = long_condition
        df['short_signal'] = short_condition
        
        # 标记信号类型
        df.loc[long_condition, 'signal_type'] = SignalType.LONG.value
        df.loc[long_condition, 'signal_reason'] = "放量+MACD金叉+SAR之上+接近支撑 → 做多"
        df.loc[short_condition, 'signal_type'] = SignalType.SHORT.value
        df.loc[short_condition, 'signal_reason'] = "放量+MACD死叉+SAR之下+接近阻力 → 做空"
        
        # 止损止盈
        df.loc[df['long_signal'], 'stop_loss'] = df.loc[df['long_signal'], 'close'] - df.loc[df['long_signal'], 'atr'] * cfg['stop_atr']
        df.loc[df['long_signal'], 'take_profit'] = df.loc[df['long_signal'], 'close'] + df.loc[df['long_signal'], 'atr'] * cfg['take_atr']
        df.loc[df['short_signal'], 'stop_loss'] = df.loc[df['short_signal'], 'close'] + df.loc[df['short_signal'], 'atr'] * cfg['stop_atr']
        df.loc[df['short_signal'], 'take_profit'] = df.loc[df['short_signal'], 'close'] - df.loc[df['short_signal'], 'atr'] * cfg['take_atr']
        
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
                timestamp=idx
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
                timestamp=df.index[-1]
            )
        return None


# ==================== 回测引擎 ====================

def run_backtest(df: pd.DataFrame, initial_capital: float = 10000, position_pct: float = 1.0) -> dict:
    """回测引擎"""
    capital = initial_capital
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trades = []
    equity_curve = [initial_capital]
    
    for i, (idx, row) in enumerate(df.iterrows()):
        # 开多仓
        if row['long_signal'] and position == 0:
            position = capital * position_pct / row['close']
            entry_price = row['close']
            stop_loss = row['stop_loss']
            take_profit = row['take_profit']
            trades.append({'type': 'BUY', 'price': entry_price, 'time': idx, 'reason': row['signal_reason']})
        
        # 开空仓
        elif row['short_signal'] and position == 0:
            position = -capital * position_pct / row['close']
            entry_price = row['close']
            stop_loss = row['stop_loss']
            take_profit = row['take_profit']
            trades.append({'type': 'SELL', 'price': entry_price, 'time': idx, 'reason': row['signal_reason']})
        
        # 多头止损止盈
        elif position > 0:
            if row['low'] <= stop_loss:
                pnl = (stop_loss - entry_price) * position
                capital += position * stop_loss
                trades.append({'type': 'STOP_LOSS', 'price': stop_loss, 'time': idx, 'pnl': pnl})
                position = 0
            elif row['high'] >= take_profit:
                pnl = (take_profit - entry_price) * position
                capital += position * take_profit
                trades.append({'type': 'TAKE_PROFIT', 'price': take_profit, 'time': idx, 'pnl': pnl})
                position = 0
        
        # 空头止损止盈
        elif position < 0:
            if row['high'] >= stop_loss:
                pnl = (entry_price - stop_loss) * abs(position)
                capital += abs(position) * stop_loss
                trades.append({'type': 'STOP_LOSS', 'price': stop_loss, 'time': idx, 'pnl': pnl})
                position = 0
            elif row['low'] <= take_profit:
                pnl = (entry_price - take_profit) * abs(position)
                capital += abs(position) * take_profit
                trades.append({'type': 'TAKE_PROFIT', 'price': take_profit, 'time': idx, 'pnl': pnl})
                position = 0
        
        # 记录权益
        if position > 0:
            equity = capital + position * row['close']
        elif position < 0:
            equity = capital + abs(position) * (2 * entry_price - row['close'])
        else:
            equity = capital
        equity_curve.append(equity)
    
    # 平仓
    if position > 0:
        capital += position * df.iloc[-1]['close']
    elif position < 0:
        capital += abs(position) * df.iloc[-1]['close']
    
    # 统计
    win_trades = [t for t in trades if t.get('type') == 'TAKE_PROFIT']
    loss_trades = [t for t in trades if t.get('type') == 'STOP_LOSS']
    entry_trades = [t for t in trades if t.get('type') in ['BUY', 'SELL']]
    
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'total_trades': len(entry_trades),
        'win_trades': len(win_trades),
        'loss_trades': len(loss_trades),
        'win_rate': len(win_trades) / max(1, len(win_trades) + len(loss_trades)) * 100,
        'equity_curve': equity_curve,
        'trades': trades
    }


# ==================== 图表 ====================

def create_chart(df: pd.DataFrame) -> go.Figure:
    """创建K线图"""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15], subplot_titles=('K线图 + SAR', '成交量', 'MACD', 'ATR'))
    
    # K线
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='K线',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    # SAR
    fig.add_trace(go.Scatter(x=df.index, y=df['sar'], mode='markers',
        marker=dict(size=4, color='orange', symbol='x'), name='SAR'), row=1, col=1)
    
    # 支撑阻力
    fig.add_trace(go.Scatter(x=df.index, y=df['support'], mode='lines',
        line=dict(color='green', dash='dot'), name='支撑'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['resistance'], mode='lines',
        line=dict(color='red', dash='dot'), name='阻力'), row=1, col=1)
    
    # 信号标记
    long_signals = df[df['long_signal'] == True]
    if len(long_signals) > 0:
        fig.add_trace(go.Scatter(x=long_signals.index, y=long_signals['low'] * 0.998,
            mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'),
            name='做多信号'), row=1, col=1)
    
    short_signals = df[df['short_signal'] == True]
    if len(short_signals) > 0:
        fig.add_trace(go.Scatter(x=short_signals.index, y=short_signals['high'] * 1.002,
            mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'),
            name='做空信号'), row=1, col=1)
    
    # 成交量
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量', opacity=0.7), row=2, col=1)
    
    # 均量线
    fig.add_trace(go.Scatter(x=df.index, y=df['vol_ma'] * cfg.get('vol_expand', 1.5), 
        mode='lines', line=dict(color='yellow', dash='dash'), name='放量线'), row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_line'], mode='lines',
        line=dict(color='blue'), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['signal_line'], mode='lines',
        line=dict(color='orange'), name='Signal'), row=3, col=1)
    hist_colors = ['#26a69a' if h >= 0 else '#ef5350' for h in df['hist'].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df['hist'], marker_color=hist_colors, name='Hist', opacity=0.7), row=3, col=1)
    
    # ATR
    fig.add_trace(go.Scatter(x=df.index, y=df['atr'], mode='lines',
        line=dict(color='purple'), name='ATR'), row=4, col=1)
    
    fig.update_layout(height=800, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis_rangeslider_visible=False, template='plotly_dark')
    return fig


# ==================== 主应用 ====================

cfg = {}

def main():
    global cfg
    st.set_page_config(page_title="ETH 5分钟四维共振策略", page_icon="📊", layout="wide")
    
    st.title("📊 ETH 5分钟四维共振策略")
    st.markdown("**放量 + MACD金叉/死叉 + SAR转向 + 接近支撑/阻力**")
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
        vol_period = st.number_input("均量周期", value=5, min_value=1, max_value=20)
        vol_shrink = st.number_input("缩量阈值", value=0.60, step=0.05)
        vol_expand = st.number_input("放量阈值", value=1.50, step=0.05)
        
        # MACD参数
        st.subheader("📈 MACD参数")
        macd_fast = st.number_input("MACD快线", value=12, min_value=5, max_value=30)
        macd_slow = st.number_input("MACD慢线", value=26, min_value=10, max_value=50)
        macd_signal = st.number_input("MACD信号线", value=9, min_value=3, max_value=20)
        
        # ATR止损止盈
        st.subheader("💰 止损止盈")
        atr_period = st.number_input("ATR周期", value=14, min_value=5, max_value=30)
        stop_atr = st.number_input("止损ATR倍数", value=1.5, step=0.1)
        take_atr = st.number_input("止盈ATR倍数", value=3.0, step=0.5)
        
        st.markdown("---")
        
        # 按钮
        run_btn = st.button("🚀 运行分析", type="primary")
        
        if auto_refresh:
            st.info("🔄 自动刷新已开启")
            time.sleep(60)
            st.rerun()
    
    # 构建配置
    cfg = {
        'vol_period': vol_period,
        'vol_shrink': vol_shrink,
        'vol_expand': vol_expand,
        'macd_fast': macd_fast,
        'macd_slow': macd_slow,
        'macd_signal': macd_signal,
        'atr_period': atr_period,
        'stop_atr': stop_atr,
        'take_atr': take_atr,
        'vol_expand': vol_expand,  # 用于图表
    }
    
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
            
            # 策略计算
            strategy = FourDimStrategy(cfg)
            df = strategy.generate_signals(df)
            st.session_state.df = df
            
            # 回测
            backtest = run_backtest(df)
        
        # 数据信息
        last = df.iloc[-1]
        st.markdown(f"""
        **数据源:** {data_source} | 
        **最新价格:** ${last['close']:,.2f} | 
        **时间:** {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')} |
        **ATR:** ${last['atr']:.2f}
        """)
        
        # 四维指标状态
        st.subheader("📍 四维指标状态")
        col1, col2, col3, col4 = st.columns(4)
        
        vol_status = "🟢 放量" if last['is_expand'] else ("🔴 缩量" if last['is_shrink'] else "⚪ 正常")
        macd_status = "🟢 金叉" if last['macd_line'] > last['signal_line'] else "🔴 死叉"
        sar_status = "🟢 SAR之上" if last['above_sar'] else "🔴 SAR之下"
        sr_status = f"支撑 ${last['support']:.0f} | 阻力 ${last['resistance']:.0f}"
        
        col1.metric("成交量", vol_status)
        col2.metric("MACD", macd_status)
        col3.metric("SAR", sar_status)
        col4.metric("支撑/阻力", sr_status[:20])
        
        # 信号统计
        st.markdown("---")
        long_count = int(df['long_signal'].sum())
        short_count = int(df['short_signal'].sum())
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🟢 做多信号", long_count)
        col2.metric("🔴 做空信号", short_count)
        col3.metric("📊 总信号", long_count + short_count)
        col4.metric("📈 K线数", len(df))
        col5.metric("💰 总收益", f"{backtest['total_return']:.2f}%")
        
        # 最新信号
        st.markdown("---")
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
            st.info(f"当前无交易信号 | 支撑: ${last['support']:,.2f} | 阻力: ${last['resistance']:,.2f}")
        
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
            st.metric("总收益", f"{backtest['total_return']:.2f}%")
            st.metric("胜率", f"{backtest['win_rate']:.1f}%")
            st.metric("交易次数", backtest['total_trades'])
        
        with col2:
            st.subheader("📋 信号历史")
            signals = strategy.get_all_signals(df)
            if signals:
                signal_df = pd.DataFrame([{
                    '时间': s.timestamp.strftime('%Y-%m-%d %H:%M'),
                    '类型': s.signal_type.value,
                    '价格': f"${s.price:,.2f}",
                    '止损': f"${s.stop_loss:,.2f}",
                    '止盈': f"${s.take_profit:,.2f}"
                } for s in signals[-10:]])
                st.dataframe(signal_df, use_container_width=True, hide_index=True)
            else:
                st.info("暂无历史信号")
    
    else:
        st.info("👈 在侧边栏配置参数后点击 '运行分析' 开始")
        
        with st.expander("📖 策略说明", expanded=True):
            st.markdown("""
            ### 四维共振策略
            
            **多头条件（四个条件同时满足）：**
            1. 🟢 **放量** - 成交量 > 均量 × 放量阈值
            2. 🟢 **MACD金叉** - MACD线 > 信号线
            3. 🟢 **SAR之上** - 价格在抛物线SAR上方
            4. 🟢 **接近支撑** - 价格接近支撑位
            
            **空头条件（四个条件同时满足）：**
            1. 🔴 **放量** - 成交量 > 均量 × 放量阈值
            2. 🔴 **MACD死叉** - MACD线 < 信号线
            3. 🔴 **SAR之下** - 价格在抛物线SAR下方
            4. 🔴 **接近阻力** - 价格接近阻力位
            
            **止损止盈：**
            - 止损 = 入场价 ± ATR × 止损倍数
            - 止盈 = 入场价 ± ATR × 止盈倍数
            """)


if __name__ == "__main__":
    main()
