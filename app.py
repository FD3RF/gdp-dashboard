"""
ETH 5分钟四维共振策略 - 独立单文件版本
无需外部依赖，可直接部署
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional


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
    volume_signal: str = ""
    macd_signal: str = ""
    sar_signal: str = ""
    support_resistance: str = ""


class FourDimStrategy:
    """四维共振策略 - 成交量 + MACD + SAR + 支撑阻力"""
    
    def __init__(self, config=None):
        self.config = config or {
            'vol_lookback': 5,
            'vol_shrink_ratio': 0.6,
            'vol_expand_ratio': 1.5,
            'vol_panic_ratio': 3.0,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'sar_start': 0.02,
            'sar_inc': 0.02,
            'sar_max': 0.2,
            'support_lookback': 20,
            'range_percent': 0.3,
            'consolidation_bars': 5,
            'sl_atr_mult': 1.5,
            'tp_atr_mult': 3.0,
            'use_ema_filter': True,
            'use_volatility_filter': True,
            'use_rsi_filter': True,
            'ema50_len': 50,
            'rsi_len': 14,
        }
    
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
        near_support = (df['close'] >= df['support_level']) & (df['close'] < df['support_level'] * 1.01)
        shrink_at_support = near_support & df['is_shrink'] & (df['low'] > df['support_level'])
        breakout_bull = df['is_expand'] & (df['close'] > df['last_bear_high'].shift(1))
        standard_bull = (shrink_at_support.shift(1) | df['is_consolidation'].shift(1)) & breakout_bull & df['hist_cross_up'] & df['is_green_triangle']
        
        # 2. 标准空头
        near_resistance = (df['close'] <= df['resistance_level']) & (df['close'] > df['resistance_level'] * 0.99)
        shrink_at_resistance = near_resistance & df['is_shrink'] & (df['high'] < df['resistance_level'])
        breakdown_bear = df['is_expand'] & (df['close'] < df['last_bull_low'].shift(1))
        standard_bear = (shrink_at_resistance.shift(1) | df['is_consolidation'].shift(1)) & breakdown_bear & df['hist_cross_down'] & df['is_purple_triangle']
        
        # 3. 诱空陷阱做多
        panic_down = df['is_panic'] & (df['low'] < df['support_level']) & (df['close'] > df['support_level'])
        confirm_up = df['close'] > (df['high'].shift(1) * 0.5 + df['low'].shift(1) * 0.5)
        trap_bull = panic_down.shift(1) & confirm_up & df['is_green_triangle'] & (df['hist'].shift(1) < 0) & (df['hist'] > df['hist'].shift(1))
        if cfg['use_rsi_filter']:
            trap_bull = trap_bull & df['bull_div']
        
        # 4. 诱多陷阱做空
        panic_up = df['is_panic'] & (df['high'] > df['resistance_level']) & (df['close'] < df['resistance_level'])
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
                volume_signal="放量" if last['is_expand'] else ("缩量" if last['is_shrink'] else "正常"),
                macd_signal="金叉" if last['hist_cross_up'] else ("死叉" if last['hist_cross_down'] else "运行中"),
                sar_signal="绿三角" if last['is_green_triangle'] else ("紫三角" if last['is_purple_triangle'] else "跟随"),
                support_resistance=f"支撑{last['support_level']:.0f}/阻力{last['resistance_level']:.0f}"
            )
        return None


# ==================== 数据生成 ====================

def generate_sample_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成模拟数据"""
    np.random.seed(seed)
    price = 2000.0
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
    opens, highs, lows, closes, volumes = [], [], [], [], []
    
    for _ in range(n_bars):
        change = np.random.normal(0, 0.002)
        open_price = price
        close_price = price * (1 + change)
        vol = abs(np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + vol)
        low_price = min(open_price, close_price) * (1 - vol)
        volume = 1000 * (1 + np.random.exponential(0.5))
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)
        price = close_price
    
    return pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes}, index=dates)


# ==================== 图表 ====================

def create_chart(df: pd.DataFrame) -> go.Figure:
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
    st.markdown("**六种进场场景 | 成交量+MACD+SAR+支撑阻力**")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ 参数设置")
        vol_lookback = st.number_input("均量周期", value=5)
        vol_shrink = st.number_input("缩量阈值", value=0.6, step=0.1)
        vol_expand = st.number_input("放量阈值", value=1.5, step=0.1)
        macd_fast = st.number_input("MACD快线", value=12)
        macd_slow = st.number_input("MACD慢线", value=26)
        sl_mult = st.number_input("止损ATR倍数", value=1.5, step=0.1)
        tp_mult = st.number_input("止盈ATR倍数", value=3.0, step=0.5)
        use_ema = st.checkbox("EMA趋势过滤", value=True)
        use_vol = st.checkbox("波动率过滤", value=True)
        use_rsi = st.checkbox("RSI背离过滤", value=True)
        st.markdown("---")
        run_btn = st.button("🚀 运行回测", type="primary")
    
    if run_btn or 'df' in st.session_state:
        with st.spinner("计算中..."):
            df = generate_sample_data(500)
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
        
        long_count = int(df['long_signal'].sum())
        short_count = int(df['short_signal'].sum())
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("做多信号", long_count)
        col2.metric("做空信号", short_count)
        col3.metric("总信号", long_count + short_count)
        col4.metric("K线数", len(df))
        
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
            rr = abs(signal.take_profit - signal.price) / abs(signal.price - signal.stop_loss)
            col4.metric("盈亏比", f"1:{rr:.1f}")
        else:
            st.info("当前无交易信号")
        
        st.markdown("---")
        st.subheader("📈 价格走势与信号")
        fig = create_chart(df)
        st.plotly_chart(fig, width='stretch')
        
        with st.expander("📊 四维指标详情"):
            if signal:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("成交量", signal.volume_signal)
                col2.metric("MACD", signal.macd_signal)
                col3.metric("SAR", signal.sar_signal)
                col4.metric("支撑/阻力", signal.support_resistance)
    
    else:
        st.info("👈 在侧边栏配置参数后点击 '运行回测' 开始")
        
        with st.expander("📖 策略说明"):
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


if __name__ == "__main__":
    main()
