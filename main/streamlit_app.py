"""
ETH 5分钟四维共振策略 - Streamlit 入口
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from analysis.four_dim_strategy import FourDimStrategy, FourDimConfig, SignalType


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
    
    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes
    }, index=dates)


def create_chart(df: pd.DataFrame) -> go.Figure:
    """创建K线图"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=('K线图', '成交量', 'MACD', 'RSI')
    )
    
    # K线
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name='K线',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1
    )
    
    # SAR
    if 'sar' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sar'], mode='markers',
                      marker=dict(size=3, color='blue'), name='SAR'),
            row=1, col=1
        )
    
    # 支撑阻力
    if 'support_level' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['support_level'], mode='lines',
                      line=dict(color='green', dash='dash'), name='支撑'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['resistance_level'], mode='lines',
                      line=dict(color='red', dash='dash'), name='阻力'),
            row=1, col=1
        )
    
    # EMA50
    if 'ema50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ema50'], mode='lines',
                      line=dict(color='orange', width=1), name='EMA50'),
            row=1, col=1
        )
    
    # 信号
    long_signals = df[df['long_signal'] == True]
    if len(long_signals) > 0:
        fig.add_trace(
            go.Scatter(x=long_signals.index, y=long_signals['low'] * 0.998,
                      mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'),
                      name='做多'), row=1, col=1
        )
    
    short_signals = df[df['short_signal'] == True]
    if len(short_signals) > 0:
        fig.add_trace(
            go.Scatter(x=short_signals.index, y=short_signals['high'] * 1.002,
                      mode='markers', marker=dict(symbol='triangle-down', size=12, color='purple'),
                      name='做空'), row=1, col=1
        )
    
    # 成交量
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='成交量', opacity=0.7),
        row=2, col=1
    )
    
    # MACD
    if 'macd_line' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_line'], mode='lines',
                      line=dict(color='blue'), name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['signal_line'], mode='lines',
                      line=dict(color='orange'), name='Signal'),
            row=3, col=1
        )
        hist_colors = ['#26a69a' if h >= 0 else '#ef5350' for h in df['hist'].fillna(0)]
        fig.add_trace(
            go.Bar(x=df.index, y=df['hist'], marker_color=hist_colors, name='Hist', opacity=0.7),
            row=3, col=1
        )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], mode='lines',
                      line=dict(color='purple'), name='RSI'),
            row=4, col=1
        )
        fig.add_hline(y=70, line_dash='dash', line_color='red', row=4, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', row=4, col=1)
    
    fig.update_layout(
        height=800, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis_rangeslider_visible=False, template='plotly_dark'
    )
    
    return fig


def main():
    st.set_page_config(page_title="ETH 5分钟四维共振策略", page_icon="📊", layout="wide")
    
    st.title("📊 ETH 5分钟四维共振策略")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 参数设置")
        
        st.subheader("成交量")
        vol_lookback = st.number_input("均量周期", value=5)
        vol_shrink = st.number_input("缩量阈值", value=0.6, step=0.1)
        vol_expand = st.number_input("放量阈值", value=1.5, step=0.1)
        
        st.subheader("MACD")
        macd_fast = st.number_input("快线", value=12)
        macd_slow = st.number_input("慢线", value=26)
        
        st.subheader("止损止盈")
        sl_mult = st.number_input("止损ATR倍数", value=1.5, step=0.1)
        tp_mult = st.number_input("止盈ATR倍数", value=3.0, step=0.5)
        
        st.subheader("过滤器")
        use_ema = st.checkbox("EMA趋势过滤", value=True)
        use_vol = st.checkbox("波动率过滤", value=True)
        use_rsi = st.checkbox("RSI背离过滤", value=True)
        
        st.markdown("---")
        run_btn = st.button("🚀 运行回测", type="primary")
    
    # 主内容
    if run_btn or 'df' in st.session_state:
        with st.spinner("计算中..."):
            # 生成数据
            df = generate_sample_data(500)
            
            # 配置
            config = FourDimConfig()
            config.VOL_LOOKBACK = vol_lookback
            config.VOL_SHRINK_RATIO = vol_shrink
            config.VOL_EXPAND_RATIO = vol_expand
            config.MACD_FAST = macd_fast
            config.MACD_SLOW = macd_slow
            config.SL_ATR_MULT = sl_mult
            config.TP_ATR_MULT = tp_mult
            config.USE_EMA_FILTER = use_ema
            config.USE_VOLATILITY_FILTER = use_vol
            config.USE_RSI_FILTER = use_rsi
            
            # 策略
            strategy = FourDimStrategy(config)
            df = strategy.generate_signals(df)
            
            st.session_state.df = df
            st.session_state.config = config
        
        # 统计
        long_count = df['long_signal'].sum()
        short_count = df['short_signal'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("做多信号", int(long_count))
        with col2:
            st.metric("做空信号", int(short_count))
        with col3:
            st.metric("总信号数", int(long_count + short_count))
        with col4:
            st.metric("K线数", len(df))
        
        st.markdown("---")
        
        # 最新信号
        st.subheader("🔔 最新信号")
        signal = strategy.get_latest_signal(df)
        
        if signal:
            direction_emoji = "🟢" if signal.direction == 'long' else "🔴"
            bg_color = '#1b5e20' if signal.direction == 'long' else '#b71c1c'
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{direction_emoji} {signal.signal_type.value}</h2>
                <p style="color: #ccc; font-size: 16px; margin: 10px 0 0 0;">{signal.reason}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("入场价", f"${signal.price:,.2f}")
            with col2:
                st.metric("止损", f"${signal.stop_loss:,.2f}")
            with col3:
                st.metric("止盈", f"${signal.take_profit:,.2f}")
            with col4:
                rr = abs(signal.take_profit - signal.price) / abs(signal.price - signal.stop_loss)
                st.metric("盈亏比", f"1:{rr:.1f}")
        else:
            st.info("当前无交易信号")
        
        st.markdown("---")
        
        # 图表
        st.subheader("📈 价格走势与信号")
        fig = create_chart(df)
        st.plotly_chart(fig, width='stretch')
        
        # 四维指标状态
        with st.expander("📊 四维指标详情"):
            if signal:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("成交量状态", signal.volume_signal)
                with col2:
                    st.metric("MACD状态", signal.macd_signal)
                with col3:
                    st.metric("SAR状态", signal.sar_signal)
                with col4:
                    st.metric("支撑/阻力", signal.support_resistance)
    
    else:
        st.info("👈 在侧边栏配置参数后点击 '运行回测' 开始")
        
        with st.expander("📖 策略说明"):
            st.markdown("""
            ### 六种进场场景
            
            | 类型 | 描述 |
            |------|------|
            | 🟢 标准多头 | 放量起涨，突破阴线，MACD翻红，绿三角现 |
            | 🔴 标准空头 | 放量下跌，跌破阳线，MACD翻绿，紫三角现 |
            | 🟢 诱空陷阱做多 | 放量暴跌后收回，假跌真买 |
            | 🔴 诱多陷阱做空 | 放量暴涨后回落，假涨真空 |
            | 🟢 平台突破做多 | 缩量横盘后放量突破 |
            | 🔴 平台跌破做空 | 缩量横盘后放量跌破 |
            
            **四维指标：**
            - 📊 成交量（缩量/放量/极端放量）
            - 📈 MACD 趋势指标
            - 🔺 SAR 抛物线转向
            - 📍 动态支撑阻力位
            """)


if __name__ == "__main__":
    main()
