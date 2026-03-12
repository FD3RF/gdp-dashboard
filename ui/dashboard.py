# ui/dashboard.py
"""
ETH AI Agent 实时仪表盘
======================
Streamlit 实时交易仪表盘，显示真实 ETH 行情和 AI 决策
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# 导入主程序
from main import run_system, SystemState
from analysis.decision_maker import Signal


# 页面配置
st.set_page_config(
    page_title="ETH AI Agent (Live)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """创建K线图"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('价格 & 均线', 'MACD', 'RSI')
    )
    
    # K线
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # 均线
    for ma, color in [('ma5', '#ff9800'), ('ma20', '#2196f3'), ('ma60', '#9c27b0')]:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df[ma], name=ma.upper(), line=dict(color=color, width=1)),
                row=1, col=1
            )
    
    # 布林带
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB上轨', line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB下轨', line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
    
    # MACD
    if 'macd' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='#2196f3')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='#ff9800')), row=2, col=1)
        colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df['macd_histogram']]
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_histogram'], name='Histogram', marker_color=colors), row=2, col=1)
    
    # RSI
    if 'rsi14' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi14'], name='RSI', line=dict(color='#9c27b0')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig


def main():
    """主函数"""
    # 标题
    st.title("🧠 ETH/USDT 5分钟 AI 盯盘终端 (真实数据版)")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 设置")
        symbol = st.selectbox("交易对", ["ETH/USDT", "BTC/USDT", "SOL/USDT"])
        auto_refresh = st.checkbox("自动刷新", value=True)
        refresh_interval = st.slider("刷新间隔(秒)", 5, 60, 10)
        
        st.divider()
        st.header("📊 风险配置")
        max_risk = st.slider("单笔最大风险(%)", 1, 10, 2)
        max_position = st.slider("最大仓位(%)", 10, 50, 30)
    
    # 获取数据
    with st.spinner("🔄 正在获取实时数据..."):
        state = run_system(symbol)
    
    if state is None:
        st.error("❌ 获取数据失败，请检查网络连接")
        st.stop()
    
    # 第一行：核心数据
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta = f"{state.indicators.get('price_change_5m', 0) * 100:.2f}%"
        st.metric(
            "当前价格",
            f"${state.price:,.2f}",
            delta=delta,
            delta_color="normal"
        )
    
    with col2:
        signal_emoji = "🟢" if state.signal == Signal.LONG else "🔴" if state.signal == Signal.SHORT else "⚪"
        st.metric("AI 决策", f"{signal_emoji} {state.signal.value}")
    
    with col3:
        st.metric("信心指数", f"{state.probabilities['confidence']:.1f}")
    
    with col4:
        risk_color = "🟢" if state.risk.score < 25 else "🟡" if state.risk.score < 50 else "🔴"
        st.metric("风险等级", f"{risk_color} {state.risk.level.value}")
    
    st.divider()
    
    # 第二行：概率分布
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📊 AI 概率分布")
        
        probs = state.probabilities
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        
        with prob_col1:
            st.progress(probs['long'] / 100)
            st.caption(f"🟢 做多: {probs['long']:.1f}%")
        
        with prob_col2:
            st.progress(probs['short'] / 100)
            st.caption(f"🔴 做空: {probs['short']:.1f}%")
        
        with prob_col3:
            st.progress(probs['hold'] / 100)
            st.caption(f"⚪ 观望: {probs['hold']:.1f}%")
        
        # 信号理由
        if probs.get('signals', {}).get('reasons'):
            st.subheader("📝 分析理由")
            for reason in probs['signals']['reasons'][:5]:
                st.write(f"• {reason}")
    
    with col_right:
        st.subheader("📈 技术指标")
        
        indicators = state.indicators
        ind_col1, ind_col2 = st.columns(2)
        
        with ind_col1:
            st.metric("RSI(14)", f"{indicators.get('rsi14', 50):.1f}")
            st.metric("MA5", f"${indicators.get('ma5', 0):,.0f}")
            st.metric("MA20", f"${indicators.get('ma20', 0):,.0f}")
        
        with ind_col2:
            st.metric("MACD", f"{indicators.get('macd', 0):.2f}")
            st.metric("BB位置", f"{indicators.get('bb_position', 0.5):.2f}")
            st.metric("量比", f"{indicators.get('volume_ratio', 1):.2f}")
    
    st.divider()
    
    # 第三行：交易计划
    if state.plan:
        st.subheader("📋 交易计划")
        
        plan_col1, plan_col2, plan_col3, plan_col4 = st.columns(4)
        
        with plan_col1:
            st.metric("入场价格", f"${state.plan.entry:,.2f}")
        
        with plan_col2:
            st.metric("止损价格", f"${state.plan.stop_loss:,.2f}")
        
        with plan_col3:
            st.metric("目标价格", f"${state.plan.take_profit_1:,.2f}")
        
        with plan_col4:
            st.metric("风险收益比", f"{state.plan.risk_reward:.2f}")
        
        st.info(f"💡 建议仓位: {state.plan.position_size:.1f}% | 杠杆: {state.plan.leverage}x")
    else:
        st.warning("当前建议观望，无交易计划")
    
    # 风险提示
    if state.risk.warnings:
        st.subheader("⚠️ 风险提示")
        for warning in state.risk.warnings:
            st.warning(warning)
        
        if state.risk.suggestions:
            st.subheader("💡 操作建议")
            for suggestion in state.risk.suggestions:
                st.info(f"• {suggestion}")
    
    # K线图 (需要重新获取完整数据)
    st.divider()
    st.subheader("📈 K线图")
    
    # 获取K线数据
    from data.market_stream import get_realtime_eth_data
    from data.kline_builder import calculate_indicators
    
    df, _ = get_realtime_eth_data(symbol, limit=100)
    if df is not None:
        df = calculate_indicators(df)
        fig = create_price_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # 底部信息
    st.caption(f"最后更新: {state.timestamp} | 数据来源: Binance")


if __name__ == "__main__":
    main()
