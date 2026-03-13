# ui/dashboard.py
"""
Layer 12: 可视化盯盘层 v5.0
===========================
信息分层 + 三栏布局 + 颜色规范
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

# 导入模块
from main import run_system
from analysis.decision_maker import Signal

# 页面配置
st.set_page_config(
    page_title="ETH AI Agent v5.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"  # 默认收起侧边栏
)


def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """创建K线图（简化版）"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
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
    for ma, color in [('ma20', '#2196f3')]:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df[ma], name='MA20', line=dict(color=color, width=1)),
                row=1, col=1
            )
    
    # RSI
    if 'rsi14' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi14'], name='RSI', line=dict(color='#9c27b0')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=400, showlegend=False, xaxis_rangeslider_visible=False, template='plotly_dark',
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig


def render_final_decision(state):
    """
    第一层：最终决策（最显眼）
    """
    signal = state.signal
    confidence = state.probabilities.get('confidence', 0)
    reliability = state.reliability.get('score', 0)
    position = state.position_multiplier
    
    # 决定颜色和文字
    if signal == Signal.LONG:
        bg_color = "#1b5e20"
        decision_text = "🟢 做多"
    elif signal == Signal.SHORT:
        bg_color = "#b71c1c"
        decision_text = "🔴 做空"
    else:
        bg_color = "#424242"
        decision_text = "⚪ 观望"
    
    # 渲染大标题
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 48px;">{decision_text}</h1>
        <p style="color: #ccc; font-size: 16px; margin: 10px 0 0 0;">
            置信度 {confidence:.0f}% | 可靠度 {reliability}/100 | 仓位 {position*100:.0f}%
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 价格显示
    st.markdown(f"""
    <div style="text-align: center; padding: 10px 0;">
        <span style="font-size: 36px; font-weight: bold;">${state.price:,.2f}</span>
        <span style="color: #888; font-size: 14px;">  {state.exchange}</span>
    </div>
    """, unsafe_allow_html=True)


def render_core_metrics(state):
    """
    第二层：核心依据（3个关键模块）
    """
    col1, col2, col3 = st.columns(3)
    
    # 市场状态
    with col1:
        regime = state.market_regime
        regime_name = regime.get('regime', 'neutral')
        regime_conf = regime.get('confidence', 0) * 100
        
        # 图标
        if 'trend_up' in regime_name:
            icon = "📈"
            text = "强势上涨"
            color = "#4caf50"
        elif 'trend_down' in regime_name:
            icon = "📉"
            text = "强势下跌"
            color = "#f44336"
        else:
            icon = "📊"
            text = "震荡"
            color = "#9e9e9e"
        
        st.markdown(f"""
        <div style="border: 1px solid #444; border-radius: 8px; padding: 15px;">
            <h3 style="margin: 0; color: {color};">{icon} {text}</h3>
            <p style="color: #888; font-size: 12px; margin: 5px 0;">市场状态 | 置信度 {regime_conf:.0f}%</p>
            <p style="font-size: 14px; margin: 5px 0;">趋势强度: {regime.get('trend_strength', 0):.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 订单流
    with col2:
        flow = state.order_flow
        cvd = flow.get('cvd', {})
        cvd_val = cvd.get('value', 0)
        cvd_dir = cvd.get('direction', 'neutral')
        
        if cvd_val > 0:
            icon = "🟢"
            text = "买方主导"
            color = "#4caf50"
        elif cvd_val < 0:
            icon = "🔴"
            text = "卖方主导"
            color = "#f44336"
        else:
            icon = "⚪"
            text = "均衡"
            color = "#9e9e9e"
        
        is_real = flow.get('is_real_data', False)
        data_source = "真实" if is_real else "模拟"
        
        st.markdown(f"""
        <div style="border: 1px solid #444; border-radius: 8px; padding: 15px;">
            <h3 style="margin: 0; color: {color};">{icon} {text}</h3>
            <p style="color: #888; font-size: 12px; margin: 5px 0;">订单流 | [{data_source}]</p>
            <p style="font-size: 14px; margin: 5px 0;">CVD: {cvd_val:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 流动性
    with col3:
        liq = state.liquidity
        supports = liq.get('support_zones', [])
        resistances = liq.get('resistance_zones', [])
        
        support_price = supports[0][0] if supports else 0
        support_strength = supports[0][1] if supports else 0
        resistance_price = resistances[0][0] if resistances else 0
        resistance_strength = resistances[0][1] if resistances else 0
        
        st.markdown(f"""
        <div style="border: 1px solid #444; border-radius: 8px; padding: 15px;">
            <h3 style="margin: 0; color: #2196f3;">📊 流动性</h3>
            <p style="color: #888; font-size: 12px; margin: 5px 0;">支撑/阻力</p>
            <p style="font-size: 14px; margin: 5px 0;">
                支撑 ${support_price:,.0f} ({support_strength:.1f}x) | 
                阻力 ${resistance_price:,.0f} ({resistance_strength:.1f}x)
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_auxiliary_info(state):
    """
    第三层：辅助验证（折叠/可展开）
    """
    with st.expander("📋 辅助验证", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        # 巨鲸
        with col1:
            whale = state.whale_alerts
            flow = whale.get('flow_summary', {})
            net_flow = flow.get('net_flow_eth', 0)
            sentiment = flow.get('sentiment', '中性')
            
            icon = "🟢" if "看涨" in sentiment else "🔴" if "看跌" in sentiment else "⚪"
            st.markdown(f"**巨鲸监控**")
            st.caption(f"{icon} {sentiment}")
            st.caption(f"净流动: {net_flow:.0f} ETH")
        
        # 资金费率
        with col2:
            funding = state.funding_extreme
            summary = funding.get('summary', {})
            avg_rate = summary.get('avg_rate', 0) * 100
            crowd = summary.get('crowding_direction', '中性')
            
            icon = "⚠️" if "拥挤" in crowd else "✅"
            st.markdown(f"**资金费率**")
            st.caption(f"平均: {avg_rate:.4f}%")
            st.caption(f"{icon} {crowd}")
        
        # 清算
        with col3:
            liq = state.liquidation
            heatmap = liq.get('heatmap', {})
            long_liq = heatmap.get('total_long_liq', 0)
            short_liq = heatmap.get('total_short_liq', 0)
            
            st.markdown(f"**清算监控**")
            st.caption(f"多头: ${long_liq/1e6:.2f}M")
            st.caption(f"空头: ${short_liq/1e6:.2f}M")


def render_block_reason(state):
    """
    拦截原因（如果有）
    """
    if not state.is_trading_allowed or state.signal == Signal.HOLD:
        ud = state.unified_decision
        hf = ud.get('hard_filter', {})
        block_reason = hf.get('block_reason', '')
        
        if block_reason:
            st.error(f"🚫 拦截原因: {block_reason}")
        
        # 或者显示四层决策原因
        fl = ud.get('four_layer', {})
        entry_reason = fl.get('entry_reason', '')
        if entry_reason and not block_reason:
            st.info(f"📋 观望原因: {entry_reason}")


def render_raw_data(state):
    """
    第四层：原始数据（完全折叠/调试用）
    """
    with st.expander("🔧 原始数据", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption(f"Hurst指数: {state.hurst:.3f} ({state.hurst_state})")
            st.caption(f"数据质量: {state.data_quality_score*100:.0f}%")
            st.caption(f"信号一致性: {state.unified_signal.get('quality_metrics', {}).get('consistency', 0)*100:.0f}%")
        
        with col2:
            votes = state.unified_signal.get('votes', {})
            st.caption(f"模块投票: ↑{votes.get('long', 0)} ↓{votes.get('short', 0)} ↔{votes.get('neutral', 0)}")
            
            sync = state.feature_sync_status
            stale = len(sync.get('stale_features', []))
            st.caption(f"过期特征: {stale} 个")


def main():
    """主函数 - v5.0 信息分层版"""
    
    # 侧边栏（简化）
    with st.sidebar:
        symbol = st.selectbox("交易对", ["ETH/USDT", "BTC/USDT"])
        data_source = st.radio("数据源", ["自动", "模拟"])
        use_simulated = data_source == "模拟"
        if st.button("🔄 刷新"):
            st.rerun()
    
    # 获取数据
    with st.spinner("分析中..."):
        state = run_system(symbol, use_simulated=use_simulated)
    
    if state is None:
        st.error("获取数据失败")
        st.stop()
    
    # 数据源状态
    if state.is_simulated:
        st.info("ℹ️ 模拟数据模式")
    
    # === 第一层：最终决策 ===
    render_final_decision(state)
    st.divider()
    
    # === 第二层：核心依据 ===
    render_core_metrics(state)
    st.divider()
    
    # === 拦截原因 ===
    render_block_reason(state)
    
    # === 第三层：辅助验证（折叠）===
    render_auxiliary_info(state)
    
    # === K线图 ===
    with st.expander("📈 K线图", expanded=False):
        from data.market_stream import get_realtime_eth_data
        from data.kline_builder import calculate_indicators
        
        df, _ = get_realtime_eth_data(symbol, limit=100, use_simulated=use_simulated)
        if df is not None:
            df = calculate_indicators(df)
            fig = create_price_chart(df)
            st.plotly_chart(fig, use_container_width=True)
    
    # === 第四层：原始数据（折叠）===
    render_raw_data(state)
    
    # 底部
    st.caption(f"v5.0 | {state.timestamp}")


if __name__ == "__main__":
    main()
