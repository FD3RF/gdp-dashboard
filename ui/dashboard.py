# ui/dashboard.py
"""
Layer 12: 可视化盯盘层
=====================
全息指挥官视图
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
    page_title="ETH AI Agent Pro",
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
    
    fig.update_layout(height=500, showlegend=True, xaxis_rangeslider_visible=False, template='plotly_dark')
    return fig


def main():
    """主函数"""
    st.title("🧠 ETH/USDT 5分钟 AI 盯盘终端")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        symbol = st.selectbox("交易对", ["ETH/USDT", "BTC/USDT", "SOL/USDT"])
        data_source = st.radio("数据源", ["自动", "模拟数据"])
        use_simulated = data_source == "模拟数据"
        
        st.divider()
        st.header("📊 风险配置")
        st.slider("单笔最大风险(%)", 1, 10, 2)
        st.slider("最大仓位(%)", 10, 50, 30)
    
    # 获取数据
    with st.spinner("🔄 分析中..."):
        state = run_system(symbol, use_simulated=use_simulated)
    
    if state is None:
        st.error("❌ 获取数据失败")
        st.stop()
    
    # 数据源状态
    if state.is_simulated:
        st.warning("⚠️ 模拟数据模式 (真实API不可达)")
        st.caption("提示: 网络限制导致无法访问交易所API，使用模拟数据进行演示")
    else:
        st.success(f"✅ {state.exchange} 真实数据")
    
    # === 核心数据行 ===
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("当前价格", f"${state.price:,.2f}")
    with col2:
        emoji = "🟢" if state.signal == Signal.LONG else "🔴" if state.signal == Signal.SHORT else "⚪"
        st.metric("AI 决策", f"{emoji} {state.signal.value}")
    with col3:
        st.metric("信心指数", f"{state.probabilities['confidence']:.1f}")
    with col4:
        risk_emoji = "🟢" if state.risk.score < 25 else "🟡" if state.risk.score < 50 else "🔴"
        st.metric("风险等级", f"{risk_emoji} {state.risk.level.value if hasattr(state.risk, 'level') else 'N/A'}")
    
    st.divider()
    
    # === AI核心决策 + 流动性热图 ===
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 AI 概率分布")
        probs = state.probabilities
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.progress(probs['long'] / 100)
            st.caption(f"🟢 做多: {probs['long']:.1f}%")
        with c2:
            st.progress(probs['short'] / 100)
            st.caption(f"🔴 做空: {probs['short']:.1f}%")
        with c3:
            st.progress(probs['hold'] / 100)
            st.caption(f"⚪ 观望: {probs['hold']:.1f}%")
    
    with col_right:
        st.subheader("🗺️ 流动性热图 (M-41)")
        
        liquidity = state.liquidity
        for support in liquidity.get('support_zones', [])[:2]:
            st.write(f"🟢 支撑墙: ${support[0]:,.2f} (强度: {support[1]:.1f}x)")
        for resistance in liquidity.get('resistance_zones', [])[:2]:
            st.write(f"🔴 阻力墙: ${resistance[0]:,.2f} (强度: {resistance[1]:.1f}x)")
        
        for warning in liquidity.get('trap_warnings', [])[:2]:
            st.warning(warning.get('description', '陷阱预警')[:50])
    
    st.divider()
    
    # === Hurst + 资金费率 ===
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Hurst指数", f"{state.hurst:.3f}", state.hurst_state)
    with c2:
        funding = state.funding
        st.metric("资金费率", f"{funding.get('rate', 0)*100:.4f}%", funding.get('signal', ''))
    with c3:
        reliability = state.reliability
        st.metric("系统可靠度", f"{reliability.get('score', 0):.0f}分", reliability.get('level', ''))
    
    st.divider()
    
    # === 交易计划 ===
    if state.plan:
        st.subheader("📋 AI 交易计划")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("入场价格", f"${state.plan.entry:,.2f}")
        with c2:
            st.metric("止损价格", f"${state.plan.stop_loss:,.2f}")
        with c3:
            st.metric("目标价格", f"${state.plan.take_profit_1:,.2f}")
        with c4:
            st.metric("风险收益比", f"{state.plan.risk_reward:.2f}")
        st.info(f"💡 建议仓位: {state.plan.position_size:.1f}%")
    else:
        st.warning("⚠️ 当前建议观望，无交易计划")
    
    st.divider()
    
    # === 新增6大功能展示 ===
    st.subheader("🚀 高级分析 (6大升级)")
    
    # 第一行: 巨鲸监控 + 社交情绪
    col_whale, col_sentiment = st.columns(2)
    
    with col_whale:
        st.markdown("**🐋 链上巨鲸监控**")
        whale = state.whale_alerts
        flow = whale.get('flow_summary', {})
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("净流入", f"{flow.get('exchange_inflow_eth', 0):.0f} ETH")
        with c2:
            st.metric("净流出", f"{flow.get('exchange_outflow_eth', 0):.0f} ETH")
        with c3:
            sentiment_icon = "🟢" if flow.get('sentiment') == "看涨" else "🔴" if flow.get('sentiment') == "看跌" else "⚪"
            st.metric("情绪", f"{sentiment_icon} {flow.get('sentiment', '中性')}")
        
        # 显示最近警报
        for alert in whale.get('alerts', [])[:2]:
            direction = alert.get('direction', '')
            icon = "🔴" if 'in' in direction else "🟢" if 'out' in direction else "🐋"
            st.caption(f"{icon} {alert.get('details', '')[:40]}")
    
    with col_sentiment:
        st.markdown("**📱 社交情绪分析**")
        sentiment = state.social_sentiment
        
        score = sentiment.get('score', 50)
        progress_color = "green" if score > 60 else "red" if score < 40 else "gray"
        st.progress(score / 100)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("情绪分数", f"{score:.1f}")
        with c2:
            st.metric("趋势", sentiment.get('trend', 'stable'))
        with c3:
            st.metric("主导力量", sentiment.get('dominance', 'neutral'))
        
        if sentiment.get('is_extreme'):
            st.warning(f"⚠️ 检测到极端情绪: {sentiment.get('extreme_type', 'unknown')}")
    
    st.divider()
    
    # 第二行: 资金费率极值 + 风险矩阵
    col_funding, col_risk = st.columns(2)
    
    with col_funding:
        st.markdown("**📊 多交易所资金费率极值**")
        funding_ext = state.funding_extreme
        summary = funding_ext.get('summary', {})
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("平均费率", f"{summary.get('avg_rate', 0)*100:.4f}%")
        with c2:
            st.metric("平均Z-Score", f"{summary.get('avg_z_score', 0):.2f}")
        with c3:
            crowd = summary.get('crowding_direction', '中性')
            icon = "🔴" if "做多" in crowd else "🟢" if "做空" in crowd else "⚪"
            st.metric("拥挤方向", f"{icon} {crowd}")
        
        # 显示警报
        for alert in funding_ext.get('alerts', [])[:2]:
            severity = alert.get('severity', 'low')
            icon = "🚨" if severity == "high" else "⚠️" if severity == "medium" else "ℹ️"
            st.caption(f"{icon} {alert.get('description', '')[:40]}")
    
    with col_risk:
        st.markdown("**🎯 多维风险矩阵**")
        risk_mat = state.risk_matrix
        
        score = risk_mat.get('score', 50)
        level = risk_mat.get('level', 'moderate')
        direction = risk_mat.get('direction', 'neutral')
        
        # 风险仪表盘
        level_icons = {
            "very_low": "🟢 极低",
            "low": "🟢 低",
            "moderate": "🟡 中等",
            "high": "🔴 高",
            "very_high": "🔴 极高",
            "extreme": "🚨 极端"
        }
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("风险分数", f"{score:.1f}")
        with c2:
            st.metric("风险等级", level_icons.get(level, level))
        with c3:
            dir_icon = "📈" if direction == "bullish" else "📉" if direction == "bearish" else "➡️"
            st.metric("方向", f"{dir_icon} {direction}")
        
        # 显示警告
        for warning in risk_mat.get('warnings', [])[:2]:
            st.warning(warning[:50])
    
    st.divider()
    
    # 第三行: 进化状态
    st.markdown("**🧬 策略进化状态**")
    evolution = state.evolution_status
    perf = evolution.get('recent_performance', {})
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("总信号数", evolution.get('total_signals', 0))
    with c2:
        st.metric("胜率", f"{perf.get('win_rate', 0)*100:.1f}%")
    with c3:
        st.metric("盈亏比", f"{perf.get('profit_factor', 0):.2f}")
    with c4:
        avg_win = perf.get('avg_win_percent', 0)
        avg_loss = perf.get('avg_loss_percent', 0)
        st.metric("平均盈亏%", f"+{avg_win:.2f}% / -{avg_loss:.2f}%")
    with c5:
        weights = evolution.get('current_weights', {})
        top_feature = max(weights, key=weights.get) if weights else 'N/A'
        st.metric("最强特征", top_feature)
    
    st.divider()
    
    # === 市场状态识别 (核心新增) ===
    st.subheader("🎯 市场状态识别 (Regime Engine)")
    regime = state.market_regime
    
    if regime:
        regime_name = regime.get('regime', 'neutral')
        regime_icons = {
            "trend_up": "📈 强势上涨",
            "trend_down": "📉 强势下跌",
            "range": "↔️ 震荡区间",
            "volatile": "⚡ 高波动",
            "liquidation": "💥 清算事件",
            "accumulation": "🏦 吸筹阶段",
            "distribution": "📤 派发阶段",
            "panic": "😱 恐慌状态",
            "euphoria": "🚀 狂热状态",
            "neutral": "➡️ 中性",
        }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            icon = regime_icons.get(regime_name, regime_name)
            st.metric("市场状态", icon)
        with col2:
            st.metric("置信度", f"{regime.get('confidence', 0)*100:.0f}%")
        with col3:
            st.metric("趋势强度", f"{regime.get('trend_strength', 0):.2f}")
        with col4:
            st.metric("波动水平", f"{regime.get('volatility_level', 0):.2f}")
        
        # 策略建议
        recommended = regime.get('recommended_strategies', [])
        avoided = regime.get('avoided_strategies', [])
        
        if recommended:
            st.success(f"✅ 推荐策略: {', '.join(recommended)}")
        if avoided:
            st.info(f"⚠️ 避免策略: {', '.join(avoided)}")
        
        # 风险因素
        risks = regime.get('risk_factors', [])
        for risk in risks[:3]:
            st.warning(f"⚠️ {risk}")
    
    st.divider()
    
    # === 订单流分析 (核心新增) ===
    st.subheader("📊 订单流分析 (Order Flow)")
    flow = state.order_flow
    
    if flow:
        cvd = flow.get('cvd', {})
        delta = flow.get('delta', {})
        imbalance = flow.get('imbalance', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cvd_val = cvd.get('value', 0)
            cvd_icon = "📈" if cvd_val > 0 else "📉" if cvd_val < 0 else "➡️"
            st.metric("CVD", f"{cvd_icon} {cvd_val:.2f}")
        with col2:
            st.metric("Delta", f"{delta.get('value', 0):.2f}")
        with col3:
            st.metric("Delta%", f"{delta.get('pct', 0):.2f}%")
        with col4:
            imb_dir = imbalance.get('direction', 'neutral')
            imb_icon = "🟢" if imb_dir == "bullish" else "🔴" if imb_dir == "bearish" else "⚪"
            st.metric("失衡方向", f"{imb_icon} {imb_dir}")
        
        # CVD信号
        cvd_signal = cvd.get('signal', 'neutral')
        cvd_div = cvd.get('divergence', False)
        if cvd_signal == "bullish":
            st.success(f"✅ CVD信号: 看涨 {'(检测到背离)' if cvd_div else ''}")
        elif cvd_signal == "bearish":
            st.error(f"🔴 CVD信号: 看跌 {'(检测到背离)' if cvd_div else ''}")
        
        # 吸收检测
        absorption = flow.get('absorption', {})
        if absorption.get('detected'):
            st.info(f"🔍 检测到吸收行为: {absorption.get('direction')}")
    
    st.divider()
    
    # === 清算监控 (核心新增) ===
    st.subheader("💥 清算监控 (Liquidation)")
    liq = state.liquidation
    
    if liq:
        heatmap = liq.get('heatmap', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("清算级别", liq.get('total_levels', 0))
        with col2:
            st.metric("多头清算", f"${heatmap.get('total_long_liq', 0):,.0f}")
        with col3:
            st.metric("空头清算", f"${heatmap.get('total_short_liq', 0):,.0f}")
        with col4:
            imb_ratio = heatmap.get('imbalance_ratio', 1)
            imb_dir = heatmap.get('imbalance_direction', 'balanced')
            imb_icon = "🔴" if imb_dir == "long_heavy" else "🟢" if imb_dir == "short_heavy" else "⚪"
            st.metric("失衡比", f"{imb_icon} {imb_ratio:.2f}")
        
        # 接近的清算警报
        approaching = liq.get('approaching_alerts', [])
        if approaching:
            for alert in approaching[:3]:
                severity = alert.get('severity', 'low')
                if severity == 'critical':
                    st.error(f"🚨 {alert.get('details')}")
                elif severity == 'high':
                    st.warning(f"⚠️ {alert.get('details')}")
                else:
                    st.info(f"ℹ️ {alert.get('details')}")
        
        # 级联风险
        cascade = liq.get('cascade_risk')
        if cascade:
            st.error(f"💥 级联风险: {cascade}")
    
    st.divider()
    
    # === AI 决策解释 ===
    st.subheader("📝 AI 决策解释 (M-35/36)")
    explanation = state.explanation
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**主要来源**: {explanation.get('primary_source', '综合')}")
        st.write(f"**市场结构**: {explanation.get('market_structure', '分析中')}")
    with c2:
        st.write(f"**分析理由**: {explanation.get('reasoning', '')}")
    
    for warning in explanation.get('warnings', [])[:3]:
        st.warning(warning)
    
    st.divider()
    
    # === K线图 ===
    st.subheader("📈 K线图")
    from data.market_stream import get_realtime_eth_data
    from data.kline_builder import calculate_indicators
    
    df, _ = get_realtime_eth_data(symbol, limit=100, use_simulated=use_simulated)
    if df is not None:
        df = calculate_indicators(df)
        fig = create_price_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # 底部
    st.caption(f"最后更新: {state.timestamp} | 交易所: {state.exchange} | 12层架构 v3.2 + 机构级模块")


if __name__ == "__main__":
    main()
