"""
风险管理面板组件
=================

风险管理监控面板，全中文界面。
"""

import sys
from pathlib import Path

# 添加 dashboard 目录到路径
DASHBOARD_DIR = Path(__file__).parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.charts import RiskGauge


# 中文标签
LABELS = {
    'risk_management': '⚠️ 风险管理',
    'no_risk_data': '暂无风险数据',
    'metrics': '风险指标',
    'positions': '持仓风险',
    'current_drawdown': '当前回撤',
    'limit': '限制',
    'total_exposure': '总敞口',
    'leverage': '杠杆倍数',
    'max': '最大',
    'open_positions': '持仓数量',
    'daily_pnl': '日内盈亏',
    'margin_used': '已用保证金',
    'free_margin': '可用保证金',
    'margin_level': '保证金率',
    'position_risk': '持仓风险分析',
    'no_positions': '暂无持仓',
    'symbol': '交易对',
    'size': '数量',
    'value': '价值',
    'risk_pct': '风险比例',
    'stop_loss': '止损价',
    'unrealized_pnl': '未实现盈亏',
    'exposure_distribution': '持仓敞口分布',
    'risk_alerts': '⚠️ 风险警报',
    'no_alerts': '无活动风险警报',
    'account_summary': '💰 账户概览',
    'total_balance': '总余额',
    'available_balance': '可用余额',
    'realized_pnl': '已实现盈亏',
    'risk_limits': '🛡️ 风险限制',
    'max_position_size': '最大持仓比例',
    'max_drawdown': '最大回撤',
    'max_leverage': '最大杠杆',
    'daily_loss': '日内亏损',
    'current': '当前',
    'status': '状态',
    'overall_risk': '整体风险',
}


class RiskPanel:
    """
    风险管理监控面板
    """
    
    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
    
    def render(self, risk_data: Dict[str, Any]) -> None:
        """
        渲染风险面板
        
        Args:
            risk_data: 风险指标和状态
        """
        st.header(LABELS['risk_management'])
        
        if not risk_data:
            st.warning(LABELS['no_risk_data'])
            return
        
        # 整体风险等级
        self._render_risk_level(risk_data)
        
        # 风险指标
        col1, col2 = st.tabs([LABELS['metrics'], LABELS['positions']])
        
        with col1:
            self._render_metrics(risk_data)
        
        with col2:
            self._render_positions(risk_data)
        
        # 警报
        if 'alerts' in risk_data:
            self._render_alerts(risk_data['alerts'])
    
    def _render_risk_level(self, risk_data: Dict[str, Any]) -> None:
        """渲染整体风险等级仪表"""
        risk_score = risk_data.get('risk_score', 50)
        overall_level = risk_data.get('overall_risk_level', '未知')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            fig = RiskGauge.create(
                value=risk_score,
                title=f"{LABELS['overall_risk']}: {overall_level}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_metrics(self, risk_data: Dict[str, Any]) -> None:
        """渲染风险指标"""
        st.subheader(LABELS['metrics'])
        
        # 主要指标行
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            drawdown = risk_data.get('drawdown', 0)
            max_dd = risk_data.get('max_drawdown_limit', 0.15)
            st.metric(
                LABELS['current_drawdown'],
                f"{drawdown * 100:.2f}%",
                delta=f"{LABELS['limit']}: {max_dd * 100:.1f}%",
                delta_color="inverse"
            )
        
        with col2:
            exposure = risk_data.get('total_exposure', 0)
            st.metric(
                LABELS['total_exposure'],
                f"{exposure * 100:.1f}%"
            )
        
        with col3:
            leverage = risk_data.get('leverage', 1.0)
            max_lev = risk_data.get('max_leverage', 3.0)
            st.metric(
                LABELS['leverage'],
                f"{leverage:.2f}x",
                delta=f"{LABELS['max']}: {max_lev:.1f}x",
                delta_color="inverse" if leverage > max_lev * 0.8 else "normal"
            )
        
        with col4:
            position_count = risk_data.get('position_count', 0)
            st.metric(
                LABELS['open_positions'],
                position_count
            )
        
        # 次要指标
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            daily_pnl = risk_data.get('daily_pnl', 0)
            st.metric(
                LABELS['daily_pnl'],
                f"${daily_pnl:,.2f}"
            )
        
        with col2:
            margin_used = risk_data.get('margin_used', 0)
            st.metric(
                LABELS['margin_used'],
                f"${margin_used:,.2f}"
            )
        
        with col3:
            free_margin = risk_data.get('free_margin', 0)
            st.metric(
                LABELS['free_margin'],
                f"${free_margin:,.2f}"
            )
        
        with col4:
            margin_level = risk_data.get('margin_level', 100)
            st.metric(
                LABELS['margin_level'],
                f"{margin_level:.1f}%"
            )
    
    def _render_positions(self, risk_data: Dict[str, Any]) -> None:
        """渲染持仓风险"""
        st.subheader(LABELS['position_risk'])
        
        positions = risk_data.get('positions', {})
        
        if not positions:
            st.info(LABELS['no_positions'])
            return
        
        # 创建持仓风险表格
        data = []
        for symbol, pos in positions.items():
            data.append({
                LABELS['symbol']: symbol,
                LABELS['size']: pos.get('quantity', 0),
                LABELS['value']: f"${pos.get('value', 0):,.2f}",
                LABELS['risk_pct']: f"{pos.get('risk_pct', 0) * 100:.2f}%",
                LABELS['stop_loss']: f"${pos.get('stop_loss', 0):,.2f}",
                LABELS['unrealized_pnl']: f"${pos.get('unrealized_pnl', 0):,.2f}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # 持仓敞口饼图
        self._render_exposure_chart(positions)
    
    def _render_exposure_chart(self, positions: Dict[str, Any]) -> None:
        """渲染持仓敞口饼图"""
        if not positions:
            return
        
        symbols = list(positions.keys())
        values = [p.get('value', 0) for p in positions.values()]
        
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=symbols,
                    values=values,
                    hole=0.4,
                    textinfo='label+percent',
                    textposition='outside',
                    textfont=dict(size=12)
                )
            ]
        )
        
        fig.update_layout(
            title=LABELS['exposure_distribution'],
            template='plotly_dark',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """渲染风险警报"""
        st.subheader(LABELS['risk_alerts'])
        
        if not alerts:
            st.success(LABELS['no_alerts'])
            return
        
        for alert in alerts[-10:]:  # 显示最近10条警报
            level = alert.get('level', 'info')
            
            if level == 'critical':
                st.error(f"🔴 {alert.get('message', '未知警报')}")
            elif level == 'warning':
                st.warning(f"🟡 {alert.get('message', '未知警报')}")
            else:
                st.info(f"ℹ️ {alert.get('message', '未知警报')}")


def render_account_summary(account: Dict[str, Any]) -> None:
    """
    渲染账户概要面板
    
    Args:
        account: 账户数据
    """
    st.subheader(LABELS['account_summary'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            LABELS['total_balance'],
            f"${account.get('total_balance', 0):,.2f}"
        )
    
    with col2:
        st.metric(
            LABELS['available_balance'],
            f"${account.get('available', 0):,.2f}"
        )
    
    with col3:
        st.metric(
            LABELS['unrealized_pnl'],
            f"${account.get('unrealized_pnl', 0):,.2f}"
        )
    
    with col4:
        st.metric(
            LABELS['realized_pnl'],
            f"${account.get('realized_pnl', 0):,.2f}"
        )


def render_risk_limits(limits: Dict[str, Any]) -> None:
    """
    渲染风险限制状态
    
    Args:
        limits: 风险限制配置
    """
    st.subheader(LABELS['risk_limits'])
    
    data = [
        {
            '限制项': LABELS['max_position_size'],
            LABELS['current']: f"{limits.get('current_position_size', 0) * 100:.2f}%",
            LABELS['limit']: f"{limits.get('max_position_size', 0) * 100:.1f}%",
            LABELS['status']: '✅' if limits.get('position_size_ok', True) else '❌'
        },
        {
            '限制项': LABELS['max_drawdown'],
            LABELS['current']: f"{limits.get('current_drawdown', 0) * 100:.2f}%",
            LABELS['limit']: f"{limits.get('max_drawdown', 0) * 100:.1f}%",
            LABELS['status']: '✅' if limits.get('drawdown_ok', True) else '❌'
        },
        {
            '限制项': LABELS['max_leverage'],
            LABELS['current']: f"{limits.get('current_leverage', 1):.2f}x",
            LABELS['limit']: f"{limits.get('max_leverage', 3):.1f}x",
            LABELS['status']: '✅' if limits.get('leverage_ok', True) else '❌'
        },
        {
            '限制项': LABELS['daily_loss'],
            LABELS['current']: f"{limits.get('current_daily_loss', 0) * 100:.2f}%",
            LABELS['limit']: f"{limits.get('max_daily_loss', 0) * 100:.1f}%",
            LABELS['status']: '✅' if limits.get('daily_loss_ok', True) else '❌'
        }
    ]
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_var_analysis(risk_data: Dict[str, Any]) -> None:
    """
    渲染风险价值分析
    
    Args:
        risk_data: 风险指标
    """
    st.subheader("📊 风险价值 (VaR)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var_95 = risk_data.get('var_95', 0)
        st.metric(
            "VaR (95%置信)",
            f"${var_95:,.2f}",
            help="95%置信水平下的预期损失"
        )
    
    with col2:
        var_99 = risk_data.get('var_99', 0)
        st.metric(
            "VaR (99%置信)",
            f"${var_99:,.2f}",
            help="99%置信水平下的预期损失"
        )
    
    with col3:
        cvar = risk_data.get('cvar', 0)
        st.metric(
            "CVaR / ES",
            f"${cvar:,.2f}",
            help="条件风险价值 / 预期损失"
        )


# 导出
__all__ = ['RiskPanel', 'render_account_summary', 'render_risk_limits', 'render_var_analysis']
