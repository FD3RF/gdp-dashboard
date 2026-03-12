"""
Risk Panel Component
====================

Risk management monitoring panel.
"""

import sys
from pathlib import Path

# Add dashboard directory to path for component imports
DASHBOARD_DIR = Path(__file__).parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.charts import RiskGauge


class RiskPanel:
    """
    Risk management monitoring panel.
    """
    
    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
    
    def render(self, risk_data: Dict[str, Any]) -> None:
        """
        Render the risk panel.
        
        Args:
            risk_data: Risk metrics and status
        """
        st.header("⚠️ Risk Management")
        
        if not risk_data:
            st.warning("No risk data available")
            return
        
        # Overall risk level
        self._render_risk_level(risk_data)
        
        # Risk metrics
        col1, col2 = st.tabs(["Metrics", "Positions"])
        
        with col1:
            self._render_metrics(risk_data)
        
        with col2:
            self._render_positions(risk_data)
        
        # Alerts
        if 'alerts' in risk_data:
            self._render_alerts(risk_data['alerts'])
    
    def _render_risk_level(self, risk_data: Dict[str, Any]) -> None:
        """Render overall risk level gauge."""
        risk_score = risk_data.get('risk_score', 50)
        overall_level = risk_data.get('overall_risk_level', 'unknown')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            fig = RiskGauge.create(
                value=risk_score,
                title=f"Overall Risk: {overall_level.upper()}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_metrics(self, risk_data: Dict[str, Any]) -> None:
        """Render risk metrics."""
        st.subheader("Risk Metrics")
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            drawdown = risk_data.get('drawdown', 0)
            max_dd = risk_data.get('max_drawdown_limit', 0.15)
            st.metric(
                "Current Drawdown",
                f"{drawdown * 100:.2f}%",
                delta=f"Limit: {max_dd * 100:.1f}%",
                delta_color="inverse"
            )
        
        with col2:
            exposure = risk_data.get('total_exposure', 0)
            st.metric(
                "Total Exposure",
                f"{exposure * 100:.1f}%"
            )
        
        with col3:
            leverage = risk_data.get('leverage', 1.0)
            max_lev = risk_data.get('max_leverage', 3.0)
            st.metric(
                "Leverage",
                f"{leverage:.2f}x",
                delta=f"Max: {max_lev:.1f}x",
                delta_color="inverse" if leverage > max_lev * 0.8 else "normal"
            )
        
        with col4:
            position_count = risk_data.get('position_count', 0)
            st.metric(
                "Open Positions",
                position_count
            )
        
        # Secondary metrics
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            daily_pnl = risk_data.get('daily_pnl', 0)
            daily_limit = risk_data.get('daily_loss_limit', 0.05)
            st.metric(
                "Daily PnL",
                f"${daily_pnl:,.2f}"
            )
        
        with col2:
            margin_used = risk_data.get('margin_used', 0)
            st.metric(
                "Margin Used",
                f"${margin_used:,.2f}"
            )
        
        with col3:
            free_margin = risk_data.get('free_margin', 0)
            st.metric(
                "Free Margin",
                f"${free_margin:,.2f}"
            )
        
        with col4:
            margin_level = risk_data.get('margin_level', 100)
            st.metric(
                "Margin Level",
                f"{margin_level:.1f}%"
            )
    
    def _render_positions(self, risk_data: Dict[str, Any]) -> None:
        """Render position-level risk."""
        st.subheader("Position Risk")
        
        positions = risk_data.get('positions', {})
        
        if not positions:
            st.info("No open positions")
            return
        
        # Create position risk table
        data = []
        for symbol, pos in positions.items():
            data.append({
                'Symbol': symbol,
                'Size': pos.get('quantity', 0),
                'Value': f"${pos.get('value', 0):,.2f}",
                'Risk %': f"{pos.get('risk_pct', 0) * 100:.2f}%",
                'Stop Loss': f"${pos.get('stop_loss', 0):,.2f}",
                'Unrealized PnL': f"${pos.get('unrealized_pnl', 0):,.2f}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Position exposure pie chart
        self._render_exposure_chart(positions)
    
    def _render_exposure_chart(self, positions: Dict[str, Any]) -> None:
        """Render position exposure pie chart."""
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
                    textposition='outside'
                )
            ]
        )
        
        fig.update_layout(
            title="Position Exposure Distribution",
            template='plotly_dark',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Render risk alerts."""
        st.subheader("⚠️ Risk Alerts")
        
        if not alerts:
            st.success("No active risk alerts")
            return
        
        for alert in alerts[-10:]:  # Show last 10 alerts
            level = alert.get('level', 'info')
            
            if level == 'critical':
                st.error(f"🔴 {alert.get('message', 'Unknown alert')}")
            elif level == 'warning':
                st.warning(f"🟡 {alert.get('message', 'Unknown alert')}")
            else:
                st.info(f"ℹ️ {alert.get('message', 'Unknown alert')}")


def render_account_summary(account: Dict[str, Any]) -> None:
    """
    Render account summary panel.
    
    Args:
        account: Account data
    """
    st.subheader("💰 Account Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Balance",
            f"${account.get('total_balance', 0):,.2f}"
        )
    
    with col2:
        st.metric(
            "Available Balance",
            f"${account.get('available', 0):,.2f}"
        )
    
    with col3:
        st.metric(
            "Unrealized PnL",
            f"${account.get('unrealized_pnl', 0):,.2f}"
        )
    
    with col4:
        st.metric(
            "Realized PnL",
            f"${account.get('realized_pnl', 0):,.2f}"
        )


def render_risk_limits(limits: Dict[str, Any]) -> None:
    """
    Render risk limits status.
    
    Args:
        limits: Risk limits configuration
    """
    st.subheader("🛡️ Risk Limits")
    
    data = [
        {
            'Limit': 'Max Position Size',
            'Current': f"{limits.get('current_position_size', 0) * 100:.2f}%",
            'Limit': f"{limits.get('max_position_size', 0) * 100:.1f}%",
            'Status': '✅' if limits.get('position_size_ok', True) else '❌'
        },
        {
            'Limit': 'Max Drawdown',
            'Current': f"{limits.get('current_drawdown', 0) * 100:.2f}%",
            'Limit': f"{limits.get('max_drawdown', 0) * 100:.1f}%",
            'Status': '✅' if limits.get('drawdown_ok', True) else '❌'
        },
        {
            'Limit': 'Max Leverage',
            'Current': f"{limits.get('current_leverage', 1):.2f}x",
            'Limit': f"{limits.get('max_leverage', 3):.1f}x",
            'Status': '✅' if limits.get('leverage_ok', True) else '❌'
        },
        {
            'Limit': 'Daily Loss',
            'Current': f"{limits.get('current_daily_loss', 0) * 100:.2f}%",
            'Limit': f"{limits.get('max_daily_loss', 0) * 100:.1f}%",
            'Status': '✅' if limits.get('daily_loss_ok', True) else '❌'
        }
    ]
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_var_analysis(risk_data: Dict[str, Any]) -> None:
    """
    Render Value at Risk analysis.
    
    Args:
        risk_data: Risk metrics
    """
    st.subheader("📊 Value at Risk (VaR)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var_95 = risk_data.get('var_95', 0)
        st.metric(
            "VaR (95%)",
            f"${var_95:,.2f}",
            help="Expected loss at 95% confidence level"
        )
    
    with col2:
        var_99 = risk_data.get('var_99', 0)
        st.metric(
            "VaR (99%)",
            f"${var_99:,.2f}",
            help="Expected loss at 99% confidence level"
        )
    
    with col3:
        cvar = risk_data.get('cvar', 0)
        st.metric(
            "CVaR / ES",
            f"${cvar:,.2f}",
            help="Conditional VaR / Expected Shortfall"
        )
