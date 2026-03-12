"""
Dashboard Tables Module
=======================

Data table components using Streamlit.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime


class TradeTable:
    """Trade history table component."""
    
    @staticmethod
    def create(
        trades: List[Dict[str, Any]],
        title: str = "Trade History",
        limit: int = 50
    ) -> None:
        """
        Display trade history table.
        
        Args:
            trades: List of trade dictionaries
            title: Table title
            limit: Maximum rows to display
        """
        st.subheader(f"📋 {title}")
        
        if not trades:
            st.info("No trades recorded")
            return
        
        df = pd.DataFrame(trades[:limit])
        
        # Format columns
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Style the dataframe
        def color_side(val):
            if val == 'buy':
                return 'color: #4CAF50'
            elif val == 'sell':
                return 'color: #f44336'
            return ''
        
        styled_df = df.style.applymap(
            color_side,
            subset=['side'] if 'side' in df.columns else []
        )
        
        # Display
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # Summary stats
        if 'pnl' in df.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_pnl = df['pnl'].sum()
                st.metric(
                    "Total PnL",
                    f"${total_pnl:,.2f}",
                    delta=f"{total_pnl / len(df):,.2f} avg"
                )
            
            with col2:
                win_rate = (df['pnl'] > 0).sum() / len(df) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col3:
                total_trades = len(df)
                st.metric("Total Trades", total_trades)


class OrderTable:
    """Active orders table component."""
    
    @staticmethod
    def create(
        orders: List[Dict[str, Any]],
        title: str = "Active Orders"
    ) -> None:
        """
        Display active orders table.
        
        Args:
            orders: List of order dictionaries
            title: Table title
        """
        st.subheader(f"📊 {title}")
        
        if not orders:
            st.info("No active orders")
            return
        
        df = pd.DataFrame(orders)
        
        # Status color mapping
        status_colors = {
            'open': '#2196F3',
            'filled': '#4CAF50',
            'cancelled': '#9E9E9E',
            'rejected': '#f44336',
            'pending': '#FFC107'
        }
        
        def color_status(val):
            color = status_colors.get(val, '#9E9E9E')
            return f'color: {color}; font-weight: bold'
        
        styled_df = df.style.applymap(
            color_status,
            subset=['status'] if 'status' in df.columns else []
        )
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=300
        )
        
        # Order counts by status
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            
            cols = st.columns(len(status_counts))
            for i, (status, count) in enumerate(status_counts.items()):
                with cols[i]:
                    st.metric(status.capitalize(), count)


class PositionTable:
    """Current positions table component."""
    
    @staticmethod
    def create(
        positions: Dict[str, Dict[str, Any]],
        title: str = "Current Positions"
    ) -> None:
        """
        Display current positions table.
        
        Args:
            positions: Dictionary of positions
            title: Table title
        """
        st.subheader(f"📈 {title}")
        
        if not positions:
            st.info("No open positions")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'Symbol': symbol,
                'Quantity': pos.get('quantity', 0),
                'Entry Price': pos.get('entry_price', 0),
                'Current Price': pos.get('current_price', 0),
                'Unrealized PnL': pos.get('unrealized_pnl', 0),
                'PnL %': pos.get('pnl_pct', 0)
            }
            for symbol, pos in positions.items()
        ])
        
        # Format columns
        if 'PnL %' in df.columns:
            df['PnL %'] = df['PnL %'].apply(lambda x: f"{x:.2f}%")
        
        if 'Unrealized PnL' in df.columns:
            df['Unrealized PnL'] = df['Unrealized PnL'].apply(lambda x: f"${x:,.2f}")
        
        # Color code PnL
        def highlight_pnl(row):
            styles = [''] * len(row)
            pnl_idx = row.index.get_loc('Unrealized PnL') if 'Unrealized PnL' in row.index else -1
            if pnl_idx >= 0:
                pnl_val = row['Unrealized PnL']
                if pnl_val.startswith('$'):
                    pnl_num = float(pnl_val.replace('$', '').replace(',', ''))
                    if pnl_num > 0:
                        styles[pnl_idx] = 'color: #4CAF50; font-weight: bold'
                    elif pnl_num < 0:
                        styles[pnl_idx] = 'color: #f44336; font-weight: bold'
            return styles
        
        styled_df = df.style.apply(highlight_pnl, axis=1)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=300
        )


class MetricsTable:
    """Performance metrics table."""
    
    @staticmethod
    def create(
        metrics: Dict[str, Any],
        title: str = "Performance Metrics"
    ) -> None:
        """
        Display metrics table.
        
        Args:
            metrics: Dictionary of metrics
            title: Table title
        """
        st.subheader(f"📊 {title}")
        
        # Create columns for metrics display
        cols = st.columns(4)
        
        metric_items = list(metrics.items())
        
        for i, (key, value) in enumerate(metric_items[:8]):
            col_idx = i % 4
            
            with cols[col_idx]:
                # Format value
                if isinstance(value, float):
                    if 'ratio' in key.lower() or 'rate' in key.lower():
                        formatted = f"{value:.2f}"
                    elif 'pct' in key.lower() or 'drawdown' in key.lower():
                        formatted = f"{value * 100:.2f}%"
                    else:
                        formatted = f"{value:,.2f}"
                else:
                    formatted = str(value)
                
                st.metric(
                    key.replace('_', ' ').title(),
                    formatted
                )


class SignalTable:
    """Trading signals table."""
    
    @staticmethod
    def create(
        signals: List[Dict[str, Any]],
        title: str = "Recent Signals"
    ) -> None:
        """
        Display trading signals table.
        
        Args:
            signals: List of signal dictionaries
            title: Table title
        """
        st.subheader(f"⚡ {title}")
        
        if not signals:
            st.info("No signals generated")
            return
        
        df = pd.DataFrame(signals[:20])
        
        # Format columns
        display_cols = ['symbol', 'side', 'signal_type', 'strength', 'timestamp']
        df = df[[c for c in display_cols if c in df.columns]]
        
        # Style
        def color_signal(row):
            styles = []
            side = row.get('side', '')
            if side == 'buy':
                styles.append('color: #4CAF50')
            elif side == 'sell':
                styles.append('color: #f44336')
            return styles
        
        if 'side' in df.columns:
            styled_df = df.style.apply(color_signal, axis=1)
        else:
            styled_df = df
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=300
        )


class AgentStatusTable:
    """Agent status overview table."""
    
    @staticmethod
    def create(
        agents: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Display agent status table.
        
        Args:
            agents: Dictionary of agent statuses
        """
        st.subheader("🤖 Agent Status")
        
        if not agents:
            st.info("No agents registered")
            return
        
        data = []
        for name, status in agents.items():
            data.append({
                'Agent': name,
                'Status': status.get('status', 'unknown'),
                'Running': '✅' if status.get('running', False) else '❌',
                'Tasks': status.get('queued_tasks', 0),
                'Uptime': status.get('uptime', 'N/A')
            })
        
        df = pd.DataFrame(data)
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
