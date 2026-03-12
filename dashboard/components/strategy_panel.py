"""
Strategy Panel Component
========================

Strategy management and monitoring panel.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class StrategyPanel:
    """
    Strategy management and monitoring panel.
    """
    
    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
    
    def render(
        self,
        strategies: Dict[str, Any],
        performance: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Render the strategy panel.
        
        Args:
            strategies: Dictionary of strategies
            performance: Performance metrics
        """
        st.header("📈 Strategy Panel")
        
        # Strategy selector
        selected_strategy = self._render_selector(strategies)
        
        # Strategy overview
        col1, col2 = st.tabs(["Overview", "Performance"])
        
        with col1:
            self._render_overview(strategies)
        
        with col2:
            self._render_performance(performance or {})
        
        # Strategy controls
        self._render_controls(selected_strategy, strategies)
    
    def _render_selector(self, strategies: Dict[str, Any]) -> Optional[str]:
        """Render strategy selector."""
        if not strategies:
            st.info("No strategies configured")
            return None
        
        strategy_names = list(strategies.keys())
        
        selected = st.selectbox(
            "Select Strategy",
            strategy_names,
            index=0
        )
        
        return selected
    
    def _render_overview(self, strategies: Dict[str, Any]) -> None:
        """Render strategy overview."""
        if not strategies:
            st.info("No strategies to display")
            return
        
        # Create cards for each strategy
        cols = st.columns(min(len(strategies), 3))
        
        for i, (name, strategy) in enumerate(strategies.items()):
            with cols[i % 3]:
                self._render_strategy_card(name, strategy)
    
    def _render_strategy_card(self, name: str, strategy: Any) -> None:
        """Render individual strategy card."""
        # Get strategy info
        if isinstance(strategy, dict):
            info = strategy
        else:
            info = self._extract_strategy_info(strategy)
        
        status = info.get('status', 'unknown')
        status_color = {
            'running': '#4CAF50',
            'stopped': '#9E9E9E',
            'paused': '#FFC107',
            'error': '#f44336'
        }.get(status, '#9E9E9E')
        
        # Card header with status
        st.markdown(
            f"""
            <div style='background-color: rgba(33, 150, 243, 0.1); 
                        padding: 15px; 
                        border-radius: 10px; 
                        border-left: 4px solid {status_color};'>
                <h4>{name}</h4>
                <p style='color: {status_color};'>● {status.upper()}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Positions",
                info.get('positions_count', 0)
            )
        
        with col2:
            st.metric(
                "Total PnL",
                f"${info.get('total_pnl', 0):,.2f}"
            )
        
        # Additional info
        with st.expander("Details"):
            st.json(info)
    
    def _render_performance(self, performance: Dict[str, Any]) -> None:
        """Render performance metrics."""
        if not performance:
            st.info("No performance data available")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{performance.get('total_return', 0) * 100:.2f}%",
                delta=performance.get('daily_return', 0) * 100
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{performance.get('sharpe_ratio', 0):.2f}"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{performance.get('max_drawdown', 0) * 100:.2f}%"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{performance.get('win_rate', 0) * 100:.1f}%"
            )
        
        # Performance chart
        if 'equity_curve' in performance:
            self._render_equity_curve(performance['equity_curve'])
        
        # Trade statistics
        if 'trade_stats' in performance:
            self._render_trade_stats(performance['trade_stats'])
    
    def _render_equity_curve(self, equity_data: List[Dict]) -> None:
        """Render equity curve chart."""
        if not equity_data:
            return
        
        df = pd.DataFrame(equity_data)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                y=df['equity'] if 'equity' in df.columns else df.iloc[:, 0],
                mode='lines',
                name='Equity',
                line=dict(color='#2196F3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)'
            )
        )
        
        fig.update_layout(
            title="Equity Curve",
            template='plotly_dark',
            height=300,
            xaxis_title='Time',
            yaxis_title='Equity ($)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_trade_stats(self, stats: Dict[str, Any]) -> None:
        """Render trade statistics."""
        st.subheader("Trade Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Trades", stats.get('total_trades', 0))
            st.metric("Winning", stats.get('winning_trades', 0))
        
        with col2:
            st.metric("Losing", stats.get('losing_trades', 0))
            st.metric("Avg Win", f"${stats.get('avg_win', 0):,.2f}")
        
        with col3:
            st.metric("Avg Loss", f"${stats.get('avg_loss', 0):,.2f}")
            st.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
    
    def _render_controls(self, selected: Optional[str], strategies: Dict) -> None:
        """Render strategy control buttons."""
        st.subheader("Controls")
        
        if not selected:
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.success(f"Starting {selected}...")
                # Would call API to start strategy
        
        with col2:
            if st.button("⏸️ Pause", use_container_width=True):
                st.warning(f"Pausing {selected}...")
                # Would call API to pause strategy
        
        with col3:
            if st.button("⏹️ Stop", use_container_width=True):
                st.error(f"Stopping {selected}...")
                # Would call API to stop strategy
    
    def _extract_strategy_info(self, strategy: Any) -> Dict[str, Any]:
        """Extract strategy info from object."""
        info = {}
        
        if hasattr(strategy, 'get_status'):
            info = strategy.get_status()
        elif hasattr(strategy, 'status'):
            info['status'] = str(strategy.status)
        
        if hasattr(strategy, '_positions'):
            info['positions_count'] = len(strategy._positions)
        
        if hasattr(strategy, '_metrics'):
            info.update(strategy._metrics)
        
        return info


def render_strategy_comparison(
    strategies: Dict[str, Dict[str, Any]]
) -> go.Figure:
    """
    Render strategy comparison chart.
    
    Args:
        strategies: Dictionary of strategies with performance data
    
    Returns:
        Plotly Figure
    """
    if not strategies:
        return go.Figure()
    
    # Extract metrics for comparison
    metrics = ['total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']
    
    data = []
    for name, perf in strategies.items():
        row = {'Strategy': name}
        for metric in metrics:
            row[metric] = perf.get(metric, 0)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric.replace('_', ' ').title(),
                x=df['Strategy'],
                y=df[metric],
                text=df[metric].apply(lambda x: f'{x:.2f}'),
                textposition='outside'
            )
        )
    
    fig.update_layout(
        title="Strategy Comparison",
        template='plotly_dark',
        barmode='group',
        height=400,
        xaxis_title='Strategy',
        yaxis_title='Value'
    )
    
    return fig


def render_signal_history(
    signals: List[Dict[str, Any]]
) -> None:
    """
    Render signal history table.
    
    Args:
        signals: List of signals
    """
    st.subheader("Recent Signals")
    
    if not signals:
        st.info("No signals generated")
        return
    
    df = pd.DataFrame(signals[:50])
    
    # Format timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Highlight buy/sell
    def highlight_side(val):
        if val == 'buy':
            return 'background-color: rgba(76, 175, 80, 0.3)'
        elif val == 'sell':
            return 'background-color: rgba(244, 67, 54, 0.3)'
        return ''
    
    if 'side' in df.columns:
        styled_df = df.style.applymap(highlight_side, subset=['side'])
    else:
        styled_df = df
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=300
    )
