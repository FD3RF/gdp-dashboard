"""
Dashboard Charts Module
=======================

Visualization components using Plotly.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime


class PriceChart:
    """Real-time price chart with OHLCV candles."""
    
    @staticmethod
    def create(
        df: pd.DataFrame,
        symbol: str = "BTC/USDT",
        show_volume: bool = True
    ) -> go.Figure:
        """
        Create candlestick chart with volume.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            show_volume: Show volume subplot
        
        Returns:
            Plotly Figure
        """
        if df.empty:
            return go.Figure()
        
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Candlestick
        candlestick = go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Volume bars
        if show_volume and 'volume' in df.columns:
            colors = np.where(
                df['close'] >= df['open'],
                '#26a69a',
                '#ef5350'
            )
            
            volume = go.Bar(
                x=df.index,
                y=df['volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.7
            )
            fig.add_trace(volume, row=2, col=1)
        
        # Layout
        fig.update_layout(
            title=f'{symbol} Price Chart',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=500,
            showlegend=False,
            hovermode='x unified'
        )
        
        fig.update_xaxes(
            title_text='Time',
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        fig.update_yaxes(
            title_text='Price' if not show_volume else '',
            gridcolor='rgba(128, 128, 128, 0.2)',
            row=1, col=1
        )
        
        if show_volume:
            fig.update_yaxes(
                title_text='Volume',
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=2, col=1
            )
        
        return fig
    
    @staticmethod
    def add_signals(
        fig: go.Figure,
        signals: pd.DataFrame,
        df: pd.DataFrame
    ) -> go.Figure:
        """Add buy/sell signal markers."""
        if signals.empty:
            return fig
        
        # Buy signals
        buy_signals = signals[signals['side'] == 'buy']
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=df.loc[buy_signals.index, 'low'] * 0.995,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='#00ff00'
                    ),
                    name='Buy Signal'
                )
            )
        
        # Sell signals
        sell_signals = signals[signals['side'] == 'sell']
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=df.loc[sell_signals.index, 'high'] * 1.005,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='#ff0000'
                    ),
                    name='Sell Signal'
                )
            )
        
        return fig


class PerformanceChart:
    """Strategy performance chart with equity curve."""
    
    @staticmethod
    def create(
        equity_curve: pd.DataFrame,
        benchmark: Optional[pd.DataFrame] = None,
        title: str = "Strategy Performance"
    ) -> go.Figure:
        """
        Create equity curve chart.
        
        Args:
            equity_curve: DataFrame with equity values
            benchmark: Optional benchmark data
            title: Chart title
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        if not equity_curve.empty:
            # Strategy equity
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve['equity'],
                    mode='lines',
                    name='Strategy',
                    line=dict(color='#2196F3', width=2)
                )
            )
            
            # Add benchmark if provided
            if benchmark is not None and not benchmark.empty:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark.index,
                        y=benchmark['equity'],
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='#9E9E9E', width=1.5, dash='dash')
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        fig.update_xaxes(gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(gridcolor='rgba(128, 128, 128, 0.2)')
        
        return fig
    
    @staticmethod
    def create_returns_distribution(
        returns: pd.Series,
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """Create returns histogram."""
        fig = go.Figure()
        
        if not returns.empty:
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Returns',
                    marker_color='#2196F3',
                    opacity=0.7
                )
            )
            
            # Add mean line
            mean_return = returns.mean()
            fig.add_vline(
                x=mean_return,
                line_dash='dash',
                line_color='red',
                annotation_text=f'Mean: {mean_return:.4f}'
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Returns',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=300
        )
        
        return fig


class DrawdownChart:
    """Drawdown visualization chart."""
    
    @staticmethod
    def create(
        equity_curve: pd.DataFrame,
        title: str = "Drawdown Analysis"
    ) -> go.Figure:
        """
        Create drawdown chart.
        
        Args:
            equity_curve: DataFrame with equity values
            title: Chart title
        
        Returns:
            Plotly Figure
        """
        if equity_curve.empty:
            return go.Figure()
        
        # Calculate drawdown
        equity = equity_curve['equity']
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.5, 0.5],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity,
                mode='lines',
                name='Equity',
                line=dict(color='#2196F3', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)'
            ),
            row=1, col=1
        )
        
        # Rolling max
        fig.add_trace(
            go.Scatter(
                x=rolling_max.index,
                y=rolling_max,
                mode='lines',
                name='Peak',
                line=dict(color='#4CAF50', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='#f44336', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(244, 67, 54, 0.3)'
            ),
            row=2, col=1
        )
        
        # Max drawdown line
        max_dd = drawdown.min()
        fig.add_hline(
            y=max_dd,
            line_dash='dash',
            line_color='yellow',
            row=2, col=1,
            annotation_text=f'Max DD: {max_dd:.2f}%'
        )
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=500,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02
            )
        )
        
        fig.update_yaxes(title_text='Equity ($)', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        fig.update_xaxes(title_text='Time', row=2, col=1)
        
        return fig


class RiskGauge:
    """Risk level gauge visualization."""
    
    @staticmethod
    def create(
        value: float,
        title: str = "Risk Level",
        max_value: float = 100,
        thresholds: Optional[Dict[str, float]] = None
    ) -> go.Figure:
        """
        Create risk gauge.
        
        Args:
            value: Current value
            title: Gauge title
            max_value: Maximum value
            thresholds: Threshold levels
        
        Returns:
            Plotly Figure
        """
        thresholds = thresholds or {
            'low': 25,
            'medium': 50,
            'high': 75,
            'critical': 100
        }
        
        fig = go.Figure()
        
        # Determine color based on value
        if value < thresholds['low']:
            color = '#4CAF50'  # Green
        elif value < thresholds['medium']:
            color = '#FFC107'  # Yellow
        elif value < thresholds['high']:
            color = '#FF9800'  # Orange
        else:
            color = '#f44336'  # Red
        
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=value,
                title={'text': title, 'font': {'size': 20}},
                gauge={
                    'axis': {
                        'range': [0, max_value],
                        'tickwidth': 1,
                        'tickcolor': 'white'
                    },
                    'bar': {'color': color},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 2,
                    'bordercolor': 'gray',
                    'steps': [
                        {'range': [0, thresholds['low']], 'color': 'rgba(76, 175, 80, 0.3)'},
                        {'range': [thresholds['low'], thresholds['medium']], 'color': 'rgba(255, 193, 7, 0.3)'},
                        {'range': [thresholds['medium'], thresholds['high']], 'color': 'rgba(255, 152, 0, 0.3)'},
                        {'range': [thresholds['high'], max_value], 'color': 'rgba(244, 67, 54, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                },
                number={'font': {'size': 40}, 'suffix': '%'}
            )
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=250,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_multi_gauge(
        metrics: Dict[str, float],
        title: str = "Risk Metrics"
    ) -> go.Figure:
        """Create multiple gauges for different risk metrics."""
        n_metrics = len(metrics)
        
        fig = make_subplots(
            rows=1, cols=n_metrics,
            specs=[[{'type': 'indicator'}] * n_metrics]
        )
        
        for i, (name, value) in enumerate(metrics.items()):
            # Determine color
            if value < 25:
                color = '#4CAF50'
            elif value < 50:
                color = '#FFC107'
            elif value < 75:
                color = '#FF9800'
            else:
                color = '#f44336'
            
            fig.add_trace(
                go.Indicator(
                    mode='gauge+number',
                    value=value,
                    title={'text': name, 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color}
                    },
                    number={'font': {'size': 24}}
                ),
                row=1, col=i + 1
            )
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=200
        )
        
        return fig


class HeatmapChart:
    """Correlation and performance heatmaps."""
    
    @staticmethod
    def create_correlation_heatmap(
        returns_df: pd.DataFrame,
        title: str = "Asset Correlation"
    ) -> go.Figure:
        """Create correlation heatmap."""
        if returns_df.empty:
            return go.Figure()
        
        corr = returns_df.corr()
        
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdBu',
                zmid=0,
                text=[[f'{v:.2f}' for v in row] for row in corr.values],
                texttemplate='%{text}',
                textfont={'size': 10},
                hoverongaps=False
            )
        )
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=400,
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )
        
        return fig
