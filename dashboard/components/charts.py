"""
专业级K线图模块
================

机构级量化交易可视化组件，包含：
- K线图（价格、成交量）
- 技术指标（MA、RSI、MACD）
- 交易信号（买入/卖出箭头）
- 支撑位/阻力位
- 全中文界面
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


# ==================== 中文标签配置 ====================
LABELS = {
    # K线数据
    'open': '开盘价',
    'high': '最高价',
    'low': '最低价',
    'close': '收盘价',
    'volume': '成交量',
    
    # 均线
    'ma5': '5日均线',
    'ma20': '20日均线',
    'ma60': '60日均线',
    
    # 技术指标
    'rsi': '相对强弱指数',
    'macd': 'MACD线',
    'macd_signal': '信号线',
    'macd_hist': '柱状图',
    
    # 信号
    'buy': '买入信号',
    'sell': '卖出信号',
    
    # 图表标题
    'price_chart': '价格K线图',
    'volume_chart': '成交量',
    'rsi_chart': '相对强弱指数 (RSI)',
    'macd_chart': '指数平滑异同移动平均 (MACD)',
    
    # 其他
    'support': '支撑位',
    'resistance': '阻力位',
    'trendline': '趋势线',
    'time': '时间',
    'price': '价格',
    'amount': '金额',
}


# ==================== 技术指标计算 ====================
class TechnicalIndicators:
    """技术指标计算工具类"""
    
    @staticmethod
    def calculate_ma(series: pd.Series, period: int) -> pd.Series:
        """计算移动平均线"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """计算相对强弱指数 RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算 MACD 指标"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def detect_support_resistance(
        df: pd.DataFrame,
        window: int = 20
    ) -> Tuple[List[float], List[float]]:
        """检测支撑位和阻力位"""
        supports = []
        resistances = []
        
        # 简化版：使用局部极值点
        for i in range(window, len(df) - window):
            # 支撑位：局部最低点
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
                supports.append(df['low'].iloc[i])
            # 阻力位：局部最高点
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
                resistances.append(df['high'].iloc[i])
        
        # 返回最近的支撑位和阻力位（去重）
        if supports:
            supports = sorted(list(set(supports)))[-3:]  # 最近3个支撑位
        if resistances:
            resistances = sorted(list(set(resistances)))[-3:]  # 最近3个阻力位
        
        return supports, resistances


# ==================== 专业K线图类 ====================
class ProfessionalCandlestickChart:
    """专业级K线图生成器"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
        # 颜色配置（专业交易软件风格）
        self.colors = {
            'increasing': '#26a69a',      # 上涨 - 绿色
            'decreasing': '#ef5350',      # 下跌 - 红色
            'ma5': '#FFD700',             # MA5 - 金色
            'ma20': '#00BFFF',            # MA20 - 天蓝色
            'ma60': '#FF69B4',            # MA60 - 粉色
            'volume_up': 'rgba(38, 166, 154, 0.7)',
            'volume_down': 'rgba(239, 83, 80, 0.7)',
            'buy_signal': '#00FF00',      # 买入信号 - 亮绿色
            'sell_signal': '#FF0000',     # 卖出信号 - 红色
            'support': '#4CAF50',         # 支撑位 - 绿色
            'resistance': '#F44336',      # 阻力位 - 红色
            'rsi': '#9C27B0',             # RSI - 紫色
            'macd': '#2196F3',            # MACD - 蓝色
            'macd_signal': '#FF9800',     # 信号线 - 橙色
            'macd_pos': 'rgba(76, 175, 80, 0.7)',
            'macd_neg': 'rgba(244, 67, 54, 0.7)',
        }
    
    def create(
        self,
        df: pd.DataFrame,
        symbol: str = "ETH/USDT",
        show_indicators: bool = True,
        show_signals: bool = True,
        signals: Optional[pd.DataFrame] = None,
        show_sr_levels: bool = True
    ) -> go.Figure:
        """
        创建专业级K线图
        
        参数:
            df: OHLCV 数据 DataFrame
            symbol: 交易对符号
            show_indicators: 是否显示技术指标
            show_signals: 是否显示交易信号
            signals: 交易信号 DataFrame
            show_sr_levels: 是否显示支撑/阻力位
            
        返回:
            Plotly Figure 对象
        """
        if df.empty:
            return go.Figure()
        
        # 确保索引是日期时间
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)
        
        # 计算技术指标
        df = self._calculate_all_indicators(df)
        
        # 创建子图布局（4行：价格+成交量+RSI+MACD）
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}]
            ]
        )
        
        # 1. 添加K线图
        self._add_candlestick(fig, df, row=1)
        
        # 2. 添加均线
        self._add_moving_averages(fig, df, row=1)
        
        # 3. 添加支撑/阻力位
        if show_sr_levels:
            self._add_support_resistance(fig, df, row=1)
        
        # 4. 添加交易信号
        if show_signals and signals is not None:
            self._add_signals(fig, df, signals, row=1)
        
        # 5. 添加成交量
        self._add_volume(fig, df, row=2)
        
        # 6. 添加RSI
        if show_indicators:
            self._add_rsi(fig, df, row=3)
            self._add_macd(fig, df, row=4)
        
        # 更新布局
        self._update_layout(fig, symbol)
        
        return fig
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()
        
        # 均线
        df['ma5'] = self.indicators.calculate_ma(df['close'], 5)
        df['ma20'] = self.indicators.calculate_ma(df['close'], 20)
        df['ma60'] = self.indicators.calculate_ma(df['close'], 60)
        
        # RSI
        df['rsi'] = self.indicators.calculate_rsi(df['close'], 14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = \
            self.indicators.calculate_macd(df['close'])
        
        return df
    
    def _add_candlestick(self, fig: go.Figure, df: pd.DataFrame, row: int):
        """添加K线图"""
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=LABELS['price_chart'],
                increasing_line_color=self.colors['increasing'],
                decreasing_line_color=self.colors['decreasing'],
                increasing_line_width=1.5,
                decreasing_line_width=1.5,
                hovertext=[
                    f"<b>{LABELS['time']}:</b> {idx.strftime('%Y-%m-%d %H:%M')}<br>"
                    f"<b>{LABELS['open']}:</b> {row['open']:,.2f}<br>"
                    f"<b>{LABELS['high']}:</b> {row['high']:,.2f}<br>"
                    f"<b>{LABELS['low']}:</b> {row['low']:,.2f}<br>"
                    f"<b>{LABELS['close']}:</b> {row['close']:,.2f}<br>"
                    f"<b>{LABELS['volume']}:</b> {row['volume']:,.0f}"
                    for idx, row in df.iterrows()
                ],
                hoverinfo='text'
            ),
            row=row, col=1
        )
    
    def _add_moving_averages(self, fig: go.Figure, df: pd.DataFrame, row: int):
        """添加均线"""
        # MA5
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ma5'],
                mode='lines',
                name=LABELS['ma5'],
                line=dict(color=self.colors['ma5'], width=1.2),
                hovertemplate=f'<b>{LABELS["ma5"]}</b>: %{{y:,.2f}}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # MA20
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ma20'],
                mode='lines',
                name=LABELS['ma20'],
                line=dict(color=self.colors['ma20'], width=1.2),
                hovertemplate=f'<b>{LABELS["ma20"]}</b>: %{{y:,.2f}}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # MA60
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ma60'],
                mode='lines',
                name=LABELS['ma60'],
                line=dict(color=self.colors['ma60'], width=1.2),
                hovertemplate=f'<b>{LABELS["ma60"]}</b>: %{{y:,.2f}}<extra></extra>'
            ),
            row=row, col=1
        )
    
    def _add_volume(self, fig: go.Figure, df: pd.DataFrame, row: int):
        """添加成交量柱状图"""
        colors = np.where(
            df['close'] >= df['open'],
            self.colors['volume_up'],
            self.colors['volume_down']
        )
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name=LABELS['volume_chart'],
                marker_color=colors,
                hovertemplate=f'<b>{LABELS["volume"]}</b>: %{{y:,.0f}}<extra></extra>'
            ),
            row=row, col=1
        )
    
    def _add_rsi(self, fig: go.Figure, df: pd.DataFrame, row: int):
        """添加RSI指标"""
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                mode='lines',
                name=LABELS['rsi'],
                line=dict(color=self.colors['rsi'], width=1.5),
                hovertemplate=f'<b>{LABELS["rsi"]}</b>: %{{y:.2f}}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # 添加超买超卖线
        fig.add_hline(
            y=70, line_dash='dash', line_color='rgba(244, 67, 54, 0.5)',
            row=row, col=1,
            annotation_text='超买', annotation_position='right'
        )
        fig.add_hline(
            y=30, line_dash='dash', line_color='rgba(76, 175, 80, 0.5)',
            row=row, col=1,
            annotation_text='超卖', annotation_position='right'
        )
        fig.add_hline(
            y=50, line_dash='dot', line_color='rgba(128, 128, 128, 0.3)',
            row=row, col=1
        )
    
    def _add_macd(self, fig: go.Figure, df: pd.DataFrame, row: int):
        """添加MACD指标"""
        # MACD线
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd'],
                mode='lines',
                name=LABELS['macd'],
                line=dict(color=self.colors['macd'], width=1.2),
                hovertemplate=f'<b>MACD</b>: %{{y:.4f}}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # 信号线
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd_signal'],
                mode='lines',
                name=LABELS['macd_signal'],
                line=dict(color=self.colors['macd_signal'], width=1.2),
                hovertemplate=f'<b>信号线</b>: %{{y:.4f}}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # 柱状图
        colors = np.where(
            df['macd_hist'] >= 0,
            self.colors['macd_pos'],
            self.colors['macd_neg']
        )
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['macd_hist'],
                name=LABELS['macd_hist'],
                marker_color=colors,
                hovertemplate=f'<b>MACD柱</b>: %{{y:.4f}}<extra></extra>'
            ),
            row=row, col=1
        )
    
    def _add_support_resistance(self, fig: go.Figure, df: pd.DataFrame, row: int):
        """添加支撑位和阻力位"""
        supports, resistances = self.indicators.detect_support_resistance(df)
        
        # 添加支撑位
        for i, level in enumerate(supports[-2:]):  # 最近2个
            fig.add_hline(
                y=level,
                line_dash='dash',
                line_color=self.colors['support'],
                line_width=1,
                row=row, col=1,
                annotation_text=f"{LABELS['support']} {i+1}: {level:.2f}",
                annotation_position='left',
                annotation_font_size=9,
                annotation_font_color=self.colors['support']
            )
        
        # 添加阻力位
        for i, level in enumerate(resistances[-2:]):  # 最近2个
            fig.add_hline(
                y=level,
                line_dash='dash',
                line_color=self.colors['resistance'],
                line_width=1,
                row=row, col=1,
                annotation_text=f"{LABELS['resistance']} {i+1}: {level:.2f}",
                annotation_position='left',
                annotation_font_size=9,
                annotation_font_color=self.colors['resistance']
            )
    
    def _add_signals(
        self, 
        fig: go.Figure, 
        df: pd.DataFrame, 
        signals: pd.DataFrame, 
        row: int
    ):
        """添加交易信号箭头"""
        if signals.empty:
            return
        
        # 确保信号索引对齐
        common_index = signals.index.intersection(df.index)
        if common_index.empty:
            return
        
        # 买入信号
        buy_signals = signals[signals['side'].str.lower() == 'buy']
        buy_index = buy_signals.index.intersection(df.index)
        
        if not buy_index.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_index,
                    y=df.loc[buy_index, 'low'] * 0.995,
                    mode='markers+text',
                    name=LABELS['buy'],
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color=self.colors['buy_signal'],
                        line=dict(color='white', width=1)
                    ),
                    text='↑',
                    textposition='bottom center',
                    textfont=dict(size=14, color=self.colors['buy_signal']),
                    hovertemplate=(
                        f'<b>{LABELS["buy"]}</b><br>'
                        '<b>时间</b>: %{x|%Y-%m-%d %H:%M}<br>'
                        '<b>价格</b>: %{y:,.2f}<extra></extra>'
                    )
                ),
                row=row, col=1
            )
        
        # 卖出信号
        sell_signals = signals[signals['side'].str.lower() == 'sell']
        sell_index = sell_signals.index.intersection(df.index)
        
        if not sell_index.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_index,
                    y=df.loc[sell_index, 'high'] * 1.005,
                    mode='markers+text',
                    name=LABELS['sell'],
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color=self.colors['sell_signal'],
                        line=dict(color='white', width=1)
                    ),
                    text='↓',
                    textposition='top center',
                    textfont=dict(size=14, color=self.colors['sell_signal']),
                    hovertemplate=(
                        f'<b>{LABELS["sell"]}</b><br>'
                        '<b>时间</b>: %{x|%Y-%m-%d %H:%M}<br>'
                        '<b>价格</b>: %{y:,.2f}<extra></extra>'
                    )
                ),
                row=row, col=1
            )
    
    def _update_layout(self, fig: go.Figure, symbol: str):
        """更新图表布局"""
        fig.update_layout(
            title={
                'text': f"<b>{symbol} 实时K线图</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=18, color='white')
            },
            template='plotly_dark',
            height=800,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font=dict(size=10)
            ),
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            margin=dict(l=60, r=60, t=80, b=40)
        )
        
        # 更新X轴
        fig.update_xaxes(
            title_text=LABELS['time'],
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True,
            row=4, col=1
        )
        
        # 更新各Y轴
        fig.update_yaxes(
            title_text=LABELS['price'],
            gridcolor='rgba(128, 128, 128, 0.2)',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text=LABELS['volume'],
            gridcolor='rgba(128, 128, 128, 0.2)',
            row=2, col=1
        )
        fig.update_yaxes(
            title_text='RSI',
            gridcolor='rgba(128, 128, 128, 0.2)',
            range=[0, 100],
            row=3, col=1
        )
        fig.update_yaxes(
            title_text='MACD',
            gridcolor='rgba(128, 128, 128, 0.2)',
            row=4, col=1
        )
        
        # 移除周末空隙
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=['sat', 'mon'])
            ]
        )


# ==================== 简化接口 ====================
def create_professional_chart(
    df: pd.DataFrame,
    symbol: str = "ETH/USDT",
    signals: Optional[pd.DataFrame] = None,
    show_indicators: bool = True,
    show_signals: bool = True,
    show_sr_levels: bool = True
) -> go.Figure:
    """
    创建专业K线图的简化接口
    
    参数:
        df: OHLCV 数据
        symbol: 交易对
        signals: 交易信号 {'side': 'buy'|'sell'}
        show_indicators: 显示技术指标
        show_signals: 显示交易信号
        show_sr_levels: 显示支撑阻力位
        
    返回:
        Plotly Figure
    """
    chart = ProfessionalCandlestickChart()
    return chart.create(
        df, symbol, show_indicators, show_signals, signals, show_sr_levels
    )


# ==================== 保留兼容的旧类 ====================
class PriceChart:
    """兼容旧接口的价格图表类"""
    
    @staticmethod
    def create(df: pd.DataFrame, symbol: str = "BTC/USDT", 
               show_volume: bool = True) -> go.Figure:
        """创建K线图（兼容接口）"""
        return create_professional_chart(df, symbol, show_indicators=True)
    
    @staticmethod
    def add_signals(fig: go.Figure, signals: pd.DataFrame, 
                    df: pd.DataFrame) -> go.Figure:
        """添加信号（兼容接口）"""
        return fig


class PerformanceChart:
    """收益表现图表"""
    
    @staticmethod
    def create(
        equity_curve: pd.DataFrame,
        benchmark: Optional[pd.DataFrame] = None,
        title: str = "策略收益表现"
    ) -> go.Figure:
        """创建权益曲线图"""
        fig = go.Figure()
        
        if not equity_curve.empty:
            # 策略权益
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve['equity'],
                    mode='lines',
                    name='策略',
                    line=dict(color='#2196F3', width=2),
                    hovertemplate='<b>权益</b>: $%{y:,.2f}<extra></extra>'
                )
            )
            
            # 基准对比
            if benchmark is not None and not benchmark.empty:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark.index,
                        y=benchmark['equity'],
                        mode='lines',
                        name='基准',
                        line=dict(color='#9E9E9E', width=1.5, dash='dash'),
                        hovertemplate='<b>基准</b>: $%{y:,.2f}<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis_title=LABELS['time'],
            yaxis_title='权益 ($)',
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
        title: str = "收益分布"
    ) -> go.Figure:
        """创建收益分布直方图"""
        fig = go.Figure()
        
        if not returns.empty:
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='收益率',
                    marker_color='#2196F3',
                    opacity=0.7,
                    hovertemplate='<b>收益率</b>: %{x:.4f}<br><b>频次</b>: %{y}<extra></extra>'
                )
            )
            
            # 添加均值线
            mean_return = returns.mean()
            fig.add_vline(
                x=mean_return,
                line_dash='dash',
                line_color='red',
                annotation_text=f'均值: {mean_return:.4f}'
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='收益率',
            yaxis_title='频次',
            template='plotly_dark',
            height=300
        )
        
        return fig


class DrawdownChart:
    """回撤分析图表"""
    
    @staticmethod
    def create(
        equity_curve: pd.DataFrame,
        title: str = "回撤分析"
    ) -> go.Figure:
        """创建回撤图表"""
        if equity_curve.empty:
            return go.Figure()
        
        # 计算回撤
        equity = equity_curve['equity']
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.5, 0.5],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # 权益曲线
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity,
                mode='lines',
                name='权益',
                line=dict(color='#2196F3', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)',
                hovertemplate='<b>权益</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 峰值
        fig.add_trace(
            go.Scatter(
                x=rolling_max.index,
                y=rolling_max,
                mode='lines',
                name='峰值',
                line=dict(color='#4CAF50', width=1, dash='dash'),
                hovertemplate='<b>峰值</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 回撤
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='回撤',
                line=dict(color='#f44336', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(244, 67, 54, 0.3)',
                hovertemplate='<b>回撤</b>: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 最大回撤线
        max_dd = drawdown.min()
        fig.add_hline(
            y=max_dd,
            line_dash='dash',
            line_color='yellow',
            row=2, col=1,
            annotation_text=f'最大回撤: {max_dd:.2f}%'
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
        
        fig.update_yaxes(title_text='权益 ($)', row=1, col=1)
        fig.update_yaxes(title_text='回撤 (%)', row=2, col=1)
        fig.update_xaxes(title_text=LABELS['time'], row=2, col=1)
        
        return fig


class RiskGauge:
    """风险仪表盘"""
    
    @staticmethod
    def create(
        value: float,
        title: str = "风险水平",
        max_value: float = 100,
        thresholds: Optional[Dict[str, float]] = None
    ) -> go.Figure:
        """创建风险仪表"""
        thresholds = thresholds or {
            'low': 25,
            'medium': 50,
            'high': 75,
            'critical': 100
        }
        
        fig = go.Figure()
        
        # 根据值确定颜色
        if value < thresholds['low']:
            color = '#4CAF50'  # 绿色
        elif value < thresholds['medium']:
            color = '#FFC107'  # 黄色
        elif value < thresholds['high']:
            color = '#FF9800'  # 橙色
        else:
            color = '#f44336'  # 红色
        
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
        title: str = "风险指标"
    ) -> go.Figure:
        """创建多指标仪表"""
        n_metrics = len(metrics)
        
        fig = make_subplots(
            rows=1, cols=n_metrics,
            specs=[[{'type': 'indicator'}] * n_metrics]
        )
        
        for i, (name, value) in enumerate(metrics.items()):
            # 确定颜色
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
    """热力图"""
    
    @staticmethod
    def create_correlation_heatmap(
        returns_df: pd.DataFrame,
        title: str = "资产相关性"
    ) -> go.Figure:
        """创建相关性热力图"""
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
                hoverongaps=False,
                hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
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


# 导出
__all__ = [
    'ProfessionalCandlestickChart',
    'create_professional_chart',
    'TechnicalIndicators',
    'PriceChart',
    'PerformanceChart',
    'DrawdownChart',
    'RiskGauge',
    'HeatmapChart',
    'LABELS'
]
