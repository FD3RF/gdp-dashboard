# ui/neural_dashboard.py
"""
第7层：神经链接仪表盘
===================

功能：实时可视化界面，一眼决策
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


# 页面配置
st.set_page_config(
    page_title="🧠 Oracle AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(123, 44, 191, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #7b2cbf;
    }
    .signal-buy {
        color: #00ff88;
        font-weight: bold;
    }
    .signal-sell {
        color: #ff4757;
        font-weight: bold;
    }
    .signal-hold {
        color: #ffa502;
        font-weight: bold;
    }
    .risk-safe { color: #00ff88; }
    .risk-caution { color: #ffa502; }
    .risk-danger { color: #ff4757; }
    .risk-critical { color: #ff0000; animation: blink 1s infinite; }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# 中文标签
LABELS = {
    'price': '当前价格',
    'change_24h': '24H涨跌',
    'high_24h': '24H最高',
    'low_24h': '24H最低',
    'volume_24h': '24H成交量',
    'balance': '账户余额',
    'position': '持仓价值',
    'pnl': '未实现盈亏',
    'risk_level': '风险等级',
    'ai_signal': 'AI信号',
    'confidence': '信心指数',
    'action': '建议操作',
    'position_size': '建议仓位',
    'stop_loss': '止损价',
    'take_profit': '止盈价',
    'ma5': 'MA5',
    'ma20': 'MA20',
    'ma60': 'MA60',
    'rsi': 'RSI',
    'macd': 'MACD',
    'bb_upper': '布林上轨',
    'bb_lower': '布林下轨',
    'volume': '成交量',
    'trend': '趋势',
    'momentum': '动量',
    'volatility': '波动率',
    'sentiment': '市场情绪',
    'funding_rate': '资金费率',
    'whale_activity': '鲸鱼活动',
}


class NeuralDashboard:
    """神经链接仪表盘"""
    
    def __init__(self):
        self.market_data = {}
        self.agent_state = {}
        self.trade_history = []
    
    def update_data(self, market_data: Dict, agent_state: Dict, trade_history: List = None):
        """更新数据"""
        self.market_data = market_data
        self.agent_state = agent_state
        if trade_history:
            self.trade_history = trade_history
    
    def render(self):
        """渲染仪表盘"""
        # 标题
        st.markdown('<h1 class="main-header">🧠 Oracle AI Agent</h1>', unsafe_allow_html=True)
        st.markdown("### 全息感知 · 深度强化 · 自我博弈 · 反脆弱风控")
        
        # 顶部指标栏
        self._render_header_metrics()
        
        # 分隔线
        st.divider()
        
        # 主内容区
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # K线图
            self._render_kline_chart()
            # 技术指标
            self._render_indicators()
        
        with col2:
            # AI 决策面板
            self._render_ai_panel()
            # 风控面板
            self._render_risk_panel()
            # 策略信号
            self._render_strategy_signals()
        
        # 底部信息
        st.divider()
        self._render_footer()
    
    def _render_header_metrics(self):
        """渲染顶部指标"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # 价格信息
        price = self.market_data.get('price', 0)
        change_24h = self.market_data.get('price_change_24h', 0) * 100
        
        with col1:
            st.metric(
                LABELS['price'],
                f"${price:,.2f}",
                f"{change_24h:+.2f}%"
            )
        
        with col2:
            balance = self.agent_state.get('balance', 0)
            st.metric(LABELS['balance'], f"${balance:,.2f}")
        
        with col3:
            pnl = self.agent_state.get('unrealized_pnl', 0)
            st.metric(LABELS['pnl'], f"${pnl:,.2f}")
        
        with col4:
            risk = self.agent_state.get('risk_level', 'safe')
            color_map = {
                'safe': 'risk-safe',
                'caution': 'risk-caution', 
                'danger': 'risk-danger',
                'critical': 'risk-critical'
            }
            st.metric(LABELS['risk_level'], risk.upper())
        
        with col5:
            signal = self.agent_state.get('signal', 'HOLD')
            color_class = {
                'LONG': 'signal-buy',
                'SHORT': 'signal-sell',
                'HOLD': 'signal-hold',
                'CLOSE': 'signal-hold'
            }.get(signal, '')
            st.markdown(f"""
            <div class="metric-card">
                <small>{LABELS['ai_signal']}</small><br>
                <span class="{color_class}" style="font-size: 1.5rem; font-weight: bold;">
                    {signal}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_kline_chart(self):
        """渲染K线图"""
        st.subheader("📊 K线图")
        
        # 生成示例数据
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        np.random.seed(42)
        
        base_price = self.market_data.get('price', 50000)
        returns = np.random.randn(100) * 0.01
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(100) * 0.002),
            'high': prices * (1 + np.abs(np.random.randn(100)) * 0.005),
            'low': prices * (1 - np.abs(np.random.randn(100)) * 0.005),
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # 计算均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
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
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4757'
            ),
            row=1, col=1
        )
        
        # 均线
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma5'], name='MA5', line=dict(color='#ffa502', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma20'], name='MA20', line=dict(color='#3498db', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma60'], name='MA60', line=dict(color='#9b59b6', width=1)), row=1, col=1)
        
        # 成交量
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume'], name='成交量', marker_color='#7b2cbf'),
            row=2, col=1
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_indicators(self):
        """渲染技术指标"""
        col1, col2, col3, col4 = st.columns(4)
        
        indicators = {
            'RSI': self.market_data.get('rsi_14', 50),
            'MACD': self.market_data.get('macd_hist', 0),
            'ATR%': self.market_data.get('atr_ratio', 0) * 100,
            '量比': self.market_data.get('volume_ratio', 1),
        }
        
        for i, (name, value) in enumerate(indicators.items()):
            with [col1, col2, col3, col4][i]:
                st.metric(name, f"{value:.2f}")
    
    def _render_ai_panel(self):
        """渲染AI决策面板"""
        st.subheader("🧠 AI决策")
        
        # 信心指数
        confidence = self.agent_state.get('confidence', 0)
        st.progress(confidence, text=f"信心指数: {confidence*100:.1f}%")
        
        # 动作概率
        probs = self.agent_state.get('action_probs', [0.25, 0.25, 0.25, 0.25])
        actions = ['做多', '做空', '平仓', '观望']
        
        for action, prob in zip(actions, probs):
            st.write(f"{action}: {prob*100:.1f}%")
        
        # 建议仓位
        position_size = self.agent_state.get('position_size', 0)
        st.metric("建议仓位", f"${position_size:,.2f}")
        
        # 止损止盈
        col1, col2 = st.columns(2)
        with col1:
            st.metric("止损价", f"${self.agent_state.get('stop_loss', 0):,.2f}")
        with col2:
            st.metric("止盈价", f"${self.agent_state.get('take_profit', 0):,.2f}")
    
    def _render_risk_panel(self):
        """渲染风控面板"""
        st.subheader("🛡️ 风控面板")
        
        # 风险指标
        metrics = self.agent_state.get('risk_metrics', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("日内盈亏", f"${metrics.get('daily_pnl', 0):,.2f}")
            st.metric("最大回撤", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
        
        with col2:
            st.metric("日内交易", f"{metrics.get('daily_trades', 0)}笔")
            st.metric("胜率", f"{metrics.get('win_rate', 0)*100:.1f}%")
        
        # 熔断状态
        is_sleep = metrics.get('is_sleep_mode', False)
        if is_sleep:
            st.error("⛔ 休眠模式激活")
    
    def _render_strategy_signals(self):
        """渲染策略信号"""
        st.subheader("📡 策略信号")
        
        signals = self.agent_state.get('strategy_signals', [])
        
        for signal in signals:
            strategy = signal.get('strategy', 'unknown')
            action = signal.get('action', 'HOLD')
            strength = signal.get('strength', 0)
            
            color = {'LONG': '🟢', 'SHORT': '🔴', 'HOLD': '🟡', 'CLOSE': '🟠'}.get(action, '⚪')
            st.write(f"{color} **{strategy}**: {action} ({strength*100:.0f}%)")
    
    def _render_footer(self):
        """渲染底部信息"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.caption("Oracle AI Agent v1.0")
        
        with col3:
            st.caption("🧠 全息感知 · 深度强化 · 自我博弈")


def run_dashboard():
    """运行仪表盘"""
    dashboard = NeuralDashboard()
    
    # 模拟数据
    market_data = {
        'price': 50000 + np.random.randn() * 1000,
        'price_change_24h': np.random.randn() * 0.05,
        'rsi_14': 30 + np.random.rand() * 40,
        'macd_hist': np.random.randn() * 100,
        'atr_ratio': np.random.rand() * 0.03,
        'volume_ratio': 0.8 + np.random.rand() * 0.4,
    }
    
    agent_state = {
        'balance': 100000,
        'unrealized_pnl': np.random.randn() * 1000,
        'risk_level': np.random.choice(['safe', 'caution', 'danger']),
        'signal': np.random.choice(['LONG', 'SHORT', 'HOLD']),
        'confidence': 0.5 + np.random.rand() * 0.4,
        'action_probs': np.random.dirichlet([1, 1, 1, 1]),
        'position_size': 10000 + np.random.rand() * 5000,
        'stop_loss': 48000,
        'take_profit': 52000,
        'risk_metrics': {
            'daily_pnl': np.random.randn() * 500,
            'daily_trades': int(np.random.rand() * 10),
            'max_drawdown': np.random.rand() * 0.05,
            'win_rate': 0.5 + np.random.randn() * 0.1,
        }
    }
    
    dashboard.update_data(market_data, agent_state)
    dashboard.render()


# 主入口
if __name__ == '__main__':
    run_dashboard()


# 导出
__all__ = ['NeuralDashboard', 'run_dashboard']
