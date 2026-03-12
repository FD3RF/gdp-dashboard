"""
AI Quant Trading System - Streamlit Dashboard
============================================

Real-time visualization dashboard for the AI Quant Trading System.
"""

import sys
from pathlib import Path

# Add dashboard directory to path for component imports
DASHBOARD_DIR = Path(__file__).parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import httpx

# Configuration
st.set_page_config(
    page_title="AI Quant Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: rgba(33, 150, 243, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
    }
    .stMetric > div {
        background-color: rgba(33, 150, 243, 0.05);
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE = "http://localhost:8000"

# Session state initialization
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
    
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5

if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'BTC/USDT'

if 'selected_timeframe' not in st.session_state:
    st.session_state.selected_timeframe = '1h'


# ============================================================
# API Functions
# ============================================================

async def fetch_api(endpoint: str) -> Optional[Dict]:
    """Fetch data from API endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE}{endpoint}")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


async def fetch_all_data() -> Dict[str, Any]:
    """Fetch all dashboard data."""
    data = {}
    
    endpoints = {
        'market': '/api/market-data',
        'trades': '/api/trades',
        'strategies': '/api/strategies',
        'risk': '/api/risk',
        'agents': '/api/agents',
        'positions': '/api/positions',
        'performance': '/api/performance',
        'alerts': '/api/alerts'
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for key, endpoint in endpoints.items():
            tasks.append(client.get(f"{API_BASE}{endpoint}"))
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (key, _) in enumerate(endpoints.items()):
                if isinstance(responses[i], httpx.Response):
                    if responses[i].status_code == 200:
                        data[key] = responses[i].json()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
    
    return data


def get_mock_data() -> Dict[str, Any]:
    """Generate mock data for demonstration."""
    np.random.seed(int(time.time()) % 1000)
    
    # Generate OHLCV data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
    base_price = 50000
    returns = np.random.normal(0.0001, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))
    
    ohlcv = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, 100)),
        'high': prices * (1 + np.random.uniform(0, 0.01, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.01, 100)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100) * prices
    }, index=dates)
    
    # Generate equity curve
    equity_dates = pd.date_range(end=datetime.now(), periods=50, freq='h')
    equity = 100000 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, 50)))
    equity_curve = pd.DataFrame({'equity': equity}, index=equity_dates)
    
    # Generate trades
    trades = []
    for i in range(20):
        side = np.random.choice(['buy', 'sell'])
        pnl = np.random.uniform(-500, 1000) if side == 'sell' else np.random.uniform(-1000, 500)
        trades.append({
            'id': f'T{1000 + i}',
            'symbol': 'BTC/USDT',
            'side': side,
            'quantity': round(np.random.uniform(0.1, 1.0), 4),
            'price': round(np.random.uniform(49000, 51000), 2),
            'pnl': round(pnl, 2),
            'timestamp': (datetime.now() - timedelta(hours=i)).isoformat()
        })
    
    # Generate positions
    positions = {
        'BTC/USDT': {
            'quantity': 0.5,
            'entry_price': 49500,
            'current_price': prices[-1],
            'unrealized_pnl': 500,
            'value': prices[-1] * 0.5
        },
        'ETH/USDT': {
            'quantity': 5.0,
            'entry_price': 3000,
            'current_price': 3100,
            'unrealized_pnl': 500,
            'value': 3100 * 5
        }
    }
    
    # Generate risk data
    risk_data = {
        'overall_risk_level': 'normal',
        'risk_score': 35,
        'drawdown': -0.05,
        'max_drawdown_limit': 0.15,
        'total_exposure': 0.45,
        'leverage': 1.5,
        'max_leverage': 3.0,
        'position_count': 2,
        'daily_pnl': 1250,
        'margin_used': 15000,
        'free_margin': 85000,
        'positions': positions,
        'alerts': []
    }
    
    # Generate agent status
    agents = {
        'PlannerAgent': {'status': 'running', 'queued_tasks': 2, 'uptime': '2h 30m'},
        'ResearchAgent': {'status': 'running', 'queued_tasks': 0, 'uptime': '2h 30m'},
        'StrategyAgent': {'status': 'running', 'queued_tasks': 1, 'uptime': '2h 30m'},
        'BacktestAgent': {'status': 'idle', 'queued_tasks': 0, 'uptime': '2h 30m'},
        'ExecutionAgent': {'status': 'running', 'queued_tasks': 0, 'uptime': '2h 30m'},
        'RiskAgent': {'status': 'running', 'queued_tasks': 0, 'uptime': '2h 30m'},
        'OptimizationAgent': {'status': 'idle', 'queued_tasks': 0, 'uptime': '2h 30m'},
        'MonitoringAgent': {'status': 'running', 'queued_tasks': 3, 'uptime': '2h 30m'}
    }
    
    # Performance metrics
    performance = {
        'total_return': 0.25,
        'sharpe_ratio': 1.85,
        'max_drawdown': -0.08,
        'win_rate': 0.62,
        'total_trades': 156,
        'winning_trades': 97,
        'losing_trades': 59,
        'equity_curve': [
            {'timestamp': t.isoformat(), 'equity': e}
            for t, e in zip(equity_dates, equity)
        ]
    }
    
    # Strategies
    strategies = {
        'trend_following': {
            'status': 'running',
            'positions_count': 1,
            'total_pnl': 2500
        },
        'mean_reversion': {
            'status': 'paused',
            'positions_count': 0,
            'total_pnl': 800
        },
        'momentum': {
            'status': 'running',
            'positions_count': 1,
            'total_pnl': 1200
        }
    }
    
    return {
        'ohlcv': ohlcv,
        'equity_curve': equity_curve,
        'trades': trades,
        'positions': positions,
        'risk': risk_data,
        'agents': agents,
        'performance': performance,
        'strategies': strategies,
        'current_price': prices[-1],
        'price_change_24h': (prices[-1] - prices[-24]) / prices[-24] * 100 if len(prices) >= 24 else 0
    }


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("⚙️ Dashboard Settings")
    
    # Auto refresh toggle
    auto_refresh = st.toggle("Auto Refresh", value=True)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=30,
            value=5
        )
        st.session_state.refresh_interval = refresh_interval
    
    st.divider()
    
    # Trading pair selection
    st.subheader("📊 Market Selection")
    
    symbol = st.selectbox(
        "Trading Pair",
        ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'],
        index=0
    )
    st.session_state.selected_symbol = symbol
    
    timeframe = st.selectbox(
        "Timeframe",
        ['1m', '5m', '15m', '1h', '4h', '1d'],
        index=3
    )
    st.session_state.selected_timeframe = timeframe
    
    st.divider()
    
    # System status
    st.subheader("🟢 System Status")
    st.info("All systems operational")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()
    
    if st.button("📊 Run Backtest", use_container_width=True):
        st.toast("Backtest started!", icon="📊")
    
    if st.button("🤖 Generate Strategy", use_container_width=True):
        st.toast("Strategy generation started!", icon="🤖")


# ============================================================
# Main Content
# ============================================================

# Title
st.markdown(
    '<p class="main-header">📊 AI Quant Trading System Dashboard</p>',
    unsafe_allow_html=True
)

# Get data (use mock data for demo)
data = get_mock_data()

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "BTC/USDT",
        f"${data['current_price']:,.2f}",
        f"{data['price_change_24h']:.2f}%"
    )

with col2:
    st.metric(
        "Total Equity",
        f"${data['equity_curve']['equity'].iloc[-1]:,.2f}",
        f"{data['performance']['total_return'] * 100:.1f}%"
    )

with col3:
    st.metric(
        "Unrealized PnL",
        f"${data['risk']['positions']['BTC/USDT']['unrealized_pnl']:,.2f}"
    )

with col4:
    st.metric(
        "Open Positions",
        data['risk']['position_count']
    )

with col5:
    st.metric(
        "Risk Level",
        data['risk']['overall_risk_level'].upper()
    )

st.divider()

# ============================================================
# Main Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Data",
    "📊 Performance",
    "💼 Positions & Trades",
    "⚠️ Risk Management",
    "🤖 AI Agents"
])

# ------------------------------------------------------------
# Tab 1: Market Data
# ------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Price Chart - {symbol}")
        
        # Create candlestick chart
        from dashboard.components.charts import PriceChart
        
        fig = PriceChart.create(data['ohlcv'], symbol)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Market Stats")
        
        # Price info
        st.metric("Latest Price", f"${data['current_price']:,.2f}")
        st.metric("24h High", f"${data['ohlcv']['high'].iloc[-24:].max():,.2f}")
        st.metric("24h Low", f"${data['ohlcv']['low'].iloc[-24:].min():,.2f}")
        st.metric("24h Volume", f"${data['ohlcv']['volume'].iloc[-24:].sum():,.0f}")
        
        st.divider()
        
        # Quick signals
        st.subheader("⚡ Latest Signals")
        signal_df = pd.DataFrame({
            'Time': ['10:30', '09:15', '08:45'],
            'Signal': ['BUY', 'SELL', 'BUY'],
            'Price': ['$49,850', '$49,200', '$48,900']
        })
        st.dataframe(signal_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Tab 2: Performance
# ------------------------------------------------------------
with tab2:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Equity Curve")
        
        from dashboard.components.charts import PerformanceChart
        
        fig = PerformanceChart.create(data['equity_curve'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        st.subheader("Drawdown Analysis")
        
        from dashboard.components.charts import DrawdownChart
        
        fig = DrawdownChart.create(data['equity_curve'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Performance Metrics")
        
        perf = data['performance']
        
        st.metric("Total Return", f"{perf['total_return'] * 100:.2f}%")
        st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{perf['max_drawdown'] * 100:.2f}%")
        st.metric("Win Rate", f"{perf['win_rate'] * 100:.1f}%")
        st.metric("Total Trades", perf['total_trades'])
        
        st.divider()
        
        # Risk gauge
        from dashboard.components.charts import RiskGauge
        
        fig = RiskGauge.create(
            value=data['risk']['risk_score'],
            title="Risk Score"
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Tab 3: Positions & Trades
# ------------------------------------------------------------
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💼 Current Positions")
        
        positions_data = []
        for sym, pos in data['positions'].items():
            positions_data.append({
                'Symbol': sym,
                'Quantity': pos['quantity'],
                'Entry Price': f"${pos['entry_price']:,.2f}",
                'Current': f"${pos['current_price']:,.2f}",
                'PnL': f"${pos['unrealized_pnl']:,.2f}",
                'Value': f"${pos['value']:,.2f}"
            })
        
        st.dataframe(
            pd.DataFrame(positions_data),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("📊 Position Distribution")
        
        # Pie chart
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(data['positions'].keys()),
                    values=[p['value'] for p in data['positions'].values()],
                    hole=0.4
                )
            ]
        )
        fig.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Trade history
    st.subheader("📜 Trade History")
    
    trades_df = pd.DataFrame(data['trades'])
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(
        trades_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )

# ------------------------------------------------------------
# Tab 4: Risk Management
# ------------------------------------------------------------
with tab4:
    from dashboard.components.risk_panel import RiskPanel, render_risk_limits
    
    risk_panel = RiskPanel()
    risk_panel.render(data['risk'])
    
    st.divider()
    
    # Risk limits
    limits = {
        'current_position_size': 0.015,
        'max_position_size': 0.02,
        'position_size_ok': True,
        'current_drawdown': abs(data['risk']['drawdown']),
        'max_drawdown': data['risk']['max_drawdown_limit'],
        'drawdown_ok': True,
        'current_leverage': data['risk']['leverage'],
        'max_leverage': data['risk']['max_leverage'],
        'leverage_ok': True,
        'current_daily_loss': -0.01,
        'max_daily_loss': 0.05,
        'daily_loss_ok': True
    }
    
    render_risk_limits(limits)

# ------------------------------------------------------------
# Tab 5: AI Agents
# ------------------------------------------------------------
with tab5:
    from dashboard.components.agent_monitor import AgentMonitor
    
    agent_monitor = AgentMonitor()
    agent_monitor.render(data['agents'])
    
    st.divider()
    
    # Agent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Agent Task Distribution")
        
        task_data = [
            {'agent': 'PlannerAgent', 'tasks': 15},
            {'agent': 'ResearchAgent', 'tasks': 42},
            {'agent': 'StrategyAgent', 'tasks': 28},
            {'agent': 'BacktestAgent', 'tasks': 35},
            {'agent': 'ExecutionAgent', 'tasks': 22},
            {'agent': 'RiskAgent', 'tasks': 18}
        ]
        
        df = pd.DataFrame(task_data)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df['agent'],
                    y=df['tasks'],
                    marker_color='#2196F3'
                )
            ]
        )
        fig.update_layout(
            template='plotly_dark',
            height=300,
            title="Tasks Completed by Agent"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Recent Agent Activity")
        
        activities = [
            {'Time': '10:45:30', 'Agent': 'ResearchAgent', 'Task': 'Market Analysis'},
            {'Time': '10:44:15', 'Agent': 'StrategyAgent', 'Task': 'Generate Strategy'},
            {'Time': '10:43:00', 'Agent': 'BacktestAgent', 'Task': 'Run Backtest'},
            {'Time': '10:42:45', 'Agent': 'RiskAgent', 'Task': 'Risk Check'},
            {'Time': '10:42:00', 'Agent': 'ExecutionAgent', 'Task': 'Place Order'}
        ]
        
        st.dataframe(
            pd.DataFrame(activities),
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# Footer
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.caption("AI Quant Trading System v1.0.0")

with col3:
    st.caption("🟢 Connected to API")

# Auto refresh
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()
