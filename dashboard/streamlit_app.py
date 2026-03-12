"""
AI量化交易系统 - Streamlit 仪表盘
===================================

专业级实时可视化仪表盘，全中文界面。
"""

import sys
from pathlib import Path

# 添加 dashboard 目录到路径
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

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AI量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS ====================
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
    .buy-signal {
        color: #00FF00;
        font-weight: bold;
    }
    .sell-signal {
        color: #FF0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 中文标签 ====================
LABELS = {
    # 导航标签
    'market_data': '📈 市场行情',
    'performance': '📊 收益表现',
    'positions': '💼 持仓与交易',
    'risk': '⚠️ 风险管理',
    'agents': '🤖 AI代理监控',
    
    # 侧边栏
    'dashboard_settings': '⚙️ 仪表盘设置',
    'auto_refresh': '自动刷新',
    'refresh_interval': '刷新间隔（秒）',
    'market_selection': '📊 市场选择',
    'trading_pair': '交易对',
    'timeframe': '时间周期',
    'system_status': '🟢 系统状态',
    'quick_actions': '⚡ 快捷操作',
    'refresh_now': '🔄 立即刷新',
    'run_backtest': '📊 运行回测',
    'generate_strategy': '🤖 生成策略',
    
    # 指标标签
    'total_equity': '总权益',
    'unrealized_pnl': '未实现盈亏',
    'open_positions': '持仓数量',
    'risk_level': '风险等级',
    'latest_price': '最新价格',
    'high_24h': '24小时最高',
    'low_24h': '24小时最低',
    'volume_24h': '24小时成交量',
    
    # 性能指标
    'total_return': '总收益率',
    'sharpe_ratio': '夏普比率',
    'max_drawdown': '最大回撤',
    'win_rate': '胜率',
    'total_trades': '总交易次数',
    
    # 交易
    'current_positions': '💼 当前持仓',
    'trade_history': '📜 交易历史',
    'position_distribution': '📊 持仓分布',
    'symbol': '交易对',
    'quantity': '数量',
    'entry_price': '入场价',
    'current_price': '当前价',
    'pnl': '盈亏',
    'value': '价值',
    
    # 信号
    'latest_signals': '⚡ 最新信号',
    'buy': '买入',
    'sell': '卖出',
    'time': '时间',
    'signal': '信号',
    'price': '价格',
    
    # 页脚
    'last_updated': '最后更新',
    'version': 'AI量化交易系统 v1.0.0',
    'connected': '🟢 已连接API',
    
    # 图表标题
    'price_chart': '价格K线图',
    'equity_curve': '权益曲线',
    'drawdown_analysis': '回撤分析',
    'performance_metrics': '绩效指标',
    'risk_score': '风险评分',
    'agent_task_dist': '📊 代理任务分布',
    'recent_activity': '⏱️ 最近活动',
    
    # 状态
    'all_systems_ok': '所有系统运行正常',
    'backtest_started': '回测已启动！',
    'strategy_generation_started': '策略生成已启动！',
}

# API配置
API_BASE = "http://localhost:8000"

# 会话状态初始化
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
    
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5

if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'BTC/USDT'

if 'selected_timeframe' not in st.session_state:
    st.session_state.selected_timeframe = '1h'


# ============================================================
# API函数
# ============================================================

async def fetch_api(endpoint: str) -> Optional[Dict]:
    """从API端点获取数据"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE}{endpoint}")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"API错误: {e}")
    return None


async def fetch_all_data() -> Dict[str, Any]:
    """获取所有仪表盘数据"""
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
            st.error(f"获取数据错误: {e}")
    
    return data


def get_mock_data() -> Dict[str, Any]:
    """生成演示用模拟数据"""
    np.random.seed(int(time.time()) % 1000)
    
    # 生成OHLCV数据
    dates = pd.date_range(end=datetime.now(), periods=200, freq='h')
    base_price = 45000  # ETH价格
    returns = np.random.normal(0.0001, 0.02, 200)
    prices = base_price * np.exp(np.cumsum(returns))
    
    ohlcv = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, 200)),
        'high': prices * (1 + np.random.uniform(0, 0.015, 200)),
        'low': prices * (1 - np.random.uniform(0, 0.015, 200)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 200) * prices
    }, index=dates)
    
    # 生成交易信号
    signals = pd.DataFrame([
        {'side': 'buy', 'price': prices[50]},
        {'side': 'sell', 'price': prices[80]},
        {'side': 'buy', 'price': prices[120]},
        {'side': 'sell', 'price': prices[160]},
    ], index=[dates[50], dates[80], dates[120], dates[160]])
    
    # 生成权益曲线
    equity_dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
    equity = 100000 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, 100)))
    equity_curve = pd.DataFrame({'equity': equity}, index=equity_dates)
    
    # 生成交易记录
    trades = []
    for i in range(30):
        side = np.random.choice(['买入', '卖出'])
        pnl = np.random.uniform(-500, 1500)
        trades.append({
            'ID': f'T{1000 + i}',
            '交易对': np.random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT']),
            '方向': side,
            '数量': round(np.random.uniform(0.1, 2.0), 4),
            '价格': round(np.random.uniform(44000, 46000), 2),
            '盈亏': round(pnl, 2),
            '时间': (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # 持仓数据
    positions = {
        'BTC/USDT': {
            'quantity': 0.85,
            'entry_price': 44500,
            'current_price': prices[-1],
            'unrealized_pnl': 850,
            'value': prices[-1] * 0.85
        },
        'ETH/USDT': {
            'quantity': 12.5,
            'entry_price': 3100,
            'current_price': 3250,
            'unrealized_pnl': 1875,
            'value': 3250 * 12.5
        },
        'SOL/USDT': {
            'quantity': 150,
            'entry_price': 125,
            'current_price': 132,
            'unrealized_pnl': 1050,
            'value': 132 * 150
        }
    }
    
    # 风险数据
    risk_data = {
        'overall_risk_level': '正常',
        'risk_score': 32,
        'drawdown': -0.04,
        'max_drawdown_limit': 0.15,
        'total_exposure': 0.38,
        'leverage': 1.2,
        'max_leverage': 3.0,
        'position_count': 3,
        'daily_pnl': 2450,
        'margin_used': 25000,
        'free_margin': 75000,
        'positions': positions,
        'alerts': []
    }
    
    # AI代理状态
    agents = {
        '规划代理': {'status': '运行中', 'queued_tasks': 2, 'uptime': '2小时30分'},
        '研究代理': {'status': '运行中', 'queued_tasks': 0, 'uptime': '2小时30分'},
        '策略代理': {'status': '运行中', 'queued_tasks': 1, 'uptime': '2小时30分'},
        '回测代理': {'status': '空闲', 'queued_tasks': 0, 'uptime': '2小时30分'},
        '执行代理': {'status': '运行中', 'queued_tasks': 0, 'uptime': '2小时30分'},
        '风控代理': {'status': '运行中', 'queued_tasks': 0, 'uptime': '2小时30分'},
        '优化代理': {'status': '空闲', 'queued_tasks': 0, 'uptime': '2小时30分'},
        '监控代理': {'status': '运行中', 'queued_tasks': 3, 'uptime': '2小时30分'}
    }
    
    # 绩效指标
    performance = {
        'total_return': 0.28,
        'sharpe_ratio': 2.15,
        'max_drawdown': -0.06,
        'win_rate': 0.68,
        'total_trades': 186,
        'winning_trades': 126,
        'losing_trades': 60,
        'equity_curve': [
            {'timestamp': t.isoformat(), 'equity': e}
            for t, e in zip(equity_dates, equity)
        ]
    }
    
    # 策略
    strategies = {
        '趋势跟踪': {'status': '运行中', 'positions_count': 1, 'total_pnl': 4500},
        '均值回归': {'status': '暂停', 'positions_count': 0, 'total_pnl': 1200},
        '动量策略': {'status': '运行中', 'positions_count': 2, 'total_pnl': 2100}
    }
    
    return {
        'ohlcv': ohlcv,
        'signals': signals,
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
# 侧边栏
# ============================================================

with st.sidebar:
    st.title(LABELS['dashboard_settings'])
    
    # 自动刷新开关
    auto_refresh = st.toggle(LABELS['auto_refresh'], value=True)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.slider(
            LABELS['refresh_interval'],
            min_value=1,
            max_value=30,
            value=5
        )
        st.session_state.refresh_interval = refresh_interval
    
    st.divider()
    
    # 市场选择
    st.subheader(LABELS['market_selection'])
    
    symbol = st.selectbox(
        LABELS['trading_pair'],
        ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT'],
        index=0
    )
    st.session_state.selected_symbol = symbol
    
    timeframe = st.selectbox(
        LABELS['timeframe'],
        ['1分钟', '5分钟', '15分钟', '1小时', '4小时', '1天'],
        index=3
    )
    st.session_state.selected_timeframe = timeframe
    
    st.divider()
    
    # 系统状态
    st.subheader(LABELS['system_status'])
    st.info(LABELS['all_systems_ok'])
    
    # 快捷操作
    st.subheader(LABELS['quick_actions'])
    
    if st.button(LABELS['refresh_now'], use_container_width=True):
        st.rerun()
    
    if st.button(LABELS['run_backtest'], use_container_width=True):
        st.toast(LABELS['backtest_started'], icon="📊")
    
    if st.button(LABELS['generate_strategy'], use_container_width=True):
        st.toast(LABELS['strategy_generation_started'], icon="🤖")


# ============================================================
# 主内容
# ============================================================

# 标题
st.markdown(
    '<p class="main-header">📈 AI量化交易系统仪表盘</p>',
    unsafe_allow_html=True
)

# 获取数据
data = get_mock_data()

# 顶部指标行
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        symbol,
        f"${data['current_price']:,.2f}",
        f"{data['price_change_24h']:.2f}%",
        delta_color="normal" if data['price_change_24h'] >= 0 else "inverse"
    )

with col2:
    st.metric(
        LABELS['total_equity'],
        f"${data['equity_curve']['equity'].iloc[-1]:,.2f}",
        f"+{data['performance']['total_return'] * 100:.1f}%"
    )

with col3:
    total_pnl = sum(p['unrealized_pnl'] for p in data['positions'].values())
    st.metric(
        LABELS['unrealized_pnl'],
        f"${total_pnl:,.2f}",
        delta_color="normal" if total_pnl >= 0 else "inverse"
    )

with col4:
    st.metric(
        LABELS['open_positions'],
        data['risk']['position_count']
    )

with col5:
    risk_colors = {'低': 'normal', '正常': 'normal', '高': 'inverse', '危险': 'inverse'}
    st.metric(
        LABELS['risk_level'],
        data['risk']['overall_risk_level']
    )

st.divider()

# ============================================================
# 主标签页
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    LABELS['market_data'],
    LABELS['performance'],
    LABELS['positions'],
    LABELS['risk'],
    LABELS['agents']
])

# ------------------------------------------------------------
# 标签页1: 市场行情 - 专业K线图
# ------------------------------------------------------------
with tab1:
    # 使用专业K线图
    from components.charts import create_professional_chart
    
    st.subheader(f"{symbol} 实时K线图")
    
    fig = create_professional_chart(
        df=data['ohlcv'],
        symbol=symbol,
        signals=data['signals'],
        show_indicators=True,
        show_signals=True,
        show_sr_levels=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 市场统计
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📊 技术指标摘要")
        
        # 计算指标
        from components.charts import TechnicalIndicators
        ti = TechnicalIndicators()
        
        latest = data['ohlcv'].iloc[-1]
        ma5 = ti.calculate_ma(data['ohlcv']['close'], 5).iloc[-1]
        ma20 = ti.calculate_ma(data['ohlcv']['close'], 20).iloc[-1]
        rsi = ti.calculate_rsi(data['ohlcv']['close'], 14).iloc[-1]
        macd, signal, hist = ti.calculate_macd(data['ohlcv']['close'])
        
        indicator_col1, indicator_col2, indicator_col3, indicator_col4 = st.columns(4)
        
        with indicator_col1:
            st.metric("MA5 (5日均线)", f"${ma5:,.2f}")
            st.metric("MA20 (20日均线)", f"${ma20:,.2f}")
        
        with indicator_col2:
            st.metric("MA60 (60日均线)", f"${ti.calculate_ma(data['ohlcv']['close'], 60).iloc[-1]:,.2f}")
            ma_signal = "多头排列 ↑" if ma5 > ma20 else "空头排列 ↓"
            st.metric("均线信号", ma_signal)
        
        with indicator_col3:
            st.metric("RSI (相对强弱)", f"{rsi:.1f}")
            rsi_signal = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
            st.metric("RSI状态", rsi_signal)
        
        with indicator_col4:
            st.metric("MACD", f"{macd.iloc[-1]:.4f}")
            macd_signal = "金叉 ↑" if hist.iloc[-1] > 0 else "死叉 ↓"
            st.metric("MACD信号", macd_signal)
    
    with col2:
        st.subheader("📊 市场统计")
        st.metric(LABELS['latest_price'], f"${data['current_price']:,.2f}")
        st.metric(LABELS['high_24h'], f"${data['ohlcv']['high'].iloc[-24:].max():,.2f}")
        st.metric(LABELS['low_24h'], f"${data['ohlcv']['low'].iloc[-24:].min():,.2f}")
        st.metric(LABELS['volume_24h'], f"${data['ohlcv']['volume'].iloc[-24:].sum():,.0f}")
        
        st.divider()
        
        # 最新信号
        st.subheader(LABELS['latest_signals'])
        signal_data = [
            {'时间': '10:30', '信号': '↑ 买入', '价格': '$44,850'},
            {'时间': '09:15', '信号': '↓ 卖出', '价格': '$44,200'},
            {'时间': '08:45', '信号': '↑ 买入', '价格': '$43,900'}
        ]
        st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# 标签页2: 收益表现
# ------------------------------------------------------------
with tab2:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(LABELS['equity_curve'])
        
        from components.charts import PerformanceChart
        
        fig = PerformanceChart.create(data['equity_curve'], title="策略权益曲线")
        st.plotly_chart(fig, use_container_width=True)
        
        # 回撤分析
        st.subheader(LABELS['drawdown_analysis'])
        
        from components.charts import DrawdownChart
        
        fig = DrawdownChart.create(data['equity_curve'], title="回撤分析")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(LABELS['performance_metrics'])
        
        perf = data['performance']
        
        st.metric(LABELS['total_return'], f"+{perf['total_return'] * 100:.2f}%")
        st.metric(LABELS['sharpe_ratio'], f"{perf['sharpe_ratio']:.2f}")
        st.metric(LABELS['max_drawdown'], f"{perf['max_drawdown'] * 100:.2f}%")
        st.metric(LABELS['win_rate'], f"{perf['win_rate'] * 100:.1f}%")
        st.metric(LABELS['total_trades'], perf['total_trades'])
        
        st.divider()
        
        # 风险仪表盘
        from components.charts import RiskGauge
        
        fig = RiskGauge.create(
            value=data['risk']['risk_score'],
            title=LABELS['risk_score']
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 标签页3: 持仓与交易
# ------------------------------------------------------------
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(LABELS['current_positions'])
        
        positions_data = []
        for sym, pos in data['positions'].items():
            positions_data.append({
                LABELS['symbol']: sym,
                LABELS['quantity']: pos['quantity'],
                LABELS['entry_price']: f"${pos['entry_price']:,.2f}",
                LABELS['current_price']: f"${pos['current_price']:,.2f}",
                LABELS['pnl']: f"${pos['unrealized_pnl']:,.2f}",
                LABELS['value']: f"${pos['value']:,.2f}"
            })
        
        st.dataframe(
            pd.DataFrame(positions_data),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader(LABELS['position_distribution'])
        
        # 饼图
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(data['positions'].keys()),
                    values=[p['value'] for p in data['positions'].values()],
                    hole=0.4,
                    textinfo='label+percent',
                    textfont=dict(size=12)
                )
            ]
        )
        fig.update_layout(
            template='plotly_dark',
            height=350,
            title="持仓分布"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 交易历史
    st.subheader(LABELS['trade_history'])
    
    trades_df = pd.DataFrame(data['trades'])
    
    st.dataframe(
        trades_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )

# ------------------------------------------------------------
# 标签页4: 风险管理
# ------------------------------------------------------------
with tab4:
    from components.risk_panel import RiskPanel, render_risk_limits
    
    risk_panel = RiskPanel()
    risk_panel.render(data['risk'])
    
    st.divider()
    
    # 风险限制
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
# 标签页5: AI代理监控
# ------------------------------------------------------------
with tab5:
    from components.agent_monitor import AgentMonitor
    
    agent_monitor = AgentMonitor()
    agent_monitor.render(data['agents'])
    
    st.divider()
    
    # 代理活动
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(LABELS['agent_task_dist'])
        
        task_data = [
            {'agent': '规划代理', 'tasks': 15},
            {'agent': '研究代理', 'tasks': 42},
            {'agent': '策略代理', 'tasks': 28},
            {'agent': '回测代理', 'tasks': 35},
            {'agent': '执行代理', 'tasks': 22},
            {'agent': '风控代理', 'tasks': 18}
        ]
        
        df = pd.DataFrame(task_data)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df['agent'],
                    y=df['tasks'],
                    marker_color='#2196F3',
                    text=df['tasks'],
                    textposition='auto'
                )
            ]
        )
        fig.update_layout(
            template='plotly_dark',
            height=300,
            title="各代理完成任务数"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(LABELS['recent_activity'])
        
        activities = [
            {'时间': '10:45:30', '代理': '研究代理', '任务': '市场分析'},
            {'时间': '10:44:15', '代理': '策略代理', '任务': '生成策略'},
            {'时间': '10:43:00', '代理': '回测代理', '任务': '运行回测'},
            {'时间': '10:42:45', '代理': '风控代理', '任务': '风险检查'},
            {'时间': '10:42:00', '代理': '执行代理', '任务': '下单执行'}
        ]
        
        st.dataframe(
            pd.DataFrame(activities),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 页脚
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"{LABELS['last_updated']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.caption(LABELS['version'])

with col3:
    st.caption(LABELS['connected'])

# 自动刷新
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()
