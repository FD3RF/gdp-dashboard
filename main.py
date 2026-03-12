# main.py
"""
Oracle AI Agent 主程序
=====================
12层架构核心整合
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# Layer 2: 数据采集层
from data.market_stream import (
    get_realtime_eth_data, 
    get_orderbook_data,
    get_funding_rate,
    get_exchange_manager
)

# Layer 3/4: 数据处理与融合
from data.kline_builder import calculate_indicators, get_latest_indicators

# Layer 5: 特征工程层
from features.orderbook_features import OrderbookAnalyzer, analyze_orderbook_imbalance
from features.hurst import HurstExponent
from features.liquidity_heatmap import generate_liquidity_heatmap
from features.funding_rate import analyze_funding_rate

# Layer 6/7/8: AI分析层
from ai.probability_model import calculate_probabilities

# Layer 9: 交易计划层
from analysis.decision_maker import make_decision, Signal
from analysis.trade_plan import generate_trade_plan
from analysis.risk_monitor import risk_warning

# Layer 11: 解释层
from explain.signal_explainer import explain_signal
from explain.signal_history import record_signal, get_signal_reliability

# 配置
from config import AGENT_CONFIG, RISK_CONFIG, BASE_PRICES


@dataclass
class SystemState:
    """系统状态"""
    symbol: str
    timestamp: str
    exchange: str
    price: float
    indicators: Dict[str, Any]
    probabilities: Dict[str, float]
    signal: Signal
    plan: Any
    risk: Any
    orderbook: Dict[str, Any]
    
    # 新增功能
    hurst: float
    hurst_state: str
    liquidity: Dict[str, Any]
    funding: Dict[str, Any]
    explanation: Dict[str, Any]
    reliability: Dict[str, Any]
    
    is_simulated: bool = False


def run_system(symbol: str = "ETH/USDT", use_simulated: bool = False) -> Optional[SystemState]:
    """
    运行完整分析流程 (12层架构整合)
    
    Args:
        symbol: 交易对
        use_simulated: 是否使用模拟数据
        
    Returns:
        SystemState 或 None
    """
    # === Layer 2: 数据采集 ===
    df, current_price = get_realtime_eth_data(symbol, use_simulated=use_simulated)
    
    if df is None:
        return None
    
    # 获取交易所信息
    manager = get_exchange_manager()
    exchange_status = manager.get_status()
    primary_exchange = next(
        (ex for ex, status in exchange_status.items() if status.get('is_primary')),
        list(exchange_status.keys())[0] if exchange_status else "unknown"
    )
    
    # === Layer 3: 数据处理 ===
    df = calculate_indicators(df)
    indicators = get_latest_indicators(df)
    
    # === Layer 5: 特征工程 ===
    # 订单簿分析
    orderbook_data = get_orderbook_data(symbol, limit=50)
    orderbook = orderbook_data.get('bids', []) and orderbook_data.get('asks', [])
    orderbook_analysis = {}
    if orderbook:
        analyzer = OrderbookAnalyzer()
        orderbook_analysis = analyzer.analyze({
            'bids': orderbook_data.get('bids', []),
            'asks': orderbook_data.get('asks', [])
        })
    
    # Hurst指数
    hurst_calc = HurstExponent()
    hurst_value, hurst_state = hurst_calc.calculate(df['close'].values)
    
    # 流动性热图
    liquidity = generate_liquidity_heatmap({
        'bids': orderbook_data.get('bids', []),
        'asks': orderbook_data.get('asks', [])
    }, current_price)
    
    # 资金费率
    funding_rate = get_funding_rate(symbol)
    funding = analyze_funding_rate(funding_rate)
    
    # === Layer 6/7/8: AI分析 ===
    probs = calculate_probabilities(df)
    
    # 融合订单簿和资金费率到概率
    if orderbook_analysis and hasattr(orderbook_analysis, 'imbalance'):
        imbalance = orderbook_analysis.imbalance
        # 微调概率
        if imbalance > 0.3:
            probs['long'] = min(90, probs['long'] + 5)
            probs['short'] = max(5, probs['short'] - 3)
        elif imbalance < -0.3:
            probs['short'] = min(90, probs['short'] + 5)
            probs['long'] = max(5, probs['long'] - 3)
    
    # === Layer 9: 决策 ===
    decision = make_decision(probs)
    plan = generate_trade_plan(df, decision.signal)
    
    # === Layer 10: 风险预警 ===
    risk = risk_warning(df)
    
    # === Layer 11: 解释 ===
    explanation = explain_signal(
        decision.signal.value,
        probs,
        indicators,
        orderbook_analysis.__dict__ if hasattr(orderbook_analysis, '__dict__') else {},
        funding
    )
    
    # 信号可靠度
    reliability = get_signal_reliability(symbol)
    
    # 检测模拟数据
    is_simulated = use_simulated or (symbol == "ETH/USDT" and 3000 < current_price < 4000)
    
    # 记录信号
    if decision.signal != Signal.HOLD:
        record_signal(
            symbol=symbol,
            signal=decision.signal.value,
            price=current_price,
            confidence=probs['confidence'],
            stop_loss=plan.stop_loss if plan else 0,
            take_profit=plan.take_profit_1 if plan else 0
        )
    
    return SystemState(
        symbol=symbol,
        timestamp=str(df['timestamp'].iloc[-1]) if len(df) > 0 else str(datetime.now()),
        exchange=primary_exchange,
        price=current_price,
        indicators=indicators,
        probabilities=probs,
        signal=decision.signal,
        plan=plan,
        risk=risk,
        orderbook={
            'bid': orderbook_data.get('bid_price', 0),
            'ask': orderbook_data.get('ask_price', 0),
            'spread': orderbook_data.get('spread', 0),
            'imbalance': orderbook_analysis.imbalance if hasattr(orderbook_analysis, 'imbalance') else 0,
            'quality': orderbook_analysis.quality_score if hasattr(orderbook_analysis, 'quality_score') else 0,
        },
        hurst=hurst_value,
        hurst_state=hurst_state,
        liquidity=liquidity,
        funding=funding,
        explanation=explanation,
        reliability=reliability,
        is_simulated=is_simulated
    )


def format_output(state: SystemState) -> str:
    """格式化输出"""
    lines = [
        "=" * 70,
        f"🧠 {state.symbol} AI 分析报告",
        "=" * 70,
        f"📅 时间: {state.timestamp}",
        f"🏦 交易所: {state.exchange}",
        f"💰 当前价格: ${state.price:,.2f}",
        "",
        "┌" + "─" * 68 + "┐",
        "│ 【AI 核心决策】                                                    │",
        "├" + "─" * 68 + "┤",
        f"│   🟢 做多概率: {state.probabilities['long']:.1f}%                                           │",
        f"│   🔴 做空概率: {state.probabilities['short']:.1f}%                                           │",
        f"│   ⚪ 观望概率: {state.probabilities['hold']:.1f}%                                           │",
        f"│   💪 信心指数: {state.probabilities['confidence']:.1f}                                          │",
        "└" + "─" * 68 + "┘",
        "",
        f"🎯 最终决策: {state.signal.value}",
        "",
        "┌" + "─" * 68 + "┐",
        "│ 【流动性热图 (M-41)】                                              │",
        "├" + "─" * 68 + "┤",
    ]
    
    # 流动性信息
    for support in state.liquidity.get('support_zones', [])[:2]:
        lines.append(f"│   🟢 支撑墙: ${support[0]:,.2f} (强度: {support[1]:.1f}x)                    │")
    for resistance in state.liquidity.get('resistance_zones', [])[:2]:
        lines.append(f"│   🔴 阻力墙: ${resistance[0]:,.2f} (强度: {resistance[1]:.1f}x)                    │")
    
    for warning in state.liquidity.get('trap_warnings', [])[:2]:
        lines.append(f"│   ⚠️ {warning.get('description', '陷阱预警')[:40]:<40}                    │")
    
    lines.extend([
        "└" + "─" * 68 + "┘",
        "",
        f"📊 Hurst指数: {state.hurst:.3f} ({state.hurst_state})",
        f"📈 资金费率: {state.funding.get('rate', 0)*100:.4f}% ({state.funding.get('signal', '中性')})",
        "",
    ])
    
    # 交易计划
    if state.plan:
        lines.extend([
            "┌" + "─" * 68 + "┐",
            "│ 【AI 交易计划】                                                    │",
            "├" + "─" * 68 + "┤",
            f"│   📍 入场价格: ${state.plan.entry:,.2f}                                      │",
            f"│   🛑 止损价格: ${state.plan.stop_loss:,.2f}                                      │",
            f"│   🎯 目标价格: ${state.plan.take_profit_1:,.2f}                                      │",
            f"│   📊 风险收益比: {state.plan.risk_reward:.2f}                                       │",
            f"│   💼 建议仓位: {state.plan.position_size:.1f}%                                    │",
            "└" + "─" * 68 + "┘",
        ])
    else:
        lines.append("⚠️ 当前建议观望，无交易计划")
    
    # 解释
    lines.extend([
        "",
        "┌" + "─" * 68 + "┐",
        "│ 【AI 决策解释】                                                    │",
        "├" + "─" * 68 + "┤",
        f"│   📝 主要来源: {state.explanation.get('primary_source', '综合'):<20}                          │",
        f"│   🏗️ 市场结构: {state.explanation.get('market_structure', '分析中')[:30]:<30}              │",
        f"│   💡 分析理由: {state.explanation.get('reasoning', '')[:40]:<40}              │",
    ])
    
    for warning in state.explanation.get('warnings', [])[:2]:
        lines.append(f"│   ⚠️ {warning[:50]:<50}              │")
    
    lines.extend([
        "└" + "─" * 68 + "┘",
        "",
        "┌" + "─" * 68 + "┐",
        "│ 【系统效能验证 (M-43)】                                            │",
        "├" + "─" * 68 + "┤",
        f"│   📈 近{state.reliability.get('sample_size', 0)}次信号胜率: {state.reliability.get('win_rate', 0):.1f}%                            │",
        f"│   💰 平均盈亏比: {state.reliability.get('avg_rr_ratio', 0):.2f}                                         │",
        f"│   🎖️ 可靠度: {state.reliability.get('level', '计算中')} ({state.reliability.get('score', 0):.0f}分)                            │",
        "└" + "─" * 68 + "┘",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# Oracle Agent 类
class OracleAgent:
    """Oracle AI Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.balance = self.config.get('initial_balance', 10000)
        self.symbol = self.config.get('symbol', 'ETH/USDT')
        
    async def step(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """执行一步分析"""
        symbol = symbol or self.symbol
        state = run_system(symbol)
        
        if state is None:
            return {'error': '获取数据失败', 'price': 0, 'final_action': 'HOLD'}
        
        return {
            'symbol': state.symbol,
            'timestamp': state.timestamp,
            'exchange': state.exchange,
            'price': state.price,
            'probabilities': state.probabilities,
            'final_action': state.signal.value,
            'confidence': state.probabilities['confidence'],
            'risk_level': state.risk.level.value if hasattr(state.risk, 'level') else 'unknown',
            'risk_score': state.risk.score if hasattr(state.risk, 'score') else 0,
            'position_size': state.plan.position_size if state.plan else 0,
            'hurst': state.hurst,
            'hurst_state': state.hurst_state,
            'funding': state.funding,
            'liquidity': state.liquidity,
            'explanation': state.explanation,
            'reliability': state.reliability,
            'plan': {
                'entry': state.plan.entry if state.plan else 0,
                'stop': state.plan.stop_loss if state.plan else 0,
                'target': state.plan.take_profit_1 if state.plan else 0,
            } if state.plan else None,
        }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧠 ETH/USDT AI 盯盘终端 (12层架构版)")
    print("=" * 70 + "\n")
    
    state = run_system()
    
    if state:
        print(format_output(state))
    else:
        print("❌ 获取数据失败")
