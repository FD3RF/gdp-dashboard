# main.py
"""
ETH AI Agent 主程序
===================
整合真实数据获取、AI分析、决策和风控
"""

import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

# 数据模块
from data.market_stream import get_realtime_eth_data, get_orderbook_data
from data.kline_builder import calculate_indicators, get_latest_indicators

# AI 模块
from ai.probability_model import calculate_probabilities

# 分析模块
from analysis.decision_maker import make_decision, Signal
from analysis.trade_plan import generate_trade_plan, TradePlan
from analysis.risk_monitor import risk_warning, RiskWarning

# 配置
from config import AGENT_CONFIG, RISK_CONFIG


@dataclass
class SystemState:
    """系统状态"""
    symbol: str
    timestamp: str
    price: float
    indicators: Dict[str, Any]
    probabilities: Dict[str, float]
    signal: Signal
    plan: Optional[TradePlan]
    risk: RiskWarning
    orderbook: Dict[str, Any]


def run_system(symbol: str = "ETH/USDT", use_simulated: bool = False) -> Optional[SystemState]:
    """
    运行完整分析流程
    
    Args:
        symbol: 交易对，默认 ETH/USDT
        use_simulated: 是否使用模拟数据
        
    Returns:
        SystemState 或 None
    """
    # 1. 获取数据
    df, current_price = get_realtime_eth_data(symbol, use_simulated=use_simulated)
    
    if df is None:
        return None

    # 2. 计算技术指标
    df = calculate_indicators(df)
    indicators = get_latest_indicators(df)

    # 3. AI 计算概率
    probs = calculate_probabilities(df)
    
    # 4. 生成最终信号 (修正逻辑)
    decision = make_decision(probs)

    # 5. 生成交易计划
    plan = generate_trade_plan(df, decision.signal)

    # 6. 风险监控
    risk = risk_warning(df)
    
    # 7. 获取订单簿
    orderbook = get_orderbook_data(symbol)

    return SystemState(
        symbol=symbol,
        timestamp=str(df['timestamp'].iloc[-1]),
        price=current_price,
        indicators=indicators,
        probabilities=probs,
        signal=decision.signal,
        plan=plan,
        risk=risk,
        orderbook=orderbook
    )


def format_output(state: SystemState) -> str:
    """格式化输出"""
    lines = [
        "=" * 60,
        f"🧠 {state.symbol} AI 分析报告",
        "=" * 60,
        f"📅 时间: {state.timestamp}",
        f"💰 当前价格: ${state.price:,.2f}",
        "",
        "📊 AI 概率分布:",
        f"   🟢 做多概率: {state.probabilities['long']:.1f}%",
        f"   🔴 做空概率: {state.probabilities['short']:.1f}%",
        f"   ⚪ 观望概率: {state.probabilities['hold']:.1f}%",
        f"   💪 信心指数: {state.probabilities['confidence']:.1f}",
        "",
        f"🎯 最终决策: {state.signal.value}",
    ]
    
    # 交易计划
    if state.plan:
        lines.extend([
            "",
            "📋 交易计划:",
            f"   入场: ${state.plan.entry:,.2f}",
            f"   止损: ${state.plan.stop_loss:,.2f}",
            f"   目标1: ${state.plan.take_profit_1:,.2f}",
            f"   目标2: ${state.plan.take_profit_2:,.2f}",
            f"   风险收益比: {state.plan.risk_reward:.2f}",
            f"   建议仓位: {state.plan.position_size:.1f}%",
        ])
    else:
        lines.append("\n当前建议观望，无交易计划")
    
    # 风险提示
    lines.extend([
        "",
        f"⚠️ 风险等级: {state.risk.level.value}",
        f"   风险分数: {state.risk.score}/100",
    ])
    
    if state.risk.warnings:
        lines.append("   提示:")
        for w in state.risk.warnings[:3]:
            lines.append(f"     {w}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# Oracle Agent 类 (兼容原有接口)
class OracleAgent:
    """Oracle AI Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.balance = self.config.get('initial_balance', 10000)
        self.symbol = self.config.get('symbol', 'ETH/USDT')
        
    async def step(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        执行一步分析
        
        Args:
            symbol: 交易对
            
        Returns:
            分析结果字典
        """
        symbol = symbol or self.symbol
        state = run_system(symbol)
        
        if state is None:
            return {
                'error': '获取数据失败',
                'price': 0,
                'final_action': 'HOLD',
                'confidence': 0,
                'risk_level': 'unknown',
                'position_size': 0,
                'trap_detected': False,
            }
        
        return {
            'symbol': state.symbol,
            'timestamp': state.timestamp,
            'price': state.price,
            'indicators': state.indicators,
            'probabilities': state.probabilities,
            'final_action': state.signal.value,
            'confidence': state.probabilities['confidence'],
            'risk_level': state.risk.level.value,
            'risk_score': state.risk.score,
            'position_size': state.plan.position_size if state.plan else 0,
            'trap_detected': False,
            'plan': {
                'entry': state.plan.entry if state.plan else 0,
                'stop': state.plan.stop_loss if state.plan else 0,
                'target': state.plan.take_profit_1 if state.plan else 0,
            } if state.plan else None,
            'warnings': state.risk.warnings,
        }


# 主程序入口
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧠 ETH/USDT AI 盯盘终端 (真实数据版)")
    print("=" * 60 + "\n")
    
    # 运行一次分析
    state = run_system()
    
    if state:
        print(format_output(state))
    else:
        print("❌ 获取数据失败")
    
    # 持续运行模式
    print("\n按 Ctrl+C 停止...\n")
    
    try:
        while True:
            state = run_system()
            if state:
                # 简洁输出
                signal_emoji = "🟢" if state.signal == Signal.LONG else "🔴" if state.signal == Signal.SHORT else "⚪"
                print(f"[{state.timestamp[-8:]}] {state.symbol} ${state.price:,.2f} | {signal_emoji} {state.signal.value} | 信心:{state.probabilities['confidence']:.0f} | 风险:{state.risk.score}")
            
            time.sleep(10)  # 10秒刷新一次
            
    except KeyboardInterrupt:
        print("\n\n程序已停止")
