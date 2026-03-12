# analysis/trade_plan.py
"""
交易计划生成器
==============
根据信号和技术指标生成入场、止损、止盈价格
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from analysis.decision_maker import Signal


@dataclass
class TradePlan:
    """交易计划"""
    signal: str
    entry: float
    stop_loss: float
    take_profit_1: float  # 第一目标
    take_profit_2: float  # 第二目标
    take_profit_3: float  # 第三目标
    risk_reward: float    # 风险收益比
    position_size: float  # 建议仓位比例
    leverage: int         # 建议杠杆


def generate_trade_plan(df: pd.DataFrame, signal: Signal, account_balance: float = 10000) -> Optional[TradePlan]:
    """
    生成交易计划
    
    Args:
        df: 带技术指标的 DataFrame
        signal: 交易信号
        account_balance: 账户余额
        
    Returns:
        TradePlan 或 None (如果是观望信号)
    """
    if signal == Signal.HOLD:
        return None
    
    if df is None or len(df) < 30:
        return None
    
    latest = df.iloc[-1]
    close = latest['close']
    atr = latest.get('atr14', close * 0.02)
    
    # 计算支撑/阻力位
    high_20 = df['high'].tail(20).max()
    low_20 = df['low'].tail(20).min()
    
    if signal == Signal.LONG:
        # 做多
        entry = close  # 或稍低于当前价
        
        # 止损: 下方 ATR * 1.5 或前低
        stop_loss = min(low_20, close - atr * 1.5)
        
        # 止盈: 上方阻力位或 ATR 倍数
        take_profit_1 = close + atr * 1.5
        take_profit_2 = min(high_20, close + atr * 3)
        take_profit_3 = high_20
        
    else:  # SHORT
        # 做空
        entry = close  # 或稍高于当前价
        
        # 止损: 上方 ATR * 1.5 或前高
        stop_loss = max(high_20, close + atr * 1.5)
        
        # 止盈: 下方支撑位或 ATR 倍数
        take_profit_1 = close - atr * 1.5
        take_profit_2 = max(low_20, close - atr * 3)
        take_profit_3 = low_20
    
    # 风险收益比计算
    risk = abs(entry - stop_loss)
    reward_1 = abs(take_profit_1 - entry)
    risk_reward = reward_1 / (risk + 1e-8)
    
    # 仓位计算 (风险 2% 原则)
    risk_percent = 0.02
    max_loss = account_balance * risk_percent
    position_size = max_loss / (risk + 1e-8)
    position_ratio = min(position_size / account_balance, 0.3)  # 最大 30%
    
    # 杠杆建议
    if risk_reward >= 3:
        leverage = 1
    elif risk_reward >= 2:
        leverage = 2
    else:
        leverage = 3
    
    return TradePlan(
        signal=signal.value,
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit_1=round(take_profit_1, 2),
        take_profit_2=round(take_profit_2, 2),
        take_profit_3=round(take_profit_3, 2),
        risk_reward=round(risk_reward, 2),
        position_size=round(position_ratio * 100, 1),
        leverage=leverage
    )


def format_trade_plan(plan: TradePlan) -> str:
    """格式化交易计划输出"""
    if plan is None:
        return "当前建议观望，无交易计划"
    
    lines = [
        f"{'='*50}",
        f"📋 交易计划 ({plan.signal})",
        f"{'='*50}",
        f"入场价格: ${plan.entry:,.2f}",
        f"止损价格: ${plan.stop_loss:,.2f}",
        f"",
        f"目标 1 (保守): ${plan.take_profit_1:,.2f}",
        f"目标 2 (适中): ${plan.take_profit_2:,.2f}",
        f"目标 3 (激进): ${plan.take_profit_3:,.2f}",
        f"",
        f"风险收益比: {plan.risk_reward:.2f}",
        f"建议仓位: {plan.position_size:.1f}%",
        f"建议杠杆: {plan.leverage}x",
        f"{'='*50}",
    ]
    
    return "\n".join(lines)


# 测试
if __name__ == "__main__":
    from data.market_stream import get_realtime_eth_data
    from data.kline_builder import calculate_indicators
    from ai.probability_model import calculate_probabilities
    from analysis.decision_maker import make_decision
    
    df, price = get_realtime_eth_data()
    
    if df is not None:
        df = calculate_indicators(df)
        probs = calculate_probabilities(df)
        decision = make_decision(probs)
        
        if decision.signal != Signal.HOLD:
            plan = generate_trade_plan(df, decision.signal)
            print(format_trade_plan(plan))
        else:
            print("当前建议观望")
