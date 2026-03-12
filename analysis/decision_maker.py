# analysis/decision_maker.py
"""
交易决策器
==========
根据概率生成最终的交易信号
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """交易信号"""
    LONG = "LONG"    # 做多
    SHORT = "SHORT"  # 做空
    HOLD = "HOLD"    # 观望
    CLOSE = "CLOSE"  # 平仓


@dataclass
class Decision:
    """决策结果"""
    signal: Signal
    probability: float
    confidence: float
    reasons: list
    risk_level: str  # low, medium, high


def make_decision(probabilities: Dict[str, Any]) -> Decision:
    """
    根据概率生成最终的交易信号
    
    Args:
        probabilities: 包含 long, short, hold 概率的字典
        
    Returns:
        Decision 对象
    """
    long_p = probabilities.get('long', 33.3)
    short_p = probabilities.get('short', 33.3)
    hold_p = probabilities.get('hold', 33.4)
    confidence = probabilities.get('confidence', 0)
    signals = probabilities.get('signals', {})
    reasons = signals.get('reasons', [])
    
    # 核心逻辑：谁大听谁的
    if hold_p > long_p and hold_p > short_p:
        signal = Signal.HOLD
        prob = hold_p
    elif long_p > short_p:
        signal = Signal.LONG
        prob = long_p
    else:
        signal = Signal.SHORT
        prob = short_p
    
    # 风险等级评估
    if confidence >= 30:
        risk_level = "low"
    elif confidence >= 15:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    # 特殊情况处理
    total_score = signals.get('total_score', 0)
    
    # 如果信号不明确，建议观望
    if confidence < 10:
        signal = Signal.HOLD
        prob = hold_p
        reasons.append("信号不明确，建议观望")
    
    # 如果多空概率接近，降低风险
    if abs(long_p - short_p) < 5:
        risk_level = "high"
        reasons.append("多空分歧较大，风险升高")
    
    return Decision(
        signal=signal,
        probability=prob,
        confidence=confidence,
        reasons=reasons,
        risk_level=risk_level
    )


def get_signal_emoji(signal: Signal) -> str:
    """获取信号对应的 emoji"""
    emojis = {
        Signal.LONG: "🟢",
        Signal.SHORT: "🔴",
        Signal.HOLD: "⚪",
        Signal.CLOSE: "🔵",
    }
    return emojis.get(signal, "⚪")


def format_decision(decision: Decision) -> str:
    """格式化决策输出"""
    emoji = get_signal_emoji(decision.signal)
    lines = [
        f"{emoji} 决策: {decision.signal.value}",
        f"📊 概率: {decision.probability:.1f}%",
        f"💪 信心: {decision.confidence:.1f}",
        f"⚠️ 风险: {decision.risk_level}",
        f"📝 理由:",
    ]
    
    for reason in decision.reasons[:5]:  # 最多显示5条
        lines.append(f"   • {reason}")
    
    return "\n".join(lines)


# 测试
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        {'long': 60, 'short': 20, 'hold': 20, 'confidence': 40, 'signals': {'reasons': ['RSI超卖', 'MACD金叉']}},
        {'long': 25, 'short': 55, 'hold': 20, 'confidence': 30, 'signals': {'reasons': ['RSI超买', 'MACD死叉']}},
        {'long': 30, 'short': 30, 'hold': 40, 'confidence': 10, 'signals': {'reasons': ['信号不明确']}},
    ]
    
    for i, probs in enumerate(test_cases, 1):
        print(f"\n{'='*40}")
        print(f"测试用例 {i}")
        print(f"{'='*40}")
        decision = make_decision(probs)
        print(format_decision(decision))
