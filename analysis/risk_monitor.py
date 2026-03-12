# analysis/risk_monitor.py
"""
风险监控模块
============
实时监控市场风险，提供预警
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """风险等级"""
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    EXTREME = "极高风险"


@dataclass
class RiskWarning:
    """风险预警"""
    level: RiskLevel
    score: int  # 风险分数 0-100
    warnings: List[str]
    suggestions: List[str]


def risk_warning(df: pd.DataFrame) -> RiskWarning:
    """
    风险预警分析
    
    Args:
        df: 带技术指标的 DataFrame
        
    Returns:
        RiskWarning 对象
    """
    if df is None or len(df) < 30:
        return RiskWarning(
            level=RiskLevel.LOW,
            score=0,
            warnings=["数据不足"],
            suggestions=["等待更多数据"]
        )
    
    warnings = []
    suggestions = []
    risk_score = 0
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. 波动率检查
    volatility = latest.get('volatility', 0)
    if volatility > 0.8:
        risk_score += 25
        warnings.append(f"⚠️ 高波动率: {volatility*100:.1f}%")
        suggestions.append("建议减小仓位")
    elif volatility > 0.5:
        risk_score += 10
        warnings.append(f"波动率偏高: {volatility*100:.1f}%")
    
    # 2. RSI 极端值
    rsi = latest.get('rsi14', 50)
    if rsi > 80:
        risk_score += 20
        warnings.append(f"⚠️ RSI 严重超买: {rsi:.1f}")
        suggestions.append("注意回调风险")
    elif rsi < 20:
        risk_score += 20
        warnings.append(f"⚠️ RSI 严重超卖: {rsi:.1f}")
        suggestions.append("注意反弹风险")
    
    # 3. 异常成交量
    vol_ratio = latest.get('volume_ratio', 1)
    if vol_ratio > 5:
        risk_score += 30
        warnings.append(f"🚨 异常放量: {vol_ratio:.1f}倍")
        suggestions.append("可能有大资金进出，谨慎操作")
    elif vol_ratio > 3:
        risk_score += 15
        warnings.append(f"放量明显: {vol_ratio:.1f}倍")
    
    # 4. 价格剧烈变动
    price_change = latest.get('price_change_5m', 0)
    if abs(price_change) > 0.05:
        risk_score += 25
        warnings.append(f"🚨 价格剧烈变动: {price_change*100:.2f}%")
        suggestions.append("市场可能异常，建议观望")
    elif abs(price_change) > 0.03:
        risk_score += 10
        warnings.append(f"价格变动较大: {price_change*100:.2f}%")
    
    # 5. 布林带突破
    bb_pos = latest.get('bb_position', 0.5)
    if bb_pos > 0.95 or bb_pos < 0.05:
        risk_score += 15
        warnings.append("价格触及布林带边界")
        suggestions.append("可能出现趋势反转或延续")
    
    # 6. ATR 扩大
    atr_current = latest.get('atr14', 0)
    atr_avg = df['atr14'].tail(20).mean()
    if atr_current > atr_avg * 1.5:
        risk_score += 15
        warnings.append("ATR 扩大，市场波动加剧")
    
    # 7. MACD 背离检查
    if len(df) >= 20:
        price_trend = df['close'].iloc[-10:].mean() - df['close'].iloc[-20:-10].mean()
        macd_trend = df['macd'].iloc[-10:].mean() - df['macd'].iloc[-20:-10].mean()
        
        if price_trend > 0 and macd_trend < 0:
            risk_score += 20
            warnings.append("⚠️ MACD 顶背离")
            suggestions.append("上涨动能减弱，注意回调")
        elif price_trend < 0 and macd_trend > 0:
            risk_score += 20
            warnings.append("⚠️ MACD 底背离")
            suggestions.append("下跌动能减弱，注意反弹")
    
    # 确定风险等级
    if risk_score >= 70:
        level = RiskLevel.EXTREME
    elif risk_score >= 50:
        level = RiskLevel.HIGH
    elif risk_score >= 25:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW
    
    # 添加默认建议
    if risk_score > 30:
        suggestions.append("建议设置严格止损")
    if risk_score > 50:
        suggestions.append("考虑减少仓位或暂时观望")
    
    # 限制分数范围
    risk_score = min(100, max(0, risk_score))
    
    return RiskWarning(
        level=level,
        score=risk_score,
        warnings=warnings,
        suggestions=suggestions
    )


def format_risk_warning(risk: RiskWarning) -> str:
    """格式化风险预警输出"""
    level_emoji = {
        RiskLevel.LOW: "🟢",
        RiskLevel.MEDIUM: "🟡",
        RiskLevel.HIGH: "🟠",
        RiskLevel.EXTREME: "🔴",
    }
    
    lines = [
        f"{'='*50}",
        f"{level_emoji.get(risk.level, '⚪')} 风险等级: {risk.level.value}",
        f"📊 风险分数: {risk.score}/100",
        f"{'='*50}",
    ]
    
    if risk.warnings:
        lines.append("⚠️ 风险提示:")
        for w in risk.warnings:
            lines.append(f"   {w}")
    
    if risk.suggestions:
        lines.append("\n💡 操作建议:")
        for s in risk.suggestions:
            lines.append(f"   • {s}")
    
    return "\n".join(lines)


# 测试
if __name__ == "__main__":
    from data.market_stream import get_realtime_eth_data
    from data.kline_builder import calculate_indicators
    
    df, price = get_realtime_eth_data()
    
    if df is not None:
        df = calculate_indicators(df)
        risk = risk_warning(df)
        print(format_risk_warning(risk))
