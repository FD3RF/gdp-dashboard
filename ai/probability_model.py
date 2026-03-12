# ai/probability_model.py
"""
AI 概率计算模型
================
基于技术指标计算多/空/观望概率
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ProbabilityResult:
    """概率结果"""
    long: float      # 做多概率
    short: float     # 做空概率
    hold: float      # 观望概率
    confidence: float  # 信心指数
    signals: Dict[str, Any]  # 信号详情


def calculate_ma_signal(df: pd.DataFrame) -> tuple:
    """均线信号"""
    latest = df.iloc[-1]
    
    score = 0
    reason = []
    
    # 多头排列: MA5 > MA10 > MA20 > MA60
    if latest['ma5'] > latest['ma10'] > latest['ma20']:
        score += 30
        reason.append("多头排列")
    elif latest['ma5'] < latest['ma10'] < latest['ma20']:
        score -= 30
        reason.append("空头排列")
    
    # 价格与均线关系
    if latest['close'] > latest['ma20']:
        score += 10
        reason.append("价格在MA20上方")
    else:
        score -= 10
        reason.append("价格在MA20下方")
    
    return score, reason


def calculate_rsi_signal(df: pd.DataFrame) -> tuple:
    """RSI 信号"""
    latest = df.iloc[-1]
    rsi = latest.get('rsi14', 50)
    
    score = 0
    reason = []
    
    if rsi < 30:
        score += 40
        reason.append(f"RSI超卖({rsi:.1f})")
    elif rsi > 70:
        score -= 40
        reason.append(f"RSI超买({rsi:.1f})")
    elif rsi < 40:
        score += 15
        reason.append(f"RSI偏低({rsi:.1f})")
    elif rsi > 60:
        score -= 15
        reason.append(f"RSI偏高({rsi:.1f})")
    
    return score, reason


def calculate_macd_signal(df: pd.DataFrame) -> tuple:
    """MACD 信号"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    reason = []
    
    # 金叉/死叉
    if prev['macd'] < prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
        score += 30
        reason.append("MACD金叉")
    elif prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
        score -= 30
        reason.append("MACD死叉")
    
    # MACD 柱状图方向
    if latest['macd_histogram'] > 0:
        score += 10
        reason.append("MACD柱正值")
    else:
        score -= 10
        reason.append("MACD柱负值")
    
    return score, reason


def calculate_bb_signal(df: pd.DataFrame) -> tuple:
    """布林带信号"""
    latest = df.iloc[-1]
    
    score = 0
    reason = []
    
    bb_pos = latest.get('bb_position', 0.5)
    
    if bb_pos < 0.1:
        score += 35
        reason.append(f"触及下轨({bb_pos:.2f})")
    elif bb_pos > 0.9:
        score -= 35
        reason.append(f"触及上轨({bb_pos:.2f})")
    elif bb_pos < 0.3:
        score += 15
        reason.append(f"接近下轨({bb_pos:.2f})")
    elif bb_pos > 0.7:
        score -= 15
        reason.append(f"接近上轨({bb_pos:.2f})")
    
    return score, reason


def calculate_volume_signal(df: pd.DataFrame) -> tuple:
    """成交量信号"""
    latest = df.iloc[-1]
    
    score = 0
    reason = []
    
    vol_ratio = latest.get('volume_ratio', 1)
    
    if vol_ratio > 2:
        score += 20 if latest['close'] > df.iloc[-2]['close'] else -20
        reason.append(f"放量({vol_ratio:.1f}倍)")
    elif vol_ratio < 0.5:
        reason.append(f"缩量({vol_ratio:.1f}倍)")
    
    return score, reason


def calculate_momentum_signal(df: pd.DataFrame) -> tuple:
    """动量信号"""
    latest = df.iloc[-1]
    
    score = 0
    reason = []
    
    momentum = latest.get('momentum', 0)
    roc = latest.get('roc', 0)
    
    if momentum > 0 and roc > 2:
        score += 15
        reason.append(f"上涨动量({roc:.1f}%)")
    elif momentum < 0 and roc < -2:
        score -= 15
        reason.append(f"下跌动量({roc:.1f}%)")
    
    return score, reason


def calculate_probabilities(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算多/空/观望概率
    
    Args:
        df: 带技术指标的 DataFrame
        
    Returns:
        包含 long, short, hold 概率的字典
    """
    if df is None or len(df) < 60:
        return {'long': 33.3, 'short': 33.3, 'hold': 33.4, 'confidence': 0, 'signals': {}}
    
    signals = {}
    total_score = 0
    all_reasons = []
    
    # 计算各指标信号
    score, reason = calculate_ma_signal(df)
    total_score += score
    all_reasons.extend(reason)
    signals['ma'] = {'score': score, 'reason': reason}
    
    score, reason = calculate_rsi_signal(df)
    total_score += score
    all_reasons.extend(reason)
    signals['rsi'] = {'score': score, 'reason': reason}
    
    score, reason = calculate_macd_signal(df)
    total_score += score
    all_reasons.extend(reason)
    signals['macd'] = {'score': score, 'reason': reason}
    
    score, reason = calculate_bb_signal(df)
    total_score += score
    all_reasons.extend(reason)
    signals['bollinger'] = {'score': score, 'reason': reason}
    
    score, reason = calculate_volume_signal(df)
    total_score += score
    all_reasons.extend(reason)
    signals['volume'] = {'score': score, 'reason': reason}
    
    score, reason = calculate_momentum_signal(df)
    total_score += score
    all_reasons.extend(reason)
    signals['momentum'] = {'score': score, 'reason': reason}
    
    # 将总分转换为概率
    # 分数范围: -150 到 +150
    # 归一化到 0-100
    normalized_score = (total_score + 150) / 300  # 0-1
    
    # 计算概率
    if total_score > 30:  # 强看多
        long_prob = 40 + min(total_score, 50)
        short_prob = max(5, 30 - total_score * 0.3)
        hold_prob = 100 - long_prob - short_prob
    elif total_score < -30:  # 强看空
        short_prob = 40 + min(-total_score, 50)
        long_prob = max(5, 30 + total_score * 0.3)
        hold_prob = 100 - long_prob - short_prob
    else:  # 观望
        hold_prob = 40 + abs(total_score)
        if total_score > 0:
            long_prob = (100 - hold_prob) * 0.6
            short_prob = (100 - hold_prob) * 0.4
        else:
            long_prob = (100 - hold_prob) * 0.4
            short_prob = (100 - hold_prob) * 0.6
    
    # 确保概率在合理范围
    long_prob = max(5, min(85, long_prob))
    short_prob = max(5, min(85, short_prob))
    hold_prob = max(10, 100 - long_prob - short_prob)
    
    # 归一化
    total = long_prob + short_prob + hold_prob
    long_prob = round(long_prob / total * 100, 1)
    short_prob = round(short_prob / total * 100, 1)
    hold_prob = round(hold_prob / total * 100, 1)
    
    # 信心指数 = 最高概率 - 次高概率
    probs = [long_prob, short_prob, hold_prob]
    probs_sorted = sorted(probs, reverse=True)
    confidence = round(probs_sorted[0] - probs_sorted[1], 1)
    
    return {
        'long': long_prob,
        'short': short_prob,
        'hold': hold_prob,
        'confidence': confidence,
        'signals': {
            'total_score': total_score,
            'reasons': all_reasons,
            'details': signals
        }
    }


# 测试
if __name__ == "__main__":
    from data.market_stream import get_realtime_eth_data
    from data.kline_builder import calculate_indicators
    
    df, price = get_realtime_eth_data()
    
    if df is not None:
        df = calculate_indicators(df)
        probs = calculate_probabilities(df)
        
        print(f"\n{'='*50}")
        print(f"AI 概率分析结果")
        print(f"{'='*50}")
        print(f"做多概率: {probs['long']:.1f}%")
        print(f"做空概率: {probs['short']:.1f}%")
        print(f"观望概率: {probs['hold']:.1f}%")
        print(f"信心指数: {probs['confidence']:.1f}")
        print(f"\n信号理由:")
        for reason in probs['signals'].get('reasons', []):
            print(f"  • {reason}")
