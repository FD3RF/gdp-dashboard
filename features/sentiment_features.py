"""
社交情绪特征工程模块 (Layer 5)
将社交情绪数据转换为量化特征
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class SentimentFeatures:
    """社交情绪特征"""
    # 基础情绪分数 (0-100)
    overall_score: float
    
    # 趋势指标
    momentum: float  # 情绪动量
    trend: str  # rising/falling/stable
    
    # 极端情绪检测
    is_extreme: bool
    extreme_type: Optional[str]  # "euphoria" / "panic"
    
    # 置信度
    confidence: float
    
    # 多空力量
    bullish_power: float
    bearish_power: float
    dominance: str  # "bulls" / "bears" / "neutral"
    
    # 历史对比
    percentile_24h: float  # 当前情绪在24小时内的百分位
    
    # 预测因子
    reversal_probability: float  # 情绪反转概率


class SentimentFeatureEngine:
    """
    社交情绪特征工程引擎
    将原始情绪数据转换为可交易的量化特征
    """
    
    def __init__(self):
        self.history: List[Dict] = []
        self.max_history = 1000
    
    def update_history(self, sentiment_data: Dict) -> None:
        """更新历史记录"""
        record = {
            "timestamp": datetime.now(),
            "score": sentiment_data.get("overall_sentiment", {}).get("score", 50),
            "bullish_ratio": sentiment_data.get("overall_sentiment", {}).get("bullish_ratio", 0.5),
            "sample_size": sentiment_data.get("overall_sentiment", {}).get("sample_size", 0),
        }
        self.history.append(record)
        
        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def calculate_momentum(self, window: int = 10) -> float:
        """
        计算情绪动量
        正值 = 情绪改善，负值 = 情绪恶化
        """
        if len(self.history) < 2:
            return 0.0
        
        recent = self.history[-window:] if len(self.history) >= window else self.history
        
        if len(recent) < 2:
            return 0.0
        
        # 计算变化率
        changes = []
        for i in range(1, len(recent)):
            change = recent[i]["score"] - recent[i-1]["score"]
            changes.append(change)
        
        momentum = np.mean(changes) if changes else 0.0
        return round(momentum, 2)
    
    def detect_extreme_sentiment(self, score: float) -> tuple:
        """
        检测极端情绪
        
        Returns:
            (is_extreme, extreme_type)
        """
        is_extreme = False
        extreme_type = None
        
        if score >= 80:
            is_extreme = True
            extreme_type = "euphoria"  # 极度乐观 - 可能是反向指标
        elif score <= 20:
            is_extreme = True
            extreme_type = "panic"  # 极度恐慌 - 可能是买入机会
        
        return is_extreme, extreme_type
    
    def calculate_percentile(self, score: float, hours: int = 24) -> float:
        """计算当前分数在历史中的百分位"""
        cutoff = datetime.now() - timedelta(hours=hours)
        historical_scores = [
            h["score"] for h in self.history 
            if h["timestamp"] >= cutoff
        ]
        
        if not historical_scores:
            return 50.0
        
        historical_scores.append(score)
        historical_scores.sort()
        
        percentile = (historical_scores.index(score) / len(historical_scores)) * 100
        return round(percentile, 1)
    
    def calculate_reversal_probability(
        self,
        score: float,
        momentum: float,
        is_extreme: bool
    ) -> float:
        """
        计算情绪反转概率
        
        当情绪极端且动量减弱时，反转概率高
        """
        prob = 0.0
        
        # 极端情绪基础概率
        if is_extreme:
            prob += 0.3
        
        # 动量背离
        if score > 70 and momentum < 0:
            # 高情绪 + 负动量 = 反转概率增加
            prob += 0.3
        elif score < 30 and momentum > 0:
            # 低情绪 + 正动量 = 反转概率增加
            prob += 0.3
        
        # 情绪极端程度
        extremeness = abs(score - 50) / 50
        prob += extremeness * 0.2
        
        return min(1.0, prob)
    
    def extract_features(self, sentiment_data: Dict) -> SentimentFeatures:
        """
        从原始情绪数据提取特征
        """
        overall = sentiment_data.get("overall_sentiment", {})
        
        score = overall.get("score", 50)
        bullish_ratio = overall.get("bullish_ratio", 0.5)
        confidence = overall.get("confidence", 0.5)
        sample_size = overall.get("sample_size", 0)
        
        # 更新历史
        self.update_history(sentiment_data)
        
        # 计算特征
        momentum = self.calculate_momentum()
        is_extreme, extreme_type = self.detect_extreme_sentiment(score)
        percentile = self.calculate_percentile(score)
        reversal_prob = self.calculate_reversal_probability(score, momentum, is_extreme)
        
        # 判断趋势
        if momentum > 1:
            trend = "rising"
        elif momentum < -1:
            trend = "falling"
        else:
            trend = "stable"
        
        # 多空力量
        bullish_power = bullish_ratio * 100
        bearish_power = (1 - bullish_ratio) * 100
        
        if bullish_ratio > 0.6:
            dominance = "bulls"
        elif bullish_ratio < 0.4:
            dominance = "bears"
        else:
            dominance = "neutral"
        
        # 调整置信度
        adjusted_confidence = confidence * min(1.0, sample_size / 30)
        
        return SentimentFeatures(
            overall_score=score,
            momentum=momentum,
            trend=trend,
            is_extreme=is_extreme,
            extreme_type=extreme_type,
            confidence=adjusted_confidence,
            bullish_power=bullish_power,
            bearish_power=bearish_power,
            dominance=dominance,
            percentile_24h=percentile,
            reversal_probability=reversal_prob
        )
    
    def to_feature_vector(self, features: SentimentFeatures) -> np.ndarray:
        """
        将特征转换为向量（用于 ML 模型输入）
        """
        # 极端情绪编码
        extreme_encoding = {
            None: 0,
            "euphoria": 1,
            "panic": -1
        }
        
        # 趋势编码
        trend_encoding = {
            "rising": 1,
            "stable": 0,
            "falling": -1
        }
        
        # 主导力量编码
        dominance_encoding = {
            "bulls": 1,
            "neutral": 0,
            "bears": -1
        }
        
        vector = np.array([
            features.overall_score / 100,  # 归一化到 0-1
            features.momentum / 10,  # 归一化
            trend_encoding[features.trend],
            1 if features.is_extreme else 0,
            extreme_encoding[features.extreme_type],
            features.confidence,
            features.bullish_power / 100,
            features.bearish_power / 100,
            dominance_encoding[features.dominance],
            features.percentile_24h / 100,
            features.reversal_probability,
        ])
        
        return vector


def get_sentiment_features(sentiment_data: Optional[Dict] = None) -> SentimentFeatures:
    """
    获取社交情绪特征（便捷函数）
    """
    from data.social_stream import social_sentiment_score
    
    if sentiment_data is None:
        sentiment_data = social_sentiment_score()
    
    engine = SentimentFeatureEngine()
    return engine.extract_features(sentiment_data)
