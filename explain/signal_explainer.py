# explain/signal_explainer.py
"""
模块 35/36: AI决策解释层
========================
信号来源说明 & 市场结构分析
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class SignalSource(Enum):
    """信号来源"""
    TECHNICAL = "技术面"
    MOMENTUM = "动量面"
    VOLUME = "成交量"
    ORDERBOOK = "订单簿"
    FUNDING = "资金费率"
    CHAIN = "链上数据"
    SENTIMENT = "市场情绪"
    COMBINED = "综合信号"


@dataclass
class SignalExplanation:
    """信号解释"""
    signal: str
    confidence: float
    primary_source: SignalSource
    contributing_factors: List[str]
    market_structure: str
    reasoning: str
    warnings: List[str]


class SignalExplainer:
    """
    信号解释器
    
    功能：
    - 解释信号来源
    - 分析市场结构
    - 提供决策理由
    """
    
    def __init__(self):
        self.factor_weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'orderbook': 0.20,
            'mean_reversion': 0.20,
        }
    
    def explain(self, 
                signal: str,
                probabilities: Dict[str, float],
                indicators: Dict[str, Any],
                orderbook_analysis: Optional[Dict] = None,
                funding_analysis: Optional[Dict] = None) -> SignalExplanation:
        """
        解释信号
        
        Args:
            signal: 信号类型 (LONG/SHORT/HOLD)
            probabilities: 概率字典
            indicators: 技术指标
            orderbook_analysis: 订单簿分析
            funding_analysis: 资金费率分析
            
        Returns:
            SignalExplanation
        """
        # 收集贡献因素
        contributing_factors = []
        factor_scores = {}
        
        # 1. 趋势分析
        trend_factor = self._analyze_trend_factor(indicators)
        contributing_factors.extend(trend_factor['reasons'])
        factor_scores['trend'] = trend_factor['score']
        
        # 2. 动量分析
        momentum_factor = self._analyze_momentum_factor(indicators)
        contributing_factors.extend(momentum_factor['reasons'])
        factor_scores['momentum'] = momentum_factor['score']
        
        # 3. 成交量分析
        volume_factor = self._analyze_volume_factor(indicators)
        contributing_factors.extend(volume_factor['reasons'])
        factor_scores['volume'] = volume_factor['score']
        
        # 4. 订单簿分析
        if orderbook_analysis:
            ob_factor = self._analyze_orderbook_factor(orderbook_analysis)
            contributing_factors.extend(ob_factor['reasons'])
            factor_scores['orderbook'] = ob_factor['score']
        
        # 5. 均值回归分析
        mr_factor = self._analyze_mean_reversion_factor(indicators)
        contributing_factors.extend(mr_factor['reasons'])
        factor_scores['mean_reversion'] = mr_factor['score']
        
        # 确定主要来源
        primary_source = self._determine_primary_source(factor_scores)
        
        # 市场结构分析
        market_structure = self._analyze_market_structure(indicators)
        
        # 生成推理
        reasoning = self._generate_reasoning(signal, contributing_factors, factor_scores)
        
        # 警告
        warnings = self._generate_warnings(indicators, orderbook_analysis, funding_analysis)
        
        return SignalExplanation(
            signal=signal,
            confidence=probabilities.get('confidence', 0),
            primary_source=primary_source,
            contributing_factors=contributing_factors[:5],
            market_structure=market_structure,
            reasoning=reasoning,
            warnings=warnings
        )
    
    def _analyze_trend_factor(self, indicators: Dict) -> Dict:
        """分析趋势因素"""
        reasons = []
        score = 0.5
        
        ma5 = indicators.get('ma5', 0)
        ma20 = indicators.get('ma20', 0)
        ma60 = indicators.get('ma60', 0)
        close = indicators.get('close', 0)
        
        # 多头排列
        if ma5 > ma20 > ma60:
            reasons.append("MA多头排列")
            score += 0.3
        elif ma5 < ma20 < ma60:
            reasons.append("MA空头排列")
            score -= 0.3
        
        # 价格与均线关系
        if close > ma20:
            reasons.append("价格在MA20上方")
            score += 0.1
        else:
            reasons.append("价格在MA20下方")
            score -= 0.1
        
        return {'reasons': reasons, 'score': max(0, min(1, score))}
    
    def _analyze_momentum_factor(self, indicators: Dict) -> Dict:
        """分析动量因素"""
        reasons = []
        score = 0.5
        
        rsi = indicators.get('rsi14', 50)
        macd = indicators.get('macd', 0)
        
        # RSI
        if rsi < 30:
            reasons.append(f"RSI超卖({rsi:.0f})")
            score += 0.2
        elif rsi > 70:
            reasons.append(f"RSI超买({rsi:.0f})")
            score -= 0.2
        
        # MACD
        if macd > 0:
            reasons.append("MACD正值")
            score += 0.1
        else:
            reasons.append("MACD负值")
            score -= 0.1
        
        return {'reasons': reasons, 'score': max(0, min(1, score))}
    
    def _analyze_volume_factor(self, indicators: Dict) -> Dict:
        """分析成交量因素"""
        reasons = []
        score = 0.5
        
        vol_ratio = indicators.get('volume_ratio', 1)
        
        if vol_ratio > 2:
            reasons.append(f"放量({vol_ratio:.1f}倍)")
            score += 0.2
        elif vol_ratio < 0.5:
            reasons.append(f"缩量({vol_ratio:.1f}倍)")
        
        return {'reasons': reasons, 'score': score}
    
    def _analyze_orderbook_factor(self, orderbook: Dict) -> Dict:
        """分析订单簿因素"""
        reasons = []
        score = 0.5
        
        imbalance = orderbook.get('imbalance', 0)
        
        if imbalance > 0.3:
            reasons.append("买盘优势明显")
            score += 0.2
        elif imbalance < -0.3:
            reasons.append("卖盘优势明显")
            score -= 0.2
        
        return {'reasons': reasons, 'score': max(0, min(1, score))}
    
    def _analyze_mean_reversion_factor(self, indicators: Dict) -> Dict:
        """分析均值回归因素"""
        reasons = []
        score = 0.5
        
        bb_pos = indicators.get('bb_position', 0.5)
        
        if bb_pos < 0.2:
            reasons.append("接近布林带下轨")
            score += 0.2
        elif bb_pos > 0.8:
            reasons.append("接近布林带上轨")
            score -= 0.2
        
        return {'reasons': reasons, 'score': max(0, min(1, score))}
    
    def _determine_primary_source(self, factor_scores: Dict) -> SignalSource:
        """确定主要信号来源"""
        if not factor_scores:
            return SignalSource.COMBINED
        
        # 找出最极端的分数
        max_factor = max(factor_scores.items(), key=lambda x: abs(x[1] - 0.5))
        factor_name = max_factor[0]
        
        source_mapping = {
            'trend': SignalSource.TECHNICAL,
            'momentum': SignalSource.MOMENTUM,
            'volume': SignalSource.VOLUME,
            'orderbook': SignalSource.ORDERBOOK,
            'mean_reversion': SignalSource.TECHNICAL,
        }
        
        return source_mapping.get(factor_name, SignalSource.COMBINED)
    
    def _analyze_market_structure(self, indicators: Dict) -> str:
        """分析市场结构"""
        ma5 = indicators.get('ma5', 0)
        ma20 = indicators.get('ma20', 0)
        ma60 = indicators.get('ma60', 0)
        close = indicators.get('close', 0)
        
        if ma5 > ma20 > ma60:
            if close > ma5:
                return "多头趋势，强势上涨"
            else:
                return "多头趋势，回调整理"
        elif ma5 < ma20 < ma60:
            if close < ma5:
                return "空头趋势，弱势下跌"
            else:
                return "空头趋势，反弹修正"
        else:
            return "震荡整理，方向不明"
    
    def _generate_reasoning(self, signal: str, factors: List[str], scores: Dict) -> str:
        """生成推理说明"""
        if signal == "LONG":
            return f"综合多方因素：{', '.join(factors[:3])}。建议做多。"
        elif signal == "SHORT":
            return f"综合空方因素：{', '.join(factors[:3])}。建议做空。"
        else:
            return "多空因素交织，信号不明确，建议观望。"
    
    def _generate_warnings(self, indicators, orderbook, funding) -> List[str]:
        """生成警告"""
        warnings = []
        
        # RSI警告
        rsi = indicators.get('rsi14', 50)
        if rsi > 80:
            warnings.append("RSI严重超买，注意回调风险")
        elif rsi < 20:
            warnings.append("RSI严重超卖，注意反弹风险")
        
        # 订单簿警告
        if orderbook:
            for fake in orderbook.get('fake_orders', []):
                warnings.append(f"检测到{fake.get('type', '异常订单')}")
        
        # 资金费率警告
        if funding:
            if funding.get('warning_level') == 'high':
                warnings.append(funding.get('description', '资金费率异常'))
        
        return warnings[:3]


def explain_signal(signal: str, probabilities: Dict, indicators: Dict, 
                   orderbook: Optional[Dict] = None, funding: Optional[Dict] = None) -> Dict[str, Any]:
    """快速解释信号"""
    explainer = SignalExplainer()
    result = explainer.explain(signal, probabilities, indicators, orderbook, funding)
    
    return {
        'signal': result.signal,
        'confidence': result.confidence,
        'primary_source': result.primary_source.value,
        'factors': result.contributing_factors,
        'market_structure': result.market_structure,
        'reasoning': result.reasoning,
        'warnings': result.warnings
    }
