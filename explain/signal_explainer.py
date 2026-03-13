# explain/signal_explainer.py
"""
模块 35/36: AI决策解释层
========================
信号来源说明 & 市场结构分析

CRITICAL FIX: 接入Feature Sync，确保解释与Signal Engine一致
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class SignalSource(Enum):
    """信号来源"""
    TECHNICAL = "技术面"
    MOMENTUM = "动量面"
    VOLUME = "成交量"
    ORDERBOOK = "订单簿"
    FUNDING = "资金费率"
    CHAIN = "链上数据"
    SENTIMENT = "市场情绪"
    ORDER_FLOW = "订单流"
    LIQUIDATION = "清算"
    REGIME = "市场状态"
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
    
    # 新增：与Signal Engine一致性
    regime_alignment: str = "unknown"  # "aligned" / "conflict" / "neutral"
    data_freshness: float = 1.0  # 数据新鲜度


class SignalExplainer:
    """
    信号解释器 (重构版)
    
    核心改进：
    1. 使用Feature Sync的特征数据
    2. 与Signal Engine保持一致
    3. 检测信号冲突
    """
    
    def __init__(self):
        self.factor_weights = {
            'hurst': 0.15,
            'regime': 0.15,
            'order_flow': 0.15,
            'orderbook': 0.12,
            'whale': 0.10,
            'sentiment': 0.10,
            'funding': 0.08,
            'liquidation': 0.08,
            'momentum': 0.07,
        }
    
    def explain_from_sync(
        self,
        signal: str,
        probabilities: Dict[str, float],
        feature_matrix: Any,  # FeatureMatrix
        unified_signal: Dict[str, Any],
        regime_data: Dict[str, Any],
    ) -> SignalExplanation:
        """
        从同步特征矩阵解释信号
        
        这是推荐的方法，确保解释与Signal Engine一致
        
        Args:
            signal: 信号类型 (LONG/SHORT/HOLD)
            probabilities: 概率字典
            feature_matrix: 特征同步层的FeatureMatrix
            unified_signal: Signal Engine的统一信号
            regime_data: 市场状态数据
        """
        contributing_factors = []
        factor_scores = {}
        warnings = []
        
        # 1. 从Feature Matrix提取特征
        features = feature_matrix.to_dict() if hasattr(feature_matrix, 'to_dict') else {}
        
        # 2. Hurst分析 (来自Feature Sync)
        hurst = features.get('hurst', 0.5)
        hurst_factor = self._analyze_hurst(hurst)
        contributing_factors.extend(hurst_factor['reasons'])
        factor_scores['hurst'] = hurst_factor['score']
        
        # 3. 市场状态分析 (来自Regime Engine)
        regime = regime_data.get('regime', 'neutral')
        regime_factor = self._analyze_regime(regime, regime_data.get('confidence', 0.5))
        contributing_factors.extend(regime_factor['reasons'])
        factor_scores['regime'] = regime_factor['score']
        
        # 4. 订单流分析 (来自Feature Sync)
        cvd = features.get('cvd', 0)
        order_flow_factor = self._analyze_order_flow(cvd, unified_signal.get('votes', {}))
        contributing_factors.extend(order_flow_factor['reasons'])
        factor_scores['order_flow'] = order_flow_factor['score']
        
        # 5. 订单簿分析 (来自Feature Sync)
        imbalance = features.get('orderbook_imbalance', 0)
        ob_factor = self._analyze_orderbook_imbalance(imbalance)
        contributing_factors.extend(ob_factor['reasons'])
        factor_scores['orderbook'] = ob_factor['score']
        
        # 6. 巨鲸流动分析 (来自Feature Sync)
        whale_flow = features.get('whale_flow', 0)
        whale_factor = self._analyze_whale(whale_flow)
        contributing_factors.extend(whale_factor['reasons'])
        factor_scores['whale'] = whale_factor['score']
        
        # 7. 情绪分析 (来自Feature Sync)
        sentiment = features.get('sentiment', 50)
        sentiment_factor = self._analyze_sentiment(sentiment)
        contributing_factors.extend(sentiment_factor['reasons'])
        factor_scores['sentiment'] = sentiment_factor['score']
        
        # 8. 资金费率分析 (来自Feature Sync)
        funding_rate = features.get('funding_rate', 0)
        funding_factor = self._analyze_funding(funding_rate)
        contributing_factors.extend(funding_factor['reasons'])
        factor_scores['funding'] = funding_factor['score']
        
        # 9. 清算分析 (来自Feature Sync)
        liquidation = features.get('liquidation', 1.0)
        liq_factor = self._analyze_liquidation(liquidation)
        contributing_factors.extend(liq_factor['reasons'])
        factor_scores['liquidation'] = liq_factor['score']
        
        # 10. 动量分析 (来自Feature Sync)
        momentum = features.get('momentum', 0)
        mom_factor = self._analyze_momentum(momentum)
        contributing_factors.extend(mom_factor['reasons'])
        factor_scores['momentum'] = mom_factor['score']
        
        # 确定主要来源
        primary_source = self._determine_primary_source(factor_scores, signal)
        
        # 市场结构分析 (使用Regime而非旧MA)
        market_structure = self._analyze_market_structure_v2(regime, hurst, imbalance)
        
        # 检测信号与Regime冲突
        regime_alignment = self._check_regime_alignment(signal, regime)
        if regime_alignment == "conflict":
            warnings.append(f"⚠️ 信号与市场状态({regime})冲突")
        
        # 生成推理
        reasoning = self._generate_reasoning_v2(
            signal, 
            contributing_factors, 
            factor_scores,
            unified_signal,
            regime_alignment
        )
        
        # 添加系统警告
        warnings.extend(self._generate_system_warnings(unified_signal, feature_matrix))
        
        # 数据新鲜度
        data_freshness = feature_matrix.feature_completeness if hasattr(feature_matrix, 'feature_completeness') else 1.0
        
        return SignalExplanation(
            signal=signal,
            confidence=probabilities.get('confidence', 0),
            primary_source=primary_source,
            contributing_factors=contributing_factors[:6],
            market_structure=market_structure,
            reasoning=reasoning,
            warnings=warnings[:4],
            regime_alignment=regime_alignment,
            data_freshness=data_freshness,
        )
    
    def _analyze_hurst(self, hurst: float) -> Dict:
        """分析Hurst指数"""
        reasons = []
        score = 0.5
        
        if hurst > 0.55:
            reasons.append(f"Hurst={hurst:.3f}趋势向上")
            score = 0.7 + (hurst - 0.55) * 0.5
        elif hurst < 0.45:
            reasons.append(f"Hurst={hurst:.3f}均值回归")
            score = 0.3 - (0.45 - hurst) * 0.5
        else:
            reasons.append(f"Hurst={hurst:.3f}随机游走")
            score = 0.5
        
        return {'reasons': reasons, 'score': min(1, max(0, score))}
    
    def _analyze_regime(self, regime: str, confidence: float) -> Dict:
        """分析市场状态"""
        reasons = []
        score = 0.5
        
        regime_text = {
            "trend_up": "强势上涨",
            "trend_down": "强势下跌",
            "range": "震荡区间",
            "volatile": "高波动",
            "panic": "恐慌状态",
            "euphoria": "狂热状态",
        }
        
        regime_name = regime_text.get(regime, regime)
        
        if "trend_up" in regime:
            reasons.append(f"市场状态: {regime_name}")
            score = 0.75
        elif "trend_down" in regime:
            reasons.append(f"市场状态: {regime_name}")
            score = 0.25
        elif "panic" in regime:
            reasons.append("市场恐慌")
            score = 0.65  # 恐慌时可能抄底
        elif "euphoria" in regime:
            reasons.append("市场狂热")
            score = 0.35  # 狂热时可能反转
        else:
            reasons.append(f"市场状态: {regime_name}")
        
        return {'reasons': reasons, 'score': score * confidence + 0.5 * (1 - confidence)}
    
    def _analyze_order_flow(self, cvd: float, votes: Dict) -> Dict:
        """分析订单流"""
        reasons = []
        score = 0.5
        
        if cvd > 10:
            reasons.append(f"CVD={cvd:.1f}买方主导")
            score = 0.7
        elif cvd < -10:
            reasons.append(f"CVD={cvd:.1f}卖方主导")
            score = 0.3
        else:
            reasons.append(f"CVD={cvd:.1f}买卖均衡")
        
        # 投票信息
        long_votes = votes.get('long', 0)
        short_votes = votes.get('short', 0)
        if long_votes > short_votes:
            reasons.append(f"模块投票: ↑{long_votes} ↓{short_votes}")
        elif short_votes > long_votes:
            reasons.append(f"模块投票: ↓{short_votes} ↑{long_votes}")
        
        return {'reasons': reasons, 'score': score}
    
    def _analyze_orderbook_imbalance(self, imbalance: float) -> Dict:
        """分析订单簿失衡"""
        reasons = []
        score = 0.5
        
        if imbalance > 0.3:
            reasons.append(f"订单簿失衡={imbalance:.2f}买盘优势")
            score = 0.7
        elif imbalance < -0.3:
            reasons.append(f"订单簿失衡={imbalance:.2f}卖盘优势")
            score = 0.3
        else:
            reasons.append(f"订单簿均衡={imbalance:.2f}")
        
        return {'reasons': reasons, 'score': score}
    
    def _analyze_whale(self, whale_flow: float) -> Dict:
        """分析巨鲸流动"""
        reasons = []
        score = 0.5
        
        if whale_flow < -500:
            reasons.append(f"巨鲸净流出{abs(whale_flow):.0f}ETH")
            score = 0.7  # 流出交易所 = 看涨
        elif whale_flow > 500:
            reasons.append(f"巨鲸净流入{whale_flow:.0f}ETH")
            score = 0.3  # 流入交易所 = 看跌
        else:
            reasons.append(f"巨鲸流动正常")
        
        return {'reasons': reasons, 'score': score}
    
    def _analyze_sentiment(self, sentiment: float) -> Dict:
        """分析社交情绪"""
        reasons = []
        score = 0.5
        
        if sentiment > 70:
            reasons.append(f"情绪过热={sentiment:.0f}")
            score = 0.35  # 反向
        elif sentiment < 30:
            reasons.append(f"情绪恐慌={sentiment:.0f}")
            score = 0.65  # 反向
        else:
            reasons.append(f"情绪中性={sentiment:.0f}")
        
        return {'reasons': reasons, 'score': score}
    
    def _analyze_funding(self, funding_rate: float) -> Dict:
        """分析资金费率"""
        reasons = []
        score = 0.5
        
        rate_pct = funding_rate * 100
        
        if abs(rate_pct) > 0.05:
            if rate_pct > 0:
                reasons.append(f"资金费率={rate_pct:.4f}%做多拥挤")
                score = 0.35  # 高费率看空
            else:
                reasons.append(f"资金费率={rate_pct:.4f}%做空拥挤")
                score = 0.65  # 负费率看多
        else:
            reasons.append(f"资金费率正常")
        
        return {'reasons': reasons, 'score': score}
    
    def _analyze_liquidation(self, liq_ratio: float) -> Dict:
        """分析清算数据"""
        reasons = []
        score = 0.5
        
        if liq_ratio > 1.5:
            reasons.append(f"多头清算聚集")
            score = 0.35  # 多头清算风险
        elif liq_ratio < 0.67:
            reasons.append(f"空头清算聚集")
            score = 0.65  # 空头清算风险
        else:
            reasons.append(f"清算均衡")
        
        return {'reasons': reasons, 'score': score}
    
    def _analyze_momentum(self, momentum: float) -> Dict:
        """分析动量"""
        reasons = []
        score = 0.5
        
        if momentum > 3:
            reasons.append(f"动量={momentum:.1f}强势向上")
            score = 0.7
        elif momentum < -3:
            reasons.append(f"动量={momentum:.1f}强势向下")
            score = 0.3
        else:
            reasons.append(f"动量={momentum:.1f}平稳")
        
        return {'reasons': reasons, 'score': score}
    
    def _check_regime_alignment(self, signal: str, regime: str) -> str:
        """检查信号与市场状态对齐"""
        if signal == "HOLD":
            return "neutral"
        
        if "trend_up" in regime:
            return "aligned" if signal == "LONG" else "conflict"
        elif "trend_down" in regime:
            return "aligned" if signal == "SHORT" else "conflict"
        elif "panic" in regime:
            return "aligned" if signal == "LONG" else "neutral"  # 恐慌时做多
        elif "euphoria" in regime:
            return "aligned" if signal == "SHORT" else "neutral"  # 狂热时做空
        
        return "neutral"
    
    def _determine_primary_source(self, factor_scores: Dict, signal: str) -> SignalSource:
        """确定主要信号来源"""
        if not factor_scores:
            return SignalSource.COMBINED
        
        # 根据信号方向找出最支持的因子
        if signal == "LONG":
            best_factor = max(factor_scores.items(), key=lambda x: x[1])
        elif signal == "SHORT":
            best_factor = min(factor_scores.items(), key=lambda x: x[1])
        else:
            return SignalSource.COMBINED
        
        factor_name = best_factor[0]
        
        source_mapping = {
            'hurst': SignalSource.TECHNICAL,
            'regime': SignalSource.REGIME,
            'order_flow': SignalSource.ORDER_FLOW,
            'orderbook': SignalSource.ORDERBOOK,
            'whale': SignalSource.CHAIN,
            'sentiment': SignalSource.SENTIMENT,
            'funding': SignalSource.FUNDING,
            'liquidation': SignalSource.LIQUIDATION,
            'momentum': SignalSource.MOMENTUM,
        }
        
        return source_mapping.get(factor_name, SignalSource.COMBINED)
    
    def _analyze_market_structure_v2(self, regime: str, hurst: float, imbalance: float) -> str:
        """
        分析市场结构 (重构版)
        使用Regime而非旧的MA计算
        """
        regime_text = {
            "trend_up": "多头趋势，强势上涨",
            "trend_down": "空头趋势，弱势下跌",
            "range": "震荡整理，方向不明",
            "volatile": "高波动状态",
            "liquidation": "清算事件",
            "accumulation": "吸筹阶段",
            "distribution": "派发阶段",
            "panic": "恐慌抛售",
            "euphoria": "狂热追涨",
            "neutral": "中性状态",
        }
        
        base = regime_text.get(regime, "分析中")
        
        # 补充Hurst信息
        if hurst > 0.6:
            base += "，趋势持续性强"
        elif hurst < 0.4:
            base += "，均值回归特征"
        
        return base
    
    def _generate_reasoning_v2(
        self,
        signal: str,
        factors: List[str],
        scores: Dict,
        unified_signal: Dict,
        regime_alignment: str,
    ) -> str:
        """生成推理说明 (重构版)"""
        
        meta_passed = unified_signal.get('meta_filter', {}).get('passed', True)
        consistency = unified_signal.get('quality_metrics', {}).get('consistency', 0)
        
        if signal == "LONG":
            if not meta_passed:
                return f"信号被Meta Filter拦截: {unified_signal.get('meta_filter', {}).get('reason', '质量不足')}"
            elif regime_alignment == "conflict":
                return f"做多信号与市场状态冲突，建议谨慎。支持因素: {', '.join(factors[:2])}"
            else:
                return f"多方信号一致性强({consistency*100:.0f}%)，建议做多。关键因素: {', '.join(factors[:3])}"
        
        elif signal == "SHORT":
            if not meta_passed:
                return f"信号被Meta Filter拦截: {unified_signal.get('meta_filter', {}).get('reason', '质量不足')}"
            elif regime_alignment == "conflict":
                return f"做空信号与市场状态冲突，建议谨慎。支持因素: {', '.join(factors[:2])}"
            else:
                return f"空方信号一致性强({consistency*100:.0f}%)，建议做空。关键因素: {', '.join(factors[:3])}"
        
        else:
            return "多空信号交织，一致性不足，建议观望等待明确信号。"
    
    def _generate_system_warnings(self, unified_signal: Dict, feature_matrix: Any) -> List[str]:
        """生成系统警告"""
        warnings = []
        
        # Meta Filter警告
        meta = unified_signal.get('meta_filter', {})
        if not meta.get('passed', True):
            warnings.append(f"⚠️ Meta Filter: {meta.get('reason', '信号被过滤')}")
        
        # 一致性警告
        consistency = unified_signal.get('quality_metrics', {}).get('consistency', 1)
        if consistency < 0.5:
            warnings.append(f"⚠️ 信号一致性低({consistency*100:.0f}%)")
        
        # 数据质量警告
        freshness = feature_matrix.feature_completeness if hasattr(feature_matrix, 'feature_completeness') else 1.0
        if freshness < 0.8:
            warnings.append(f"⚠️ 数据完整度低({freshness*100:.0f}%)")
        
        return warnings
    
    # 保持向后兼容
    def explain(self, 
                signal: str,
                probabilities: Dict[str, float],
                indicators: Dict[str, Any],
                orderbook_analysis: Optional[Dict] = None,
                funding_analysis: Optional[Dict] = None) -> SignalExplanation:
        """
        解释信号 (旧接口，保持兼容)
        不推荐使用，请使用 explain_from_sync
        """
        # 简化实现，从indicators提取
        contributing_factors = []
        
        # 趋势分析
        ma5 = indicators.get('ma5', 0)
        ma20 = indicators.get('ma20', 0)
        if ma5 > ma20:
            contributing_factors.append("MA多头排列")
        elif ma5 < ma20:
            contributing_factors.append("MA空头排列")
        
        # RSI
        rsi = indicators.get('rsi14', 50)
        if rsi < 30:
            contributing_factors.append(f"RSI超卖({rsi:.0f})")
        elif rsi > 70:
            contributing_factors.append(f"RSI超买({rsi:.0f})")
        
        # 市场结构
        if ma5 > ma20:
            market_structure = "多头趋势"
        else:
            market_structure = "空头趋势"
        
        # 推理
        if signal == "LONG":
            reasoning = f"建议做多。因素: {', '.join(contributing_factors[:2])}"
        elif signal == "SHORT":
            reasoning = f"建议做空。因素: {', '.join(contributing_factors[:2])}"
        else:
            reasoning = "建议观望。"
        
        return SignalExplanation(
            signal=signal,
            confidence=probabilities.get('confidence', 0),
            primary_source=SignalSource.COMBINED,
            contributing_factors=contributing_factors,
            market_structure=market_structure,
            reasoning=reasoning,
            warnings=[],
        )


def explain_signal(
    signal: str, 
    probabilities: Dict, 
    indicators: Dict, 
    orderbook: Optional[Dict] = None, 
    funding: Optional[Dict] = None,
    feature_matrix: Any = None,
    unified_signal: Dict = None,
    regime_data: Dict = None,
) -> Dict[str, Any]:
    """
    解释信号 (统一接口)
    
    推荐使用新参数:
    - feature_matrix: 特征同步矩阵
    - unified_signal: Signal Engine输出
    - regime_data: 市场状态数据
    """
    explainer = SignalExplainer()
    
    # 如果提供了新参数，使用新方法
    if feature_matrix is not None and unified_signal is not None:
        result = explainer.explain_from_sync(
            signal=signal,
            probabilities=probabilities,
            feature_matrix=feature_matrix,
            unified_signal=unified_signal,
            regime_data=regime_data or {},
        )
    else:
        # 向后兼容
        result = explainer.explain(signal, probabilities, indicators, orderbook, funding)
    
    return {
        'signal': result.signal,
        'confidence': result.confidence,
        'primary_source': result.primary_source.value,
        'factors': result.contributing_factors,
        'market_structure': result.market_structure,
        'reasoning': result.reasoning,
        'warnings': result.warnings,
        'regime_alignment': result.regime_alignment,
        'data_freshness': result.data_freshness,
    }
