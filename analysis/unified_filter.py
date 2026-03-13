"""
统一过滤器 (Unified Filter)
==========================
整合 MetaFilter + HardRiskFilter + RiskMatrix 关键检查

核心原则：
1. 单一入口 - 所有过滤检查集中在一处
2. 去除重复 - 每项检查只做一次
3. 清晰优先级 - 关键检查优先
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FilterResult(Enum):
    """过滤结果"""
    PASS = "pass"
    WARNING = "warning"
    BLOCK = "block"


@dataclass
class FilterDecision:
    """过滤决策"""
    result: FilterResult
    signal: str
    confidence: float
    block_reason: str = ""
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def is_trading_allowed(self) -> bool:
        return self.result != FilterResult.BLOCK


class UnifiedFilter:
    """
    统一过滤器
    
    整合检查项：
    1. 数据质量 (data_quality)
    2. 风险分数 (risk_score)
    3. 信号置信度 (confidence)
    4. 系统可靠度 (reliability)
    5. 盈亏比 (risk_reward)
    6. Regime 对齐 (regime_alignment)
    
    阈值设计（渐进式）：
    - 积累阶段：低阈值（允许更多信号）
    - 成熟阶段：高阈值（严格风控）
    """
    
    # === 核心阈值 ===
    # 数据质量
    MIN_DATA_QUALITY = 0.7
    
    # 置信度
    MIN_CONFIDENCE = 20            # 最低置信度
    HIGH_CONFIDENCE_OVERRIDE = 50  # 高置信度覆盖阈值
    
    # 可靠度
    MIN_RELIABILITY = 25           # 最低可靠度
    
    # 风险
    MAX_RISK_SCORE = 70            # 最高风险分数
    MIN_RISK_REWARD = 1.2          # 最小盈亏比
    
    # 信号质量
    MIN_CONSISTENCY = 0.25         # 最小一致性
    
    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: 严格模式，适用于系统成熟后
        """
        self.strict_mode = strict_mode
        if strict_mode:
            self.MIN_CONFIDENCE = 35
            self.MIN_RELIABILITY = 40
            self.MIN_CONSISTENCY = 0.4
    
    def check_all(
        self,
        signal: str,
        confidence: float,
        data_quality: float,
        risk_score: float,
        reliability: Dict,
        risk_reward: float,
        consistency: float,
        regime: str = "neutral",
        regime_confidence: float = 0.5,
    ) -> FilterDecision:
        """
        执行所有过滤检查
        
        检查顺序（按优先级）：
        1. 数据质量 - 最基础
        2. 风险分数 - 安全第一
        3. 置信度 + 可靠度 - 核心判断
        4. 盈亏比 - 收益评估
        5. 一致性 - 信号质量
        """
        block_reasons = []
        warnings = []
        
        # === 1. 数据质量检查 ===
        if data_quality < self.MIN_DATA_QUALITY:
            block_reasons.append(f"数据质量={data_quality*100:.0f}% < {self.MIN_DATA_QUALITY*100:.0f}%")
        
        # === 2. 风险分数检查 ===
        if risk_score > self.MAX_RISK_SCORE:
            block_reasons.append(f"风险分数={risk_score:.0f} > {self.MAX_RISK_SCORE}")
        
        # === 3. 置信度 + 可靠度检查 ===
        # 高置信度覆盖机制
        reliability_score = reliability.get('score', 0)
        
        if confidence >= self.HIGH_CONFIDENCE_OVERRIDE:
            # 高置信度信号，降低可靠度要求
            if reliability_score < self.MIN_RELIABILITY:
                warnings.append(f"高置信度({confidence:.0f}%)覆盖：可靠度{reliability_score}偏低")
        else:
            # 正常检查
            if confidence < self.MIN_CONFIDENCE:
                block_reasons.append(f"置信度={confidence:.0f}% < {self.MIN_CONFIDENCE}%")
            
            if reliability_score < self.MIN_RELIABILITY:
                block_reasons.append(f"可靠度={reliability_score} < {self.MIN_RELIABILITY}")
        
        # === 4. 盈亏比检查（仅交易信号）===
        if signal != "HOLD":
            if risk_reward < self.MIN_RISK_REWARD:
                warnings.append(f"盈亏比={risk_reward:.2f} < {self.MIN_RISK_REWARD}")
        
        # === 5. 信号一致性检查 ===
        if consistency < self.MIN_CONSISTENCY and signal != "HOLD":
            warnings.append(f"信号一致性={consistency*100:.0f}% 偏低")
        
        # === 6. Regime 对齐检查（仅警告）===
        if signal != "HOLD" and regime != "neutral":
            # 检查信号与市场状态是否对齐
            if "trend_up" in regime and signal == "SHORT":
                warnings.append(f"逆势做空 (Regime={regime})")
            elif "trend_down" in regime and signal == "LONG":
                warnings.append(f"逆势做多 (Regime={regime})")
        
        # === 确定结果 ===
        if block_reasons:
            result = FilterResult.BLOCK
            block_reason = "; ".join(block_reasons)
            logger.warning(f"🚫 信号被阻止: {block_reason}")
        elif warnings:
            result = FilterResult.WARNING
            block_reason = ""
            logger.info(f"⚠️ 信号警告: {'; '.join(warnings)}")
        else:
            result = FilterResult.PASS
            block_reason = ""
            logger.info(f"✅ 信号通过过滤 (置信度={confidence:.0f}%)")
        
        return FilterDecision(
            result=result,
            signal=signal,
            confidence=confidence,
            block_reason=block_reason,
            warnings=warnings,
        )
    
    def quick_check(
        self,
        confidence: float,
        reliability_score: float,
    ) -> bool:
        """
        快速检查 - 用于提前判断
        
        仅检查核心指标：
        - 置信度
        - 可靠度
        
        Returns:
            True = 可能通过，False = 必定阻止
        """
        if confidence >= self.HIGH_CONFIDENCE_OVERRIDE:
            return True
        return confidence >= self.MIN_CONFIDENCE and reliability_score >= self.MIN_RELIABILITY
