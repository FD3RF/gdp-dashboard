"""
硬规则风险过滤器 (Risk Filter)
================================
CRITICAL: 在交易执行前的最后检查

规则：
- RR < 1.2 → 禁止交易
- 系统可靠度 < 40 → 禁止交易
- 数据质量 < 0.7 → 禁止交易
- Meta Filter 未通过 → 禁止交易
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FilterResult(Enum):
    """过滤结果"""
    PASS = "pass"           # 通过
    WARNING = "warning"     # 警告但允许
    BLOCK = "block"         # 禁止交易


@dataclass
class RiskFilterDecision:
    """风险过滤决策"""
    result: FilterResult
    signal: str
    reasons: List[str]
    block_reason: str = ""
    
    # 过滤详情
    rr_check: str = "N/A"
    reliability_check: str = "N/A"
    data_quality_check: str = "N/A"
    meta_filter_check: str = "N/A"
    regime_alignment_check: str = "N/A"
    
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def is_trading_allowed(self) -> bool:
        """是否允许交易"""
        return self.result != FilterResult.BLOCK
    
    def to_dict(self) -> Dict:
        return {
            "result": self.result.value,
            "signal": self.signal,
            "reasons": self.reasons,
            "block_reason": self.block_reason,
            "checks": {
                "risk_reward": self.rr_check,
                "reliability": self.reliability_check,
                "data_quality": self.data_quality_check,
                "meta_filter": self.meta_filter_check,
                "regime_alignment": self.regime_alignment_check,
            },
            "is_allowed": self.is_trading_allowed(),
            "timestamp": self.timestamp.isoformat(),
        }


class HardRiskFilter:
    """
    硬规则风险过滤器
    
    在交易执行前进行最终检查
    任何一项不满足都会阻止交易
    """
    
    # 硬规则阈值
    MIN_RISK_REWARD = 1.2          # 最小风险收益比
    MIN_RELIABILITY = 40           # 最小系统可靠度
    MIN_DATA_QUALITY = 0.7         # 最小数据质量
    MIN_CONFIDENCE = 45            # 最小置信度
    MIN_CONSISTENCY = 0.4          # 最小信号一致性
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: 严格模式，所有规则都必须通过
        """
        self.strict_mode = strict_mode
        self.filter_history: List[RiskFilterDecision] = []
    
    def check_all(
        self,
        signal: str,
        trade_plan: Any,
        reliability: Dict,
        data_quality_score: float,
        unified_signal: Dict,
        probabilities: Dict,
    ) -> RiskFilterDecision:
        """
        执行所有检查
        
        Args:
            signal: 信号 (LONG/SHORT/HOLD)
            trade_plan: 交易计划
            reliability: 系统可靠度
            data_quality_score: 数据质量分数
            unified_signal: Signal Engine输出
            probabilities: 概率分布
        
        Returns:
            RiskFilterDecision
        """
        if signal == "HOLD":
            return RiskFilterDecision(
                result=FilterResult.PASS,
                signal=signal,
                reasons=["观望信号，无需过滤"],
            )
        
        reasons = []
        block_reasons = []
        
        # 1. 风险收益比检查
        rr_check = self._check_risk_reward(trade_plan)
        if rr_check['status'] == 'fail':
            block_reasons.append(rr_check['reason'])
        elif rr_check['status'] == 'warn':
            reasons.append(rr_check['reason'])
        
        # 2. 系统可靠度检查
        rel_check = self._check_reliability(reliability)
        if rel_check['status'] == 'fail':
            block_reasons.append(rel_check['reason'])
        
        # 3. 数据质量检查
        dq_check = self._check_data_quality(data_quality_score)
        if dq_check['status'] == 'fail':
            block_reasons.append(dq_check['reason'])
        
        # 4. Meta Filter检查
        mf_check = self._check_meta_filter(unified_signal)
        if mf_check['status'] == 'fail':
            block_reasons.append(mf_check['reason'])
        
        # 5. 置信度检查
        conf_check = self._check_confidence(probabilities)
        if conf_check['status'] == 'fail':
            block_reasons.append(conf_check['reason'])
        
        # 6. 信号一致性检查
        cons_check = self._check_consistency(unified_signal)
        if cons_check['status'] == 'warn':
            reasons.append(cons_check['reason'])
        
        # 确定结果
        if block_reasons:
            result = FilterResult.BLOCK
            block_reason = "; ".join(block_reasons)
        elif reasons:
            result = FilterResult.WARNING
            block_reason = ""
        else:
            result = FilterResult.PASS
            block_reason = ""
        
        decision = RiskFilterDecision(
            result=result,
            signal=signal,
            reasons=reasons,
            block_reason=block_reason,
            rr_check=rr_check['reason'],
            reliability_check=rel_check['reason'],
            data_quality_check=dq_check['reason'],
            meta_filter_check=mf_check['reason'],
            regime_alignment_check=cons_check['reason'],
        )
        
        self.filter_history.append(decision)
        
        # 日志
        if result == FilterResult.BLOCK:
            logger.warning(f"🚫 交易被阻止: {block_reason}")
        elif result == FilterResult.WARNING:
            logger.info(f"⚠️ 交易警告: {'; '.join(reasons)}")
        else:
            logger.info(f"✅ 交易通过所有检查")
        
        return decision
    
    def _check_risk_reward(self, trade_plan) -> Dict:
        """检查风险收益比"""
        if trade_plan is None:
            return {'status': 'fail', 'reason': '无交易计划'}
        
        rr = getattr(trade_plan, 'risk_reward', 0)
        
        if rr < self.MIN_RISK_REWARD:
            return {
                'status': 'fail',
                'reason': f"RR={rr:.2f} < {self.MIN_RISK_REWARD}，风险收益比过低"
            }
        elif rr < 1.5:
            return {
                'status': 'warn',
                'reason': f"RR={rr:.2f}，略低于推荐值1.5"
            }
        else:
            return {
                'status': 'pass',
                'reason': f"RR={rr:.2f} ✓"
            }
    
    def _check_reliability(self, reliability: Dict) -> Dict:
        """检查系统可靠度"""
        score = reliability.get('score', 0)
        win_rate = reliability.get('win_rate', 0) * 100
        
        if score < self.MIN_RELIABILITY:
            return {
                'status': 'fail',
                'reason': f"系统可靠度={score} < {self.MIN_RELIABILITY}，历史胜率{win_rate:.1f}%"
            }
        elif score < 50:
            return {
                'status': 'warn',
                'reason': f"系统可靠度={score}偏低"
            }
        else:
            return {
                'status': 'pass',
                'reason': f"系统可靠度={score} ✓"
            }
    
    def _check_data_quality(self, quality_score: float) -> Dict:
        """检查数据质量"""
        if quality_score < self.MIN_DATA_QUALITY:
            return {
                'status': 'fail',
                'reason': f"数据质量={quality_score*100:.0f}% < {self.MIN_DATA_QUALITY*100:.0f}%"
            }
        elif quality_score < 0.85:
            return {
                'status': 'warn',
                'reason': f"数据质量={quality_score*100:.0f}%略低"
            }
        else:
            return {
                'status': 'pass',
                'reason': f"数据质量={quality_score*100:.0f}% ✓"
            }
    
    def _check_meta_filter(self, unified_signal: Dict) -> Dict:
        """检查Meta Filter"""
        meta = unified_signal.get('meta_filter', {})
        passed = meta.get('passed', True)
        reason = meta.get('reason', '')
        
        if not passed:
            return {
                'status': 'fail',
                'reason': f"Meta Filter未通过: {reason}"
            }
        else:
            return {
                'status': 'pass',
                'reason': f"Meta Filter通过 ✓"
            }
    
    def _check_confidence(self, probabilities: Dict) -> Dict:
        """检查置信度"""
        confidence = probabilities.get('confidence', 0)
        
        if confidence < self.MIN_CONFIDENCE:
            return {
                'status': 'fail',
                'reason': f"置信度={confidence:.1f} < {self.MIN_CONFIDENCE}"
            }
        else:
            return {
                'status': 'pass',
                'reason': f"置信度={confidence:.1f} ✓"
            }
    
    def _check_consistency(self, unified_signal: Dict) -> Dict:
        """检查信号一致性"""
        quality = unified_signal.get('quality_metrics', {})
        consistency = quality.get('consistency', 1)
        
        if consistency < self.MIN_CONSISTENCY:
            return {
                'status': 'warn',
                'reason': f"信号一致性={consistency*100:.0f}%偏低"
            }
        else:
            return {
                'status': 'pass',
                'reason': f"信号一致性={consistency*100:.0f}% ✓"
            }
    
    def get_filter_stats(self) -> Dict:
        """获取过滤统计"""
        if not self.filter_history:
            return {"total": 0}
        
        total = len(self.filter_history)
        passed = sum(1 for d in self.filter_history if d.result == FilterResult.PASS)
        warned = sum(1 for d in self.filter_history if d.result == FilterResult.WARNING)
        blocked = sum(1 for d in self.filter_history if d.result == FilterResult.BLOCK)
        
        return {
            "total": total,
            "passed": passed,
            "warned": warned,
            "blocked": blocked,
            "pass_rate": passed / total if total > 0 else 0,
            "block_rate": blocked / total if total > 0 else 0,
        }


# 全局实例
_risk_filter: Optional[HardRiskFilter] = None


def get_risk_filter() -> HardRiskFilter:
    """获取全局风险过滤器"""
    global _risk_filter
    if _risk_filter is None:
        _risk_filter = HardRiskFilter()
    return _risk_filter


def apply_hard_filter(
    signal: str,
    trade_plan: Any,
    reliability: Dict,
    data_quality_score: float,
    unified_signal: Dict,
    probabilities: Dict,
) -> RiskFilterDecision:
    """
    应用硬规则过滤 (便捷函数)
    """
    rf = get_risk_filter()
    return rf.check_all(
        signal=signal,
        trade_plan=trade_plan,
        reliability=reliability,
        data_quality_score=data_quality_score,
        unified_signal=unified_signal,
        probabilities=probabilities,
    )
