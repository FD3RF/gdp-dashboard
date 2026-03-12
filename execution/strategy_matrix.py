# execution/strategy_matrix.py
"""
第4层：自适应策略矩阵
====================

功能：多策略组合，动态权重调整
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch


class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = 'trend_following'      # 趋势跟踪
    MEAN_REVERSION = 'mean_reversion'        # 均值回归
    MOMENTUM = 'momentum'                    # 动量策略
    SCALPING = 'scalping'                    # 短线策略
    ARBITRAGE = 'arbitrage'                  # 套利策略
    MARKET_MAKING = 'market_making'          # 做市策略


@dataclass
class StrategySignal:
    """策略信号"""
    strategy_type: StrategyType
    action: int  # 0=多, 1=空, 2=平, 3=观望
    strength: float  # 信号强度 0-1
    confidence: float  # 信心度 0-1
    reason: str


class BaseStrategy:
    """策略基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.weight = 1.0
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
        }
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """生成信号"""
        raise NotImplementedError
    
    def update_performance(self, pnl: float, is_win: bool):
        """更新绩效"""
        self.performance['total_trades'] += 1
        if is_win:
            self.performance['winning_trades'] += 1
        self.performance['total_pnl'] += pnl
    
    def get_win_rate(self) -> float:
        """获取胜率"""
        if self.performance['total_trades'] == 0:
            return 0.5
        return self.performance['winning_trades'] / self.performance['total_trades']


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    def __init__(self):
        super().__init__("trend_following")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        # MA 多空排列
        ma5 = market_data.get('ma5', 0)
        ma20 = market_data.get('ma20', 0)
        ma60 = market_data.get('ma60', 0)
        macd_hist = market_data.get('macd_hist', 0)
        
        # 多头趋势
        if ma5 > ma20 > ma60 and macd_hist > 0:
            return StrategySignal(
                strategy_type=StrategyType.TREND_FOLLOWING,
                action=0,  # 做多
                strength=min(abs(macd_hist) / 100, 1.0),
                confidence=0.7,
                reason="多头趋势确认"
            )
        
        # 空头趋势
        elif ma5 < ma20 < ma60 and macd_hist < 0:
            return StrategySignal(
                strategy_type=StrategyType.TREND_FOLLOWING,
                action=1,  # 做空
                strength=min(abs(macd_hist) / 100, 1.0),
                confidence=0.7,
                reason="空头趋势确认"
            )
        
        return None


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def __init__(self):
        super().__init__("mean_reversion")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        rsi = market_data.get('rsi_14', 50)
        price = market_data.get('price', 0)
        bb_lower = market_data.get('bb_lower', price)
        bb_upper = market_data.get('bb_upper', price)
        
        # 超卖反弹
        if rsi < 30 and price <= bb_lower:
            return StrategySignal(
                strategy_type=StrategyType.MEAN_REVERSION,
                action=0,  # 做多
                strength=(30 - rsi) / 30,
                confidence=0.6,
                reason=f"超卖反弹 RSI={rsi:.1f}"
            )
        
        # 超买回落
        elif rsi > 70 and price >= bb_upper:
            return StrategySignal(
                strategy_type=StrategyType.MEAN_REVERSION,
                action=1,  # 做空
                strength=(rsi - 70) / 30,
                confidence=0.6,
                reason=f"超买回落 RSI={rsi:.1f}"
            )
        
        return None


class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def __init__(self):
        super().__init__("momentum")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        volume_ratio = market_data.get('volume_ratio', 1.0)
        momentum_5m = market_data.get('momentum_5m', 0)
        momentum_15m = market_data.get('momentum_15m', 0)
        
        # 放量上涨
        if volume_ratio > 1.5 and momentum_5m > 0.002 and momentum_15m > 0:
            return StrategySignal(
                strategy_type=StrategyType.MOMENTUM,
                action=0,  # 做多
                strength=min(volume_ratio / 3, 1.0),
                confidence=0.65,
                reason=f"放量上涨 量比={volume_ratio:.2f}"
            )
        
        # 放量下跌
        elif volume_ratio > 1.5 and momentum_5m < -0.002 and momentum_15m < 0:
            return StrategySignal(
                strategy_type=StrategyType.MOMENTUM,
                action=1,  # 做空
                strength=min(volume_ratio / 3, 1.0),
                confidence=0.65,
                reason=f"放量下跌 量比={volume_ratio:.2f}"
            )
        
        return None


class ScalpingStrategy(BaseStrategy):
    """短线策略"""
    
    def __init__(self):
        super().__init__("scalping")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        ob_imbalance = market_data.get('orderbook_imbalance', 0.5)
        spread = market_data.get('spread_pct', 0)
        
        # 订单簿不平衡 + 低波动
        if spread < 0.001:  # 点差小于 0.1%
            if ob_imbalance > 0.65:
                return StrategySignal(
                    strategy_type=StrategyType.SCALPING,
                    action=0,
                    strength=(ob_imbalance - 0.5) * 2,
                    confidence=0.55,
                    reason=f"买单占优 {ob_imbalance*100:.1f}%"
                )
            elif ob_imbalance < 0.35:
                return StrategySignal(
                    strategy_type=StrategyType.SCALPING,
                    action=1,
                    strength=(0.5 - ob_imbalance) * 2,
                    confidence=0.55,
                    reason=f"卖单占优 {(1-ob_imbalance)*100:.1f}%"
                )
        
        return None


class StrategyMatrix:
    """
    自适应策略矩阵
    
    功能：
    1. 多策略组合
    2. 动态权重调整
    3. 信号融合
    4. 绩效追踪
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 初始化策略
        self.strategies: List[BaseStrategy] = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy(),
            ScalpingStrategy(),
        ]
        
        # 权重配置
        self.base_weights = {
            StrategyType.TREND_FOLLOWING: 0.35,
            StrategyType.MEAN_REVERSION: 0.25,
            StrategyType.MOMENTUM: 0.25,
            StrategyType.SCALPING: 0.15,
        }
        
        # 动态权重
        self.dynamic_weights = self.base_weights.copy()
        
        # 市场状态
        self.market_regime = 'neutral'  # trending, ranging, volatile
        
        # 统计
        self.stats = {
            'total_signals': 0,
            'signals_by_strategy': {},
        }
    
    def detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """检测市场状态"""
        volatility = market_data.get('volatility_1h', 0)
        momentum = abs(market_data.get('momentum_1h', 0))
        adx = market_data.get('adx', 25)
        
        # 高波动
        if volatility > 0.03:
            self.market_regime = 'volatile'
        # 趋势市
        elif adx > 25 and momentum > 0.001:
            self.market_regime = 'trending'
        # 震荡市
        else:
            self.market_regime = 'ranging'
        
        return self.market_regime
    
    def adjust_weights(self):
        """根据市场状态调整权重"""
        if self.market_regime == 'trending':
            # 趋势市增加趋势策略权重
            self.dynamic_weights[StrategyType.TREND_FOLLOWING] = 0.5
            self.dynamic_weights[StrategyType.MOMENTUM] = 0.3
            self.dynamic_weights[StrategyType.MEAN_REVERSION] = 0.1
            self.dynamic_weights[StrategyType.SCALPING] = 0.1
        
        elif self.market_regime == 'ranging':
            # 震荡市增加均值回归权重
            self.dynamic_weights[StrategyType.MEAN_REVERSION] = 0.4
            self.dynamic_weights[StrategyType.SCALPING] = 0.3
            self.dynamic_weights[StrategyType.TREND_FOLLOWING] = 0.2
            self.dynamic_weights[StrategyType.MOMENTUM] = 0.1
        
        elif self.market_regime == 'volatile':
            # 高波动降低短线策略权重
            self.dynamic_weights[StrategyType.TREND_FOLLOWING] = 0.3
            self.dynamic_weights[StrategyType.MOMENTUM] = 0.3
            self.dynamic_weights[StrategyType.MEAN_REVERSION] = 0.25
            self.dynamic_weights[StrategyType.SCALPING] = 0.15
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """生成所有策略信号"""
        # 检测市场状态
        self.detect_market_regime(market_data)
        
        # 调整权重
        self.adjust_weights()
        
        signals = []
        for strategy in self.strategies:
            signal = strategy.generate_signal(market_data)
            if signal:
                signals.append(signal)
                self.stats['total_signals'] += 1
                self.stats['signals_by_strategy'][signal.strategy_type.value] = \
                    self.stats['signals_by_strategy'].get(signal.strategy_type.value, 0) + 1
        
        return signals
    
    def fuse_signals(self, signals: List[StrategySignal]) -> Tuple[int, float, str]:
        """
        融合多个策略信号
        
        Args:
            signals: 信号列表
            
        Returns:
            (最终动作, 信心度, 原因)
        """
        if not signals:
            return 3, 0.0, "无信号"  # 观望
        
        # 计算加权投票
        action_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        reasons = []
        
        for signal in signals:
            weight = self.dynamic_weights.get(signal.strategy_type, 0.25)
            score = signal.strength * signal.confidence * weight
            action_scores[signal.action] += score
            reasons.append(f"{signal.strategy_type.value}:{signal.reason}")
        
        # 选择得分最高的动作
        best_action = max(action_scores, key=action_scores.get)
        confidence = action_scores[best_action] / sum(action_scores.values()) if sum(action_scores.values()) > 0 else 0
        
        return best_action, confidence, " | ".join(reasons[:3])
    
    def update_strategy_performance(
        self, 
        strategy_type: StrategyType, 
        pnl: float, 
        is_win: bool
    ):
        """更新策略绩效"""
        for strategy in self.strategies:
            if hasattr(strategy, 'strategy_type') and strategy.strategy_type == strategy_type:
                strategy.update_performance(pnl, is_win)
                break
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'market_regime': self.market_regime,
            'dynamic_weights': {k.value: v for k, v in self.dynamic_weights.items()},
            'strategies': [
                {
                    'name': s.name,
                    'weight': s.weight,
                    'win_rate': s.get_win_rate(),
                    'total_pnl': s.performance['total_pnl'],
                }
                for s in self.strategies
            ],
            'stats': self.stats,
        }


# 导出
__all__ = [
    'StrategyMatrix',
    'StrategyType',
    'StrategySignal',
    'BaseStrategy',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'ScalpingStrategy',
]
