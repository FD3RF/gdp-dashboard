"""
在线学习进化模块 (Layer 11)
信号历史验证 + 动态权重更新 + 策略进化
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from decimal import Decimal
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """信号记录"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal: str  # "LONG", "SHORT", "HOLD"
    entry_price: Decimal
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    
    # 概率分布
    long_prob: float
    short_prob: float
    hold_prob: float
    
    # 特征状态
    features: Dict[str, float]
    
    # 结果（事后填写）
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[Decimal] = None
    pnl_pct: Optional[float] = None
    hit_sl: bool = False
    hit_tp: bool = False
    hold_duration: Optional[int] = None  # 分钟


@dataclass
class WeightAdjustment:
    """权重调整记录"""
    timestamp: datetime
    feature_name: str
    old_weight: float
    new_weight: float
    adjustment_reason: str
    performance_delta: float


class SignalPerformanceTracker:
    """
    信号性能追踪器
    记录每次信号的结果并计算性能指标
    """
    
    def __init__(self, storage_path: str = ".signal_history.json"):
        self.storage_path = storage_path
        self.signals: List[SignalRecord] = []
        self.load_history()
    
    def load_history(self) -> None:
        """加载历史记录"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # 兼容两种格式：数组格式 [] 或对象格式 {"signals": []}
                if isinstance(data, list):
                    items = data
                else:
                    items = data.get("signals", [])
                
                for item in items:
                    try:
                        record = SignalRecord(
                            signal_id=item.get("signal_id", f"sig_{len(self.signals)}"),
                            timestamp=datetime.fromisoformat(item["timestamp"]) if item.get("timestamp") else datetime.now(),
                            symbol=item.get("symbol", "ETH/USDT"),
                            signal=item.get("signal", "HOLD"),
                            entry_price=Decimal(str(item.get("price", item.get("entry_price", 0)))),
                            stop_loss=Decimal(str(item["stop_loss"])) if item.get("stop_loss") else None,
                            take_profit=Decimal(str(item.get("take_profit", 0))) if item.get("take_profit") else None,
                            long_prob=item.get("long_prob", 33.3),
                            short_prob=item.get("short_prob", 33.3),
                            hold_prob=item.get("hold_prob", 33.4),
                            features=item.get("features", {}),
                            exit_price=Decimal(str(item["exit_price"])) if item.get("exit_price") else None,
                            exit_time=datetime.fromisoformat(item["exit_time"]) if item.get("exit_time") else None,
                            pnl=Decimal(str(item["pnl"])) if item.get("pnl") else None,
                            pnl_pct=item.get("pnl_pct"),
                            hit_sl=item.get("hit_sl", False),
                            hit_tp=item.get("hit_tp", False),
                            hold_duration=item.get("hold_duration"),
                        )
                        self.signals.append(record)
                    except Exception as e:
                        logger.debug(f"Skip invalid record: {e}")
                        continue
                        
                logger.info(f"Loaded {len(self.signals)} signal records")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
    
    def save_history(self) -> None:
        """保存历史记录"""
        try:
            data = {
                "signals": [
                    {
                        **asdict(s),
                        "timestamp": s.timestamp.isoformat(),
                        "entry_price": float(s.entry_price),
                        "stop_loss": float(s.stop_loss) if s.stop_loss else None,
                        "take_profit": float(s.take_profit) if s.take_profit else None,
                        "exit_price": float(s.exit_price) if s.exit_price else None,
                        "exit_time": s.exit_time.isoformat() if s.exit_time else None,
                        "pnl": float(s.pnl) if s.pnl else None,
                    }
                    for s in self.signals[-1000:]  # 保留最近1000条
                ],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")
    
    def record_signal(self, record: SignalRecord) -> None:
        """记录新信号"""
        self.signals.append(record)
        self.save_history()
    
    def update_signal_result(
        self,
        signal_id: str,
        exit_price: Decimal,
        exit_time: datetime,
        hit_sl: bool = False,
        hit_tp: bool = False
    ) -> None:
        """更新信号结果"""
        for signal in self.signals:
            if signal.signal_id == signal_id:
                signal.exit_price = exit_price
                signal.exit_time = exit_time
                signal.hit_sl = hit_sl
                signal.hit_tp = hit_tp
                
                # 计算盈亏
                if signal.signal == "LONG":
                    signal.pnl = exit_price - signal.entry_price
                elif signal.signal == "SHORT":
                    signal.pnl = signal.entry_price - exit_price
                else:
                    signal.pnl = Decimal("0")
                
                # 计算盈亏百分比
                if signal.entry_price > 0:
                    signal.pnl_pct = float(signal.pnl / signal.entry_price * 100)
                
                # 持续时间
                if signal.exit_time and signal.timestamp:
                    signal.hold_duration = int(
                        (signal.exit_time - signal.timestamp).total_seconds() / 60
                    )
                
                self.save_history()
                break
    
    def get_recent_performance(
        self,
        days: int = 7,
        signal_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取最近性能"""
        cutoff = datetime.now() - timedelta(days=days)
        
        filtered = [
            s for s in self.signals
            if s.timestamp >= cutoff and s.pnl is not None
        ]
        
        if signal_type:
            filtered = [s for s in filtered if s.signal == signal_type]
        
        if not filtered:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl_pct": 0,
                "profit_factor": 0,
            }
        
        wins = [s for s in filtered if s.pnl > 0]
        losses = [s for s in filtered if s.pnl < 0]
        
        total_pnl = sum(float(s.pnl) for s in filtered)
        win_pnl = sum(float(s.pnl) for s in wins)
        loss_pnl = abs(sum(float(s.pnl) for s in losses))
        
        return {
            "total_trades": len(filtered),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(filtered) if filtered else 0,
            "avg_pnl_pct": np.mean([s.pnl_pct for s in filtered if s.pnl_pct]),
            "total_pnl": total_pnl,
            "profit_factor": win_pnl / loss_pnl if loss_pnl > 0 else 0,
            "avg_hold_duration": np.mean([s.hold_duration for s in filtered if s.hold_duration]),
        }


class FeatureWeightOptimizer:
    """
    特征权重优化器
    基于历史信号结果动态调整特征权重
    """
    
    # 默认特征权重
    DEFAULT_WEIGHTS = {
        "trend": 0.25,
        "momentum": 0.20,
        "volume": 0.15,
        "orderbook": 0.20,
        "mean_reversion": 0.10,
        "sentiment": 0.05,
        "funding": 0.05,
    }
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.adjustment_history: List[WeightAdjustment] = []
    
    def calculate_feature_contribution(
        self,
        signal: SignalRecord
    ) -> Dict[str, float]:
        """
        计算每个特征对信号的贡献度
        """
        contributions = {}
        
        for feature, weight in self.weights.items():
            feature_value = signal.features.get(feature, 0)
            contribution = feature_value * weight
            contributions[feature] = contribution
        
        return contributions
    
    def update_weights(
        self,
        signals: List[SignalRecord],
        performance_window: int = 50
    ) -> Dict[str, float]:
        """
        基于最近信号结果更新权重
        
        使用在线学习：正确预测的特征权重增加，错误预测的特征权重减少
        """
        recent_signals = [s for s in signals if s.pnl is not None][-performance_window:]
        
        if len(recent_signals) < 10:
            return self.weights
        
        # 计算每个特征的有效性
        feature_effectiveness = defaultdict(list)
        
        for signal in recent_signals:
            contributions = self.calculate_feature_contribution(signal)
            
            # 判断信号是否正确
            is_correct = signal.pnl > 0
            
            for feature, contribution in contributions.items():
                # 如果贡献方向与结果一致，给予正向反馈
                if is_correct and contribution > 0:
                    feature_effectiveness[feature].append(1)
                elif not is_correct and contribution > 0:
                    feature_effectiveness[feature].append(-1)
                else:
                    feature_effectiveness[feature].append(0)
        
        # 更新权重
        for feature, effectiveness in feature_effectiveness.items():
            if effectiveness:
                avg_effectiveness = np.mean(effectiveness)
                adjustment = self.learning_rate * avg_effectiveness
                
                old_weight = self.weights.get(feature, 0.1)
                new_weight = max(0.01, min(0.5, old_weight + adjustment))
                
                if abs(new_weight - old_weight) > 0.001:
                    self.weights[feature] = new_weight
                    
                    self.adjustment_history.append(WeightAdjustment(
                        timestamp=datetime.now(),
                        feature_name=feature,
                        old_weight=old_weight,
                        new_weight=new_weight,
                        adjustment_reason=f"Average effectiveness: {avg_effectiveness:.3f}",
                        performance_delta=adjustment,
                    ))
        
        # 归一化权重
        total = sum(self.weights.values())
        for feature in self.weights:
            self.weights[feature] /= total
        
        return self.weights
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性排名"""
        return dict(sorted(
            self.weights.items(),
            key=lambda x: x[1],
            reverse=True
        ))


class StrategyEvolver:
    """
    策略进化器
    实现策略参数的日常微调和进化
    """
    
    def __init__(self):
        # 可进化参数
        self.parameters = {
            "confidence_threshold": 0.6,  # 信心阈值
            "stop_loss_atr_multiplier": 2.0,  # 止损ATR倍数
            "take_profit_ratio": 1.5,  # 止盈比例
            "max_position_size": 0.1,  # 最大仓位
            "holding_period_max": 60,  # 最大持仓分钟
            "risk_per_trade": 0.02,  # 单笔风险
        }
        
        # 参数边界
        self.param_bounds = {
            "confidence_threshold": (0.5, 0.8),
            "stop_loss_atr_multiplier": (1.0, 4.0),
            "take_profit_ratio": (1.0, 3.0),
            "max_position_size": (0.05, 0.2),
            "holding_period_max": (30, 240),
            "risk_per_trade": (0.01, 0.05),
        }
        
        # 进化历史
        self.evolution_history: List[Dict] = []
    
    def mutate_parameters(
        self,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1
    ) -> Dict[str, Any]:
        """
        参数变异（类似基因突变）
        
        Args:
            mutation_rate: 变异概率
            mutation_strength: 变异强度
        
        Returns:
            变异后的参数
        """
        import random
        
        mutations = {}
        new_params = self.parameters.copy()
        
        for param, value in self.parameters.items():
            if random.random() < mutation_rate:
                # 执行变异
                bounds = self.param_bounds.get(param, (value * 0.5, value * 1.5))
                
                # 随机方向变异
                change = (random.random() - 0.5) * 2 * mutation_strength * (bounds[1] - bounds[0])
                new_value = value + change
                
                # 限制在边界内
                new_value = max(bounds[0], min(bounds[1], new_value))
                
                mutations[param] = {
                    "old": value,
                    "new": new_value,
                    "change_pct": (new_value - value) / value * 100 if value != 0 else 0,
                }
                new_params[param] = new_value
        
        if mutations:
            self.evolution_history.append({
                "timestamp": datetime.now().isoformat(),
                "mutations": mutations,
            })
        
        return new_params
    
    def evolve_based_on_performance(
        self,
        win_rate: float,
        profit_factor: float,
        avg_pnl: float
    ) -> Dict[str, Any]:
        """
        基于性能指标进化参数
        
        性能好 -> 微调，性能差 -> 大幅变异
        """
        # 计算健康分数
        health_score = (
            win_rate * 0.4 +
            min(1, profit_factor / 2) * 0.3 +
            (1 if avg_pnl > 0 else 0) * 0.3
        )
        
        # 根据健康分数决定变异强度
        if health_score > 0.7:
            # 性能好，轻微微调
            mutation_rate = 0.1
            mutation_strength = 0.05
        elif health_score > 0.4:
            # 性能一般，中等调整
            mutation_rate = 0.3
            mutation_strength = 0.15
        else:
            # 性能差，大幅变异
            mutation_rate = 0.5
            mutation_strength = 0.3
        
        new_params = self.mutate_parameters(mutation_rate, mutation_strength)
        
        return {
            "health_score": health_score,
            "mutation_rate": mutation_rate,
            "mutation_strength": mutation_strength,
            "new_parameters": new_params,
        }
    
    def get_optimal_parameters(self) -> Dict[str, float]:
        """获取当前最优参数"""
        return self.parameters.copy()


class EvolutionManager:
    """
    进化管理器
    整合信号追踪、权重优化、策略进化
    """
    
    def __init__(self, storage_path: str = ".signal_history.json"):
        self.tracker = SignalPerformanceTracker(storage_path)
        self.weight_optimizer = FeatureWeightOptimizer()
        self.evolver = StrategyEvolver()
        
        self.last_evolution_time: Optional[datetime] = None
        self.evolution_interval = timedelta(hours=24)  # 每24小时进化一次
    
    def record_and_validate(
        self,
        signal_record: SignalRecord
    ) -> Dict[str, Any]:
        """
        记录信号并验证历史信号
        """
        # 记录新信号
        self.tracker.record_signal(signal_record)
        
        # 检查需要更新的历史信号
        # (在实际系统中，这应该由外部触发)
        
        return {
            "recorded": True,
            "signal_id": signal_record.signal_id,
        }
    
    def run_evolution_cycle(self) -> Dict[str, Any]:
        """
        运行进化周期
        """
        now = datetime.now()
        
        # 检查是否需要进化
        if self.last_evolution_time:
            if now - self.last_evolution_time < self.evolution_interval:
                return {
                    "status": "skipped",
                    "reason": "Too soon since last evolution",
                    "next_evolution": (
                        self.last_evolution_time + self.evolution_interval
                    ).isoformat(),
                }
        
        # 获取最近性能
        performance = self.tracker.get_recent_performance(days=7)
        
        # 更新权重
        new_weights = self.weight_optimizer.update_weights(self.tracker.signals)
        
        # 进化参数
        evolution_result = self.evolver.evolve_based_on_performance(
            win_rate=performance.get("win_rate", 0.5),
            profit_factor=performance.get("profit_factor", 1.0),
            avg_pnl=performance.get("total_pnl", 0) / max(1, performance.get("total_trades", 1)),
        )
        
        self.last_evolution_time = now
        
        return {
            "status": "completed",
            "timestamp": now.isoformat(),
            "performance": performance,
            "new_weights": new_weights,
            "feature_importance": self.weight_optimizer.get_feature_importance(),
            "parameter_evolution": evolution_result,
        }
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """获取进化状态 - 使用三重标签法统计"""
        # 从 signal_history 获取正确的统计
        from explain.signal_history import get_tracker
        tracker = get_tracker()
        stats = tracker.get_stats()
        reliability = tracker.get_reliability()
        
        return {
            "total_signals": stats.total_signals,
            "completed_signals": stats.completed_signals,
            "pending_signals": stats.pending_signals,
            "recent_performance": {
                "win_rate": stats.win_rate,
                "profit_factor": stats.profit_factor,
                "expectancy": stats.expectancy,
                "avg_r_multiple": stats.avg_r_multiple,
                "avg_win_percent": stats.avg_win_percent,
                "avg_loss_percent": stats.avg_loss_percent,
                "total_trades": stats.completed_signals,
                "wins": stats.wins,
                "losses": stats.losses,
            },
            "reliability": reliability,
            "current_weights": self.weight_optimizer.weights,
            "feature_importance": self.weight_optimizer.get_feature_importance(),
            "current_parameters": self.evolver.get_optimal_parameters(),
            "last_evolution": self.last_evolution_time.isoformat() if self.last_evolution_time else None,
            "adjustments_count": len(self.weight_optimizer.adjustment_history),
        }


# 便捷函数
def run_evolution() -> Dict[str, Any]:
    """
    运行进化周期（同步接口）
    """
    manager = EvolutionManager()
    return manager.run_evolution_cycle()


def get_evolution_status() -> Dict[str, Any]:
    """
    获取进化状态
    """
    manager = EvolutionManager()
    return manager.get_evolution_status()
