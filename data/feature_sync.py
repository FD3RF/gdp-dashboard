"""
特征时间同步层 (Feature Synchronization Layer)
==============================================
解决多数据源时间错位问题

核心原理：
- 所有特征必须统一到同一时间点 t
- 过期数据标记时间差
- 特征矩阵每行代表一个完整时间切片
"""

import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimestampedFeature:
    """带时间戳的特征"""
    name: str
    value: Any
    timestamp: datetime
    source: str  # 数据源
    freshness_seconds: float = 0.0  # 数据新鲜度（秒）
    is_stale: bool = False  # 是否过期
    
    def age_seconds(self, current_time: datetime = None) -> float:
        """计算数据年龄（秒）"""
        if current_time is None:
            current_time = datetime.now()
        return (current_time - self.timestamp).total_seconds()
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "freshness_seconds": self.freshness_seconds,
            "is_stale": self.is_stale,
        }


@dataclass
class FeatureMatrix:
    """
    特征矩阵
    
    每行代表一个完整的时间切片
    所有特征对齐到同一时间点
    """
    timestamp: datetime
    features: Dict[str, TimestampedFeature] = field(default_factory=dict)
    
    # 元数据
    candle_interval: int = 300  # K线周期（秒），默认5分钟
    feature_completeness: float = 1.0  # 特征完整度
    
    def get_feature(self, name: str) -> Optional[Any]:
        """获取特征值"""
        if name in self.features:
            return self.features[name].value
        return None
    
    def set_feature(self, feature: TimestampedFeature):
        """设置特征"""
        self.features[feature.name] = feature
        self._update_completeness()
    
    def _update_completeness(self):
        """更新特征完整度"""
        total_expected = 15  # 预期特征数量
        self.feature_completeness = len(self.features) / total_expected
    
    def get_age_report(self, current_time: datetime = None) -> Dict[str, float]:
        """获取所有特征的年龄报告"""
        if current_time is None:
            current_time = datetime.now()
        
        return {
            name: f.age_seconds(current_time)
            for name, f in self.features.items()
        }
    
    def get_stale_features(self, threshold_seconds: float = 600) -> List[str]:
        """获取过期特征列表"""
        current_time = datetime.now()
        return [
            name for name, f in self.features.items()
            if f.age_seconds(current_time) > threshold_seconds
        ]
    
    def to_dict(self) -> Dict:
        """转换为字典（用于AI输入）"""
        result = {
            "_metadata": {
                "timestamp": self.timestamp.isoformat(),
                "completeness": self.feature_completeness,
                "candle_interval": self.candle_interval,
            }
        }
        
        for name, feature in self.features.items():
            result[name] = feature.value
            result[f"{name}_age_seconds"] = feature.age_seconds()
            result[f"{name}_is_stale"] = feature.is_stale
        
        return result
    
    def to_ai_input(self) -> Dict:
        """
        转换为AI模型输入
        
        只包含数值特征，排除元数据
        """
        result = {}
        
        for name, feature in self.features.items():
            value = feature.value
            
            # 处理不同类型的值
            if isinstance(value, (int, float)):
                result[name] = value
            elif isinstance(value, bool):
                result[name] = 1.0 if value else 0.0
            elif isinstance(value, str):
                # 分类变量转为数值
                result[f"{name}_encoded"] = hash(value) % 1000 / 1000
            elif isinstance(value, dict):
                # 展平字典
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        result[f"{name}_{k}"] = v
        
        # 添加时间元信息
        result["_oldest_feature_age"] = max(
            (f.age_seconds() for f in self.features.values()),
            default=0
        )
        result["_feature_completeness"] = self.feature_completeness
        
        return result


class FeatureSyncLayer:
    """
    特征时间同步层
    
    核心功能：
    1. 统一所有数据源的时间
    2. 管理数据新鲜度
    3. 生成时间对齐的特征矩阵
    """
    
    # 数据源过期阈值（秒）
    STALE_THRESHOLDS = {
        "kline": 360,  # K线 6分钟过期
        "orderbook": 10,  # 订单簿 10秒过期
        "trades": 5,  # 成交数据 5秒过期
        "funding_rate": 28800,  # 资金费率 8小时过期
        "whale": 600,  # 链上数据 10分钟过期
        "sentiment": 300,  # 情绪数据 5分钟过期
        "liquidation": 30,  # 清算数据 30秒过期
        "order_flow": 60,  # 订单流 1分钟过期
        "regime": 300,  # 市场状态 5分钟过期
        "indicator": 360,  # 技术指标 6分钟过期
    }
    
    # 关键程度权重
    FEATURE_WEIGHTS = {
        "price": 1.0,
        "hurst": 0.9,
        "regime": 0.85,
        "orderbook_imbalance": 0.8,
        "cvd": 0.8,
        "funding_rate": 0.7,
        "whale_flow": 0.75,
        "sentiment": 0.6,
        "liquidation": 0.65,
        "rsi": 0.7,
        "momentum": 0.7,
    }
    
    def __init__(self, candle_interval: int = 300):
        """
        Args:
            candle_interval: K线周期（秒）
        """
        self.candle_interval = candle_interval
        self.features: Dict[str, TimestampedFeature] = {}
        self.lock = threading.Lock()
        
        # 历史特征矩阵
        self.history: deque = deque(maxlen=100)
        
        # 数据源最后更新时间
        self.last_update: Dict[str, datetime] = {}
        
        # 回调函数
        self._callbacks: List[Callable] = []
    
    def register_callback(self, callback: Callable):
        """注册回调函数（特征更新时触发）"""
        self._callbacks.append(callback)
    
    def update_feature(
        self,
        name: str,
        value: Any,
        source: str,
        timestamp: datetime = None,
    ) -> TimestampedFeature:
        """
        更新特征
        
        Args:
            name: 特征名称
            value: 特征值
            source: 数据源
            timestamp: 数据时间戳（None表示当前时间）
        
        Returns:
            TimestampedFeature
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 计算新鲜度
        threshold = self.STALE_THRESHOLDS.get(source, 600)
        age = (datetime.now() - timestamp).total_seconds()
        is_stale = age > threshold
        
        feature = TimestampedFeature(
            name=name,
            value=value,
            timestamp=timestamp,
            source=source,
            freshness_seconds=threshold,
            is_stale=is_stale,
        )
        
        with self.lock:
            self.features[name] = feature
            self.last_update[source] = timestamp
        
        # 触发回调
        for callback in self._callbacks:
            try:
                callback(feature)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return feature
    
    def sync_to_timestamp(
        self,
        target_timestamp: datetime = None,
        max_age_seconds: float = 600,
    ) -> FeatureMatrix:
        """
        同步所有特征到指定时间点
        
        Args:
            target_timestamp: 目标时间（None表示当前K线时间）
            max_age_seconds: 最大允许数据年龄
        
        Returns:
            FeatureMatrix
        """
        if target_timestamp is None:
            # 对齐到最近的K线时间
            target_timestamp = self._align_to_candle(datetime.now())
        
        matrix = FeatureMatrix(
            timestamp=target_timestamp,
            candle_interval=self.candle_interval,
        )
        
        with self.lock:
            for name, feature in self.features.items():
                # 检查数据年龄
                age = feature.age_seconds(target_timestamp)
                
                if age <= max_age_seconds:
                    # 标记是否过期
                    feature.is_stale = age > feature.freshness_seconds
                    matrix.set_feature(feature)
                else:
                    logger.warning(
                        f"Feature {name} too old: {age:.0f}s > {max_age_seconds}s"
                    )
        
        # 保存历史
        self.history.append(matrix)
        
        return matrix
    
    def _align_to_candle(self, timestamp: datetime) -> datetime:
        """对齐到K线时间"""
        epoch = timestamp.timestamp()
        aligned_epoch = (epoch // self.candle_interval) * self.candle_interval
        return datetime.fromtimestamp(aligned_epoch)
    
    def get_sync_status(self) -> Dict[str, Any]:
        """获取同步状态"""
        current_time = datetime.now()
        
        status = {
            "current_time": current_time.isoformat(),
            "features_count": len(self.features),
            "last_update": {
                source: ts.isoformat() 
                for source, ts in self.last_update.items()
            },
            "feature_ages": {},
            "stale_features": [],
            "source_status": {},
        }
        
        for name, feature in self.features.items():
            age = feature.age_seconds(current_time)
            status["feature_ages"][name] = {
                "age_seconds": age,
                "is_stale": feature.is_stale,
                "source": feature.source,
            }
            
            if feature.is_stale:
                status["stale_features"].append(name)
        
        # 数据源状态
        for source, threshold in self.STALE_THRESHOLDS.items():
            last_ts = self.last_update.get(source)
            if last_ts:
                age = (current_time - last_ts).total_seconds()
                status["source_status"][source] = {
                    "age_seconds": age,
                    "threshold": threshold,
                    "is_stale": age > threshold,
                    "last_update": last_ts.isoformat(),
                }
            else:
                status["source_status"][source] = {
                    "age_seconds": None,
                    "threshold": threshold,
                    "is_stale": True,
                    "last_update": None,
                }
        
        return status
    
    def get_data_quality_score(self) -> float:
        """
        计算数据质量分数
        
        考虑：
        1. 特征完整度
        2. 数据新鲜度
        3. 特征权重
        
        Returns:
            0-1之间的分数
        """
        if not self.features:
            return 0.0
        
        current_time = datetime.now()
        weighted_score = 0.0
        total_weight = 0.0
        
        for name, feature in self.features.items():
            weight = self.FEATURE_WEIGHTS.get(name, 0.5)
            age = feature.age_seconds(current_time)
            threshold = feature.freshness_seconds
            
            # 新鲜度分数：越新鲜分数越高
            if age <= threshold:
                freshness_score = 1.0 - (age / threshold) * 0.3  # 最多扣30%
            else:
                freshness_score = max(0.1, 1.0 - (age - threshold) / 600)  # 过期后线性衰减
            
            weighted_score += freshness_score * weight
            total_weight += weight
        
        # 特征完整度
        expected_features = len(self.FEATURE_WEIGHTS)
        completeness = len(self.features) / expected_features if expected_features > 0 else 1.0
        
        # 综合分数
        final_score = (weighted_score / total_weight * 0.7 + completeness * 0.3) if total_weight > 0 else 0.0
        
        return min(1.0, max(0.0, final_score))
    
    def create_snapshot(self) -> Dict[str, Any]:
        """创建快照（用于调试和日志）"""
        return {
            "timestamp": datetime.now().isoformat(),
            "features": {
                name: feature.to_dict()
                for name, feature in self.features.items()
            },
            "quality_score": self.get_data_quality_score(),
            "sync_status": self.get_sync_status(),
        }


# 全局实例
_feature_sync: Optional[FeatureSyncLayer] = None


def get_feature_sync() -> FeatureSyncLayer:
    """获取全局特征同步层"""
    global _feature_sync
    if _feature_sync is None:
        _feature_sync = FeatureSyncLayer()
    return _feature_sync


def sync_features(
    price: float = None,
    hurst: float = None,
    regime: str = None,
    orderbook_imbalance: float = None,
    cvd: float = None,
    funding_rate: float = None,
    whale_flow: float = None,
    sentiment: float = None,
    liquidation: Dict = None,
    rsi: float = None,
    momentum: float = None,
    **kwargs
) -> FeatureMatrix:
    """
    同步所有特征（便捷函数）
    
    Returns:
        FeatureMatrix
    """
    sync = get_feature_sync()
    
    # 更新各个特征
    if price is not None:
        sync.update_feature("price", price, "kline")
    if hurst is not None:
        sync.update_feature("hurst", hurst, "indicator")
    if regime is not None:
        sync.update_feature("regime", regime, "regime")
    if orderbook_imbalance is not None:
        sync.update_feature("orderbook_imbalance", orderbook_imbalance, "orderbook")
    if cvd is not None:
        sync.update_feature("cvd", cvd, "order_flow")
    if funding_rate is not None:
        sync.update_feature("funding_rate", funding_rate, "funding_rate")
    if whale_flow is not None:
        sync.update_feature("whale_flow", whale_flow, "whale")
    if sentiment is not None:
        sync.update_feature("sentiment", sentiment, "sentiment")
    if liquidation is not None:
        sync.update_feature("liquidation", liquidation, "liquidation")
    if rsi is not None:
        sync.update_feature("rsi", rsi, "indicator")
    if momentum is not None:
        sync.update_feature("momentum", momentum, "indicator")
    
    # 更新额外特征
    for name, value in kwargs.items():
        if value is not None:
            sync.update_feature(name, value, "other")
    
    # 生成同步的特征矩阵
    return sync.sync_to_timestamp()
