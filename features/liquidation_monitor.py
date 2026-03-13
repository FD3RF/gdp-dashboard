"""
清算监控模块 (Liquidation Monitor)
=================================
核心功能：监控多空清算级别，预测清算事件
- 清算热度图
- 多空清算不平衡
- 清算级联预警
- 清算级别价格预测
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass
class LiquidationLevel:
    """清算级别"""
    price: float
    side: str  # "long" or "short"
    estimated_size: float  # 预估清算金额 (USD)
    leverage: float  # 估计杠杆
    
    # 距离
    distance_pct: float  # 距离当前价格百分比
    
    # 风险评分
    risk_score: float = 0.0  # 0-100
    cascade_probability: float = 0.0  # 级联概率


@dataclass
class LiquidationHeatmap:
    """清算热度图"""
    timestamp: datetime
    
    # 价格区间清算分布
    price_levels: List[Tuple[float, float, float]]  # (price, long_liq, short_liq)
    
    # 热点区域
    hot_zones: List[Dict[str, Any]]  # 高清算密度区域
    
    # 不平衡
    total_long_liquidations: float
    total_short_liquidations: float
    imbalance_ratio: float  # long/short
    imbalance_direction: str  # "long_heavy" / "short_heavy" / "balanced"


@dataclass
class LiquidationAlert:
    """清算预警"""
    timestamp: datetime
    alert_type: str  # "approaching" / "cascade_risk" / "imbalance_extreme"
    severity: str  # "low" / "medium" / "high" / "critical"
    
    details: str
    affected_side: str  # "long" / "short" / "both"
    
    # 价格预测
    trigger_price: float
    expected_impact: float  # 预期价格变动百分比
    
    # 建议
    recommended_action: str


class LiquidationMonitor:
    """
    清算监控器
    
    核心功能：
    1. 计算多空清算级别
    2. 生成清算热度图
    3. 检测清算级联风险
    4. 预测清算引发的价格变动
    """
    
    # 默认杠杆分布（用于估算）
    DEFAULT_LEVERAGE_DISTRIBUTION = {
        10: 0.05,   # 5% 使用10x杠杆
        20: 0.15,   # 15% 使用20x杠杆
        25: 0.25,   # 25% 使用25x杠杆
        50: 0.30,   # 30% 使用50x杠杆
        75: 0.15,   # 15% 使用75x杠杆
        100: 0.10,  # 10% 使用100x杠杆
    }
    
    # 清算阈值（强平保证金率）
    LIQUIDATION_THRESHOLD = 0.8  # 80% 保证金率触发强平
    
    # 风险等级阈值
    RISK_THRESHOLDS = {
        "low": 30,
        "medium": 50,
        "high": 70,
        "critical": 85,
    }
    
    def __init__(
        self,
        leverage_distribution: Dict[int, float] = None,
        open_interest: float = 1000000,  # 默认持仓量 (USD)
    ):
        self.leverage_distribution = leverage_distribution or self.DEFAULT_LEVERAGE_DISTRIBUTION
        self.open_interest = open_interest
        
        # 清算级别缓存
        self.liquidation_levels: List[LiquidationLevel] = []
        
        # 历史记录
        self.history: List[Dict] = []
        self.alerts: List[LiquidationAlert] = []
        
        # 统计
        self.stats = {
            "total_alerts": 0,
            "cascade_warnings": 0,
            "imbalance_warnings": 0,
        }
    
    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: float,
        side: str,
        maintenance_margin: float = 0.005
    ) -> float:
        """
        计算清算价格
        
        Args:
            entry_price: 入场价格
            leverage: 杠杆倍数
            side: 方向 ("long" / "short")
            maintenance_margin: 维持保证金率
        
        Returns:
            清算价格
        """
        if side == "long":
            # 做多清算价 = 入场价 * (1 - 1/杠杆 + 维持保证金)
            liquidation_price = entry_price * (1 - 1/leverage + maintenance_margin)
        else:
            # 做空清算价 = 入场价 * (1 + 1/杠杆 - 维持保证金)
            liquidation_price = entry_price * (1 + 1/leverage - maintenance_margin)
        
        return liquidation_price
    
    def estimate_liquidation_levels(
        self,
        current_price: float,
        open_interest_long: float = None,
        open_interest_short: float = None,
        avg_entry_long: float = None,
        avg_entry_short: float = None,
    ) -> List[LiquidationLevel]:
        """
        估算清算级别
        
        Args:
            current_price: 当前价格
            open_interest_long: 多头持仓量
            open_interest_short: 空头持仓量
            avg_entry_long: 多头平均入场价
            avg_entry_short: 空头平均入场价
        
        Returns:
            清算级别列表
        """
        oi_long = open_interest_long or self.open_interest * 0.5
        oi_short = open_interest_short or self.open_interest * 0.5
        entry_long = avg_entry_long or current_price
        entry_short = avg_entry_short or current_price
        
        levels = []
        
        # 遍历杠杆分布
        for leverage, weight in self.leverage_distribution.items():
            # 多头清算级别
            long_liq_price = self.calculate_liquidation_price(entry_long, leverage, "long")
            long_size = oi_long * weight
            distance_pct = (long_liq_price - current_price) / current_price * 100
            
            if distance_pct < 0:  # 只有低于当前价格的清算级别才可能触发
                levels.append(LiquidationLevel(
                    price=long_liq_price,
                    side="long",
                    estimated_size=long_size,
                    leverage=leverage,
                    distance_pct=distance_pct,
                    risk_score=self._calculate_risk_score(abs(distance_pct), long_size),
                    cascade_probability=self._estimate_cascade_probability(leverage, long_size),
                ))
            
            # 空头清算级别
            short_liq_price = self.calculate_liquidation_price(entry_short, leverage, "short")
            short_size = oi_short * weight
            distance_pct = (short_liq_price - current_price) / current_price * 100
            
            if distance_pct > 0:  # 只有高于当前价格的清算级别才可能触发
                levels.append(LiquidationLevel(
                    price=short_liq_price,
                    side="short",
                    estimated_size=short_size,
                    leverage=leverage,
                    distance_pct=distance_pct,
                    risk_score=self._calculate_risk_score(abs(distance_pct), short_size),
                    cascade_probability=self._estimate_cascade_probability(leverage, short_size),
                ))
        
        # 按距离排序
        levels.sort(key=lambda x: abs(x.distance_pct))
        
        self.liquidation_levels = levels
        
        return levels
    
    def _calculate_risk_score(self, distance_pct: float, size: float) -> float:
        """计算风险评分"""
        # 距离越近风险越高
        distance_score = max(0, 100 - distance_pct * 10)
        
        # 规模越大风险越高
        size_score = min(100, size / 10000)  # 每10000 USD 加1分
        
        return min(100, distance_score * 0.7 + size_score * 0.3)
    
    def _estimate_cascade_probability(self, leverage: float, size: float) -> float:
        """估算级联概率"""
        # 高杠杆更容易级联
        leverage_factor = leverage / 100
        
        # 大规模更容易引发连锁
        size_factor = min(1, size / 100000)
        
        return min(1, leverage_factor * 0.5 + size_factor * 0.5)
    
    def generate_heatmap(
        self,
        current_price: float,
        price_range_pct: float = 10,
        num_levels: int = 20
    ) -> LiquidationHeatmap:
        """
        生成清算热度图
        
        Args:
            current_price: 当前价格
            price_range_pct: 价格范围百分比
            num_levels: 价格级别数量
        
        Returns:
            LiquidationHeatmap
        """
        # 生成价格区间
        price_min = current_price * (1 - price_range_pct / 100)
        price_max = current_price * (1 + price_range_pct / 100)
        price_step = (price_max - price_min) / num_levels
        
        price_levels = []
        total_long = 0
        total_short = 0
        
        for i in range(num_levels):
            price = price_min + price_step * (i + 0.5)
            
            # 累加该价格区间的清算量
            long_liq = 0
            short_liq = 0
            
            for level in self.liquidation_levels:
                if abs(level.price - price) < price_step:
                    if level.side == "long":
                        long_liq += level.estimated_size
                        total_long += level.estimated_size
                    else:
                        short_liq += level.estimated_size
                        total_short += level.estimated_size
            
            price_levels.append((price, long_liq, short_liq))
        
        # 识别热点区域
        hot_zones = []
        for i, (price, long_liq, short_liq) in enumerate(price_levels):
            total_liq = long_liq + short_liq
            if total_liq > 100000:  # 超过100k USD
                hot_zones.append({
                    "price": price,
                    "total_liquidations": total_liq,
                    "long_liq": long_liq,
                    "short_liq": short_liq,
                    "dominant_side": "long" if long_liq > short_liq else "short",
                })
        
        # 计算不平衡
        if total_short > 0:
            imbalance_ratio = total_long / total_short
        else:
            imbalance_ratio = 10 if total_long > 0 else 1
        
        if imbalance_ratio > 1.5:
            imbalance_direction = "long_heavy"
        elif imbalance_ratio < 0.67:
            imbalance_direction = "short_heavy"
        else:
            imbalance_direction = "balanced"
        
        return LiquidationHeatmap(
            timestamp=datetime.now(),
            price_levels=price_levels,
            hot_zones=hot_zones,
            total_long_liquidations=total_long,
            total_short_liquidations=total_short,
            imbalance_ratio=imbalance_ratio,
            imbalance_direction=imbalance_direction,
        )
    
    def check_approaching_liquidations(
        self,
        current_price: float,
        threshold_pct: float = 2.0
    ) -> List[LiquidationAlert]:
        """
        检查接近的清算级别
        
        Args:
            current_price: 当前价格
            threshold_pct: 距离阈值百分比
        
        Returns:
            清算预警列表
        """
        alerts = []
        
        for level in self.liquidation_levels:
            distance_pct = abs(level.distance_pct)
            
            if distance_pct <= threshold_pct:
                # 确定严重程度
                if distance_pct <= 0.5:
                    severity = "critical"
                elif distance_pct <= 1.0:
                    severity = "high"
                elif distance_pct <= 1.5:
                    severity = "medium"
                else:
                    severity = "low"
                
                alert = LiquidationAlert(
                    timestamp=datetime.now(),
                    alert_type="approaching",
                    severity=severity,
                    details=f"{level.side.upper()} 清算级别距离 {distance_pct:.2f}%，预估规模 ${level.estimated_size:,.0f}",
                    affected_side=level.side,
                    trigger_price=level.price,
                    expected_impact=level.cascade_probability * 2,  # 预估价格影响
                    recommended_action=self._get_action_recommendation(level.side, severity),
                )
                alerts.append(alert)
        
        self.alerts.extend(alerts)
        self.stats["total_alerts"] += len(alerts)
        
        return alerts
    
    def detect_cascade_risk(self, current_price: float) -> Optional[LiquidationAlert]:
        """
        检测级联风险
        
        级联条件：
        1. 多个高杠杆清算级别接近
        2. 清算规模较大
        """
        critical_levels = [
            level for level in self.liquidation_levels
            if abs(level.distance_pct) < 3 and level.cascade_probability > 0.5
        ]
        
        if len(critical_levels) >= 2:
            # 按方向分组
            long_levels = [l for l in critical_levels if l.side == "long"]
            short_levels = [l for l in critical_levels if l.side == "short"]
            
            if long_levels:
                total_long_size = sum(l.estimated_size for l in long_levels)
                if total_long_size > 500000:  # 超过50万
                    alert = LiquidationAlert(
                        timestamp=datetime.now(),
                        alert_type="cascade_risk",
                        severity="critical",
                        details=f"多头清算级联风险：{len(long_levels)} 个级别，总计 ${total_long_size:,.0f}",
                        affected_side="long",
                        trigger_price=min(l.price for l in long_levels),
                        expected_impact=-3.0,  # 预估下跌3%
                        recommended_action="⚠️ 建议减多或观望，清算级联可能发生",
                    )
                    self.alerts.append(alert)
                    self.stats["cascade_warnings"] += 1
                    return alert
            
            if short_levels:
                total_short_size = sum(l.estimated_size for l in short_levels)
                if total_short_size > 500000:
                    alert = LiquidationAlert(
                        timestamp=datetime.now(),
                        alert_type="cascade_risk",
                        severity="critical",
                        details=f"空头清算级联风险：{len(short_levels)} 个级别，总计 ${total_short_size:,.0f}",
                        affected_side="short",
                        trigger_price=max(l.price for l in short_levels),
                        expected_impact=3.0,  # 预估上涨3%
                        recommended_action="⚠️ 建议减空或观望，清算级联可能发生",
                    )
                    self.alerts.append(alert)
                    self.stats["cascade_warnings"] += 1
                    return alert
        
        return None
    
    def _get_action_recommendation(self, side: str, severity: str) -> str:
        """获取行动建议"""
        if severity == "critical":
            if side == "long":
                return "🚨 紧急：多头清算迫在眉睫，建议立即减仓或止损"
            else:
                return "🚨 紧急：空头清算迫在眉睫，建议立即减仓或止损"
        elif severity == "high":
            if side == "long":
                return "⚠️ 警告：多头清算接近，建议降低多仓"
            else:
                return "⚠️ 警告：空头清算接近，建议降低空仓"
        else:
            return "ℹ️ 关注清算级别，做好风险管理"
    
    def get_liquidation_summary(
        self,
        current_price: float,
        open_interest_long: float = None,
        open_interest_short: float = None,
    ) -> Dict[str, Any]:
        """
        获取清算监控摘要
        """
        # 估算清算级别
        levels = self.estimate_liquidation_levels(
            current_price,
            open_interest_long,
            open_interest_short,
        )
        
        # 生成热度图
        heatmap = self.generate_heatmap(current_price)
        
        # 检查接近的清算
        approaching = self.check_approaching_liquidations(current_price)
        
        # 检测级联风险
        cascade_alert = self.detect_cascade_risk(current_price)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "total_levels": len(levels),
            "nearest_long": min(
                [l for l in levels if l.side == "long"],
                key=lambda x: abs(x.distance_pct),
                default=None
            ).__dict__ if any(l.side == "long" for l in levels) else None,
            "nearest_short": min(
                [l for l in levels if l.side == "short"],
                key=lambda x: abs(x.distance_pct),
                default=None
            ).__dict__ if any(l.side == "short" for l in levels) else None,
            "heatmap": {
                "total_long_liq": heatmap.total_long_liquidations,
                "total_short_liq": heatmap.total_short_liquidations,
                "imbalance_ratio": round(heatmap.imbalance_ratio, 2),
                "imbalance_direction": heatmap.imbalance_direction,
                "hot_zones": heatmap.hot_zones[:5],
            },
            "approaching_alerts": [
                {
                    "severity": a.severity,
                    "side": a.affected_side,
                    "details": a.details,
                    "trigger_price": a.trigger_price,
                }
                for a in approaching[:5]
            ],
            "cascade_risk": cascade_alert.details if cascade_alert else None,
            "stats": self.stats,
        }


# 便捷函数
def monitor_liquidations(
    current_price: float,
    open_interest: float = 1000000,
) -> Dict[str, Any]:
    """
    监控清算（便捷函数）
    
    Args:
        current_price: 当前价格
        open_interest: 持仓量
    
    Returns:
        清算监控摘要
    """
    monitor = LiquidationMonitor(open_interest=open_interest)
    return monitor.get_liquidation_summary(current_price)
