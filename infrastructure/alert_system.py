"""
实时预警系统 (Real-time Alert System)
====================================
价格预警、清算预警、背离预警

功能：
1. 价格接近支撑/阻力预警
2. 清算墙触发预警
3. CVD背离预警
4. 黑天鹅检测
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """预警类型"""
    PRICE_LEVEL = "price_level"       # 价格接近关键位
    LIQUIDATION = "liquidation"       # 清算墙
    DIVERGENCE = "divergence"         # 背离
    BLACK_SWAN = "black_swan"         # 黑天鹅
    SIGNAL = "signal"                 # 交易信号
    RISK = "risk"                     # 风险预警


class AlertSeverity(Enum):
    """预警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """预警"""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


class AlertEngine:
    """
    预警引擎
    
    检测：
    1. 价格接近支撑/阻力
    2. 清算墙即将触发
    3. CVD背离
    4. 极端波动（黑天鹅）
    """
    
    # 预警阈值
    PRICE_PROXIMITY_THRESHOLD = 0.005  # 0.5%距离触发
    LIQUIDATION_PROXIMITY_THRESHOLD = 0.003  # 0.3%距离触发
    BLACK_SWAN_VOLATILITY = 0.05  # 5%波动
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.max_alerts = 50
        self._callbacks: List[Callable] = []
    
    def register_callback(self, callback: Callable):
        """注册回调"""
        self._callbacks.append(callback)
    
    def _trigger_callbacks(self, alert: Alert):
        """触发回调"""
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _add_alert(self, alert: Alert):
        """添加预警"""
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # 触发回调
        self._trigger_callbacks(alert)
        
        # 日志
        if alert.severity == AlertSeverity.EMERGENCY:
            logger.critical(f"🚨 {alert.title}: {alert.message}")
        elif alert.severity == AlertSeverity.CRITICAL:
            logger.warning(f"⚠️ {alert.title}: {alert.message}")
        else:
            logger.info(f"ℹ️ {alert.title}: {alert.message}")
    
    def check_price_levels(
        self,
        current_price: float,
        support_zones: List[tuple],
        resistance_zones: List[tuple],
    ) -> List[Alert]:
        """
        检查价格接近支撑/阻力
        """
        alerts = []
        
        # 检查支撑
        for price, strength in support_zones[:3]:
            distance_pct = abs(current_price - price) / current_price
            
            if distance_pct < self.PRICE_PROXIMITY_THRESHOLD:
                severity = AlertSeverity.CRITICAL if strength > 3 else AlertSeverity.WARNING
                
                alert = Alert(
                    alert_type=AlertType.PRICE_LEVEL,
                    severity=severity,
                    title="支撑位预警",
                    message=f"价格接近支撑 ${price:,.0f} (强度: {strength:.1f}x)",
                    details={
                        "level_type": "support",
                        "level_price": price,
                        "strength": strength,
                        "distance_pct": distance_pct * 100,
                    }
                )
                alerts.append(alert)
                self._add_alert(alert)
        
        # 检查阻力
        for price, strength in resistance_zones[:3]:
            distance_pct = abs(current_price - price) / current_price
            
            if distance_pct < self.PRICE_PROXIMITY_THRESHOLD:
                severity = AlertSeverity.CRITICAL if strength > 3 else AlertSeverity.WARNING
                
                alert = Alert(
                    alert_type=AlertType.PRICE_LEVEL,
                    severity=severity,
                    title="阻力位预警",
                    message=f"价格接近阻力 ${price:,.0f} (强度: {strength:.1f}x)",
                    details={
                        "level_type": "resistance",
                        "level_price": price,
                        "strength": strength,
                        "distance_pct": distance_pct * 100,
                    }
                )
                alerts.append(alert)
                self._add_alert(alert)
        
        return alerts
    
    def check_liquidation(
        self,
        current_price: float,
        liquidation_zones: List[Dict],
    ) -> List[Alert]:
        """
        检查清算墙预警
        """
        alerts = []
        
        for liq in liquidation_zones[:5]:
            liq_price = liq.get('price', 0)
            liq_dir = liq.get('direction', 'unknown')
            liq_amount = liq.get('amount', 0)
            
            distance_pct = abs(current_price - liq_price) / current_price
            
            if distance_pct < self.LIQUIDATION_PROXIMITY_THRESHOLD:
                severity = AlertSeverity.CRITICAL if liq_amount > 1e6 else AlertSeverity.WARNING
                
                alert = Alert(
                    alert_type=AlertType.LIQUIDATION,
                    severity=severity,
                    title="清算墙预警",
                    message=f"价格接近{liq_dir}头清算墙 ${liq_price:,.0f} (${liq_amount/1e6:.1f}M)",
                    details={
                        "liquidation_price": liq_price,
                        "direction": liq_dir,
                        "amount": liq_amount,
                        "distance_pct": distance_pct * 100,
                    }
                )
                alerts.append(alert)
                self._add_alert(alert)
        
        return alerts
    
    def check_divergence(
        self,
        price_trend: str,  # "up" / "down"
        cvd_trend: str,    # "bullish" / "bearish"
        cvd_value: float,
    ) -> Optional[Alert]:
        """
        检查CVD背离
        """
        # 价格上涨但CVD看跌 = 顶背离
        if price_trend == "up" and cvd_trend == "bearish":
            alert = Alert(
                alert_type=AlertType.DIVERGENCE,
                severity=AlertSeverity.WARNING,
                title="顶背离预警",
                message=f"价格上涨但CVD走弱 ({cvd_value:.1f})，可能见顶",
                details={
                    "price_trend": price_trend,
                    "cvd_trend": cvd_trend,
                    "cvd_value": cvd_value,
                    "divergence_type": "bearish",
                }
            )
            self._add_alert(alert)
            return alert
        
        # 价格下跌但CVD看涨 = 底背离
        if price_trend == "down" and cvd_trend == "bullish":
            alert = Alert(
                alert_type=AlertType.DIVERGENCE,
                severity=AlertSeverity.WARNING,
                title="底背离预警",
                message=f"价格下跌但CVD走强 ({cvd_value:.1f})，可能见底",
                details={
                    "price_trend": price_trend,
                    "cvd_trend": cvd_trend,
                    "cvd_value": cvd_value,
                    "divergence_type": "bullish",
                }
            )
            self._add_alert(alert)
            return alert
        
        return None
    
    def check_black_swan(
        self,
        price_change_pct: float,  # 5分钟价格变化百分比
        volatility: float,         # 波动率
    ) -> Optional[Alert]:
        """
        检查黑天鹅事件
        """
        if abs(price_change_pct) > self.BLACK_SWAN_VOLATILITY * 100:
            direction = "暴涨" if price_change_pct > 0 else "暴跌"
            
            alert = Alert(
                alert_type=AlertType.BLACK_SWAN,
                severity=AlertSeverity.EMERGENCY,
                title=f"黑天鹅预警: {direction}",
                message=f"5分钟{direction} {abs(price_change_pct):.2f}%，暂停交易建议",
                details={
                    "price_change_pct": price_change_pct,
                    "volatility": volatility,
                    "action_recommended": "pause_trading",
                }
            )
            self._add_alert(alert)
            return alert
        
        return None
    
    def check_all(
        self,
        current_price: float,
        support_zones: List[tuple],
        resistance_zones: List[tuple],
        liquidation_zones: List[Dict],
        price_trend: str,
        cvd_trend: str,
        cvd_value: float,
        price_change_pct: float,
        volatility: float,
    ) -> List[Alert]:
        """
        执行所有检查
        """
        all_alerts = []
        
        # 价格接近关键位
        all_alerts.extend(self.check_price_levels(
            current_price, support_zones, resistance_zones
        ))
        
        # 清算墙
        all_alerts.extend(self.check_liquidation(
            current_price, liquidation_zones
        ))
        
        # 背离
        div_alert = self.check_divergence(price_trend, cvd_trend, cvd_value)
        if div_alert:
            all_alerts.append(div_alert)
        
        # 黑天鹅
        swan_alert = self.check_black_swan(price_change_pct, volatility)
        if swan_alert:
            all_alerts.append(swan_alert)
        
        return all_alerts
    
    def get_active_alerts(self, minutes: int = 30) -> List[Alert]:
        """获取活跃预警"""
        cutoff = datetime.now().timestamp() - minutes * 60
        return [
            a for a in self.alerts
            if a.timestamp.timestamp() > cutoff and not a.acknowledged
        ]
    
    def get_summary(self) -> Dict:
        """获取预警摘要"""
        active = self.get_active_alerts()
        
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active),
            "by_severity": {
                "emergency": len([a for a in active if a.severity == AlertSeverity.EMERGENCY]),
                "critical": len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
                "warning": len([a for a in active if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in active if a.severity == AlertSeverity.INFO]),
            },
            "by_type": {
                a_type.value: len([a for a in active if a.alert_type == a_type])
                for a_type in AlertType
            },
            "latest_alerts": [a.to_dict() for a in active[:5]],
        }


# 全局实例
_alert_engine: Optional[AlertEngine] = None


def get_alert_engine() -> AlertEngine:
    """获取预警引擎"""
    global _alert_engine
    if _alert_engine is None:
        _alert_engine = AlertEngine()
    return _alert_engine


def check_alerts(**kwargs) -> List[Alert]:
    """检查预警（便捷函数）"""
    engine = get_alert_engine()
    return engine.check_all(**kwargs)
