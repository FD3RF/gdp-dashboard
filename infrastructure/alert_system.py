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
    BREAKOUT = "breakout"             # 突破预警


class AlertSeverity(Enum):
    """预警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    
    @property
    def color(self) -> str:
        """获取预警级别对应的颜色"""
        colors = {
            "info": "#4caf50",      # 绿色
            "warning": "#ff9800",   # 橙色
            "critical": "#f44336",  # 红色
            "emergency": "#9c27b0", # 紫色（紧急）
        }
        return colors.get(self.value, "#9e9e9e")
    
    @property
    def icon(self) -> str:
        """获取预警级别对应的图标"""
        icons = {
            "info": "🟢",
            "warning": "🟡",
            "critical": "🔴",
            "emergency": "🚨",
        }
        return icons.get(self.value, "📍")


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
    
    @property
    def color(self) -> str:
        return self.severity.color
    
    @property
    def icon(self) -> str:
        return self.severity.icon
    
    def to_dict(self) -> Dict:
        return {
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "severity_color": self.color,
            "severity_icon": self.icon,
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
    
    预警级别规则：
    ┌────────────┬────────────────────────────────────────────────────────────┐
    │ INFO       │ 信息提示，无需紧急行动                                      │
    │            │ - 价格距离关键位 1.5%~2%                                   │
    │            │ - 清算墙距离 0.8%~1%                                       │
    ├────────────┼────────────────────────────────────────────────────────────┤
    │ WARNING    │ 需要关注，可能需要行动                                      │
    │            │ - 价格距离关键位 0.5%~1.5%                                 │
    │            │ - 清算墙距离 0.3%~0.8%                                     │
    │            │ - CVD背离检测                                             │
    ├────────────┼────────────────────────────────────────────────────────────┤
    │ CRITICAL   │ 需要立即关注，建议采取行动                                  │
    │            │ - 价格距离关键位 < 0.5%                                    │
    │            │ - 清算墙距离 < 0.3% 且金额 > 500K                           │
    │            │ - 强力关键位被触及                                         │
    ├────────────┼────────────────────────────────────────────────────────────┤
    │ EMERGENCY  │ 紧急情况，必须立即行动                                      │
    │            │ - 黑天鹅事件 (5分钟波动 > 5%)                               │
    │            │ - 极端清算墙触发 (距离 < 0.1% 且金额 > 5M)                   │
    │            │ - 系统级风险                                               │
    └────────────┴────────────────────────────────────────────────────────────┘
    """
    
    # 预警阈值 - 价格关键位
    PRICE_INFO_THRESHOLD = 0.02      # 2% - INFO
    PRICE_WARNING_THRESHOLD = 0.015  # 1.5% - WARNING起点
    PRICE_CRITICAL_THRESHOLD = 0.005 # 0.5% - CRITICAL起点
    
    # 预警阈值 - 清算墙
    LIQUIDATION_INFO_THRESHOLD = 0.01      # 1% - INFO
    LIQUIDATION_WARNING_THRESHOLD = 0.008  # 0.8% - WARNING起点
    LIQUIDATION_CRITICAL_THRESHOLD = 0.003 # 0.3% - CRITICAL起点
    LIQUIDATION_EMERGENCY_THRESHOLD = 0.001 # 0.1% - EMERGENCY
    
    # 黑天鹅阈值
    BLACK_SWAN_VOLATILITY = 0.05  # 5%波动
    
    # 清算金额阈值
    LARGE_LIQUIDATION = 1_000_000      # 100万 - 大额
    HUGE_LIQUIDATION = 5_000_000       # 500万 - 巨额
    MASSIVE_LIQUIDATION = 10_000_000   # 1000万 - 超大额
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.max_alerts = 50
        self._callbacks: List[Callable] = []
        
        # Telegram 通知器
        self._telegram_enabled = False
        self._init_telegram()
    
    def _init_telegram(self):
        """初始化 Telegram 通知"""
        try:
            from infrastructure.telegram_notify import get_notifier
            self._telegram = get_notifier()
            if self._telegram.config.enabled:
                self._telegram_enabled = True
                logger.info("Telegram notifications enabled")
        except Exception as e:
            logger.debug(f"Telegram not available: {e}")
            self._telegram = None
    
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
        
        # Telegram 推送 (仅 CRITICAL 和 EMERGENCY)
        if self._telegram_enabled and alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            self._send_telegram_alert(alert)
        
        # 日志
        if alert.severity == AlertSeverity.EMERGENCY:
            logger.critical(f"🚨 {alert.title}: {alert.message}")
        elif alert.severity == AlertSeverity.CRITICAL:
            logger.warning(f"⚠️ {alert.title}: {alert.message}")
        else:
            logger.info(f"ℹ️ {alert.title}: {alert.message}")
    
    def _send_telegram_alert(self, alert: Alert):
        """发送 Telegram 预警"""
        try:
            import asyncio
            alert_dict = alert.to_dict()
            
            # 异步发送
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果循环正在运行，创建任务
                    asyncio.create_task(self._telegram.send_alert(alert_dict))
                else:
                    loop.run_until_complete(self._telegram.send_alert(alert_dict))
            except RuntimeError:
                # 没有事件循环，创建新的
                asyncio.run(self._telegram.send_alert(alert_dict))
        except Exception as e:
            logger.debug(f"Failed to send Telegram alert: {e}")
    
    def _determine_price_severity(self, distance_pct: float, strength: float) -> AlertSeverity:
        """
        根据距离和强度确定价格预警级别
        
        规则：
        - CRITICAL: 距离 < 0.5% 或 (距离 < 1% 且强度 > 3)
        - WARNING:  距离 < 1.5% 或 (距离 < 2% 且强度 > 2)
        - INFO:     距离 < 2%
        """
        if distance_pct < self.PRICE_CRITICAL_THRESHOLD:
            return AlertSeverity.CRITICAL
        elif distance_pct < self.PRICE_WARNING_THRESHOLD:
            # 0.5%~1.5% 区间，高强度升级为 CRITICAL
            if strength > 3:
                return AlertSeverity.CRITICAL
            return AlertSeverity.WARNING
        elif distance_pct < self.PRICE_INFO_THRESHOLD:
            # 1.5%~2% 区间，中等强度升级为 WARNING
            if strength > 2:
                return AlertSeverity.WARNING
            return AlertSeverity.INFO
        return AlertSeverity.INFO
    
    def check_price_levels(
        self,
        current_price: float,
        support_zones: List[tuple],
        resistance_zones: List[tuple],
    ) -> List[Alert]:
        """
        检查价格接近支撑/阻力
        
        级别规则：
        - INFO:     距离 1.5%~2%
        - WARNING:  距离 0.5%~1.5%，或强度>2时1.5%~2%
        - CRITICAL: 距离 < 0.5%，或强度>3时0.5%~1%
        
        去重规则：
        - 同一价位的多个强度支撑/阻力合并为一条预警
        - 显示最高强度和综合强度
        """
        alerts = []
        
        # 按价格分组支撑位（去重）
        support_by_price = {}
        for price, strength in support_zones[:5]:
            price_key = round(price, 0)  # 按整数价格分组
            if price_key not in support_by_price:
                support_by_price[price_key] = {"price": price, "strengths": [], "max_strength": 0}
            support_by_price[price_key]["strengths"].append(strength)
            support_by_price[price_key]["max_strength"] = max(support_by_price[price_key]["max_strength"], strength)
        
        # 检查支撑（合并后）
        for price_key, data in support_by_price.items():
            price = data["price"]
            max_strength = data["max_strength"]
            total_strength = sum(data["strengths"])
            distance_pct = abs(current_price - price) / current_price
            
            if distance_pct < self.PRICE_INFO_THRESHOLD:
                severity = self._determine_price_severity(distance_pct, max_strength)
                
                # 级别图标
                icon = {"info": "📍", "warning": "⚠️", "critical": "🔴"}.get(severity.value, "📍")
                
                # 强度显示：多强度合并显示
                if len(data["strengths"]) > 1:
                    strength_str = f"{max_strength:.1f}x (合计{total_strength:.1f}x, {len(data['strengths'])}个)"
                else:
                    strength_str = f"{max_strength:.1f}x"
                
                alert = Alert(
                    alert_type=AlertType.PRICE_LEVEL,
                    severity=severity,
                    title=f"{icon} 支撑位预警 ({strength_str})",
                    message=f"价格接近支撑 ${price:,.0f} (距离: {distance_pct*100:.2f}%)",
                    details={
                        "level_type": "support",
                        "level_price": price,
                        "strength": max_strength,
                        "total_strength": total_strength,
                        "strength_count": len(data["strengths"]),
                        "distance_pct": distance_pct * 100,
                        "action_hint": "关注反弹机会" if distance_pct < 0.01 else "留意支撑有效性",
                    }
                )
                alerts.append(alert)
                self._add_alert(alert)
        
        # 按价格分组阻力位（去重）
        resistance_by_price = {}
        for price, strength in resistance_zones[:5]:
            price_key = round(price, 0)
            if price_key not in resistance_by_price:
                resistance_by_price[price_key] = {"price": price, "strengths": [], "max_strength": 0}
            resistance_by_price[price_key]["strengths"].append(strength)
            resistance_by_price[price_key]["max_strength"] = max(resistance_by_price[price_key]["max_strength"], strength)
        
        # 检查阻力（合并后）
        for price_key, data in resistance_by_price.items():
            price = data["price"]
            max_strength = data["max_strength"]
            total_strength = sum(data["strengths"])
            distance_pct = abs(current_price - price) / current_price
            
            if distance_pct < self.PRICE_INFO_THRESHOLD:
                severity = self._determine_price_severity(distance_pct, max_strength)
                
                # 级别图标
                icon = {"info": "📍", "warning": "⚠️", "critical": "🔴"}.get(severity.value, "📍")
                
                # 强度显示
                if len(data["strengths"]) > 1:
                    strength_str = f"{max_strength:.1f}x (合计{total_strength:.1f}x, {len(data['strengths'])}个)"
                else:
                    strength_str = f"{max_strength:.1f}x"
                
                alert = Alert(
                    alert_type=AlertType.PRICE_LEVEL,
                    severity=severity,
                    title=f"{icon} 阻力位预警 ({strength_str})",
                    message=f"价格接近阻力 ${price:,.0f} (距离: {distance_pct*100:.2f}%)",
                    details={
                        "level_type": "resistance",
                        "level_price": price,
                        "strength": max_strength,
                        "total_strength": total_strength,
                        "strength_count": len(data["strengths"]),
                        "distance_pct": distance_pct * 100,
                        "action_hint": "关注突破/回调" if distance_pct < 0.01 else "留意阻力强度",
                    }
                )
                alerts.append(alert)
                self._add_alert(alert)
        
        return alerts
    
    def _determine_liquidation_severity(self, distance_pct: float, amount: float) -> AlertSeverity:
        """
        根据距离和金额确定清算预警级别
        
        规则：
        - EMERGENCY: 距离 < 0.1% 且金额 > 5M
        - CRITICAL:  距离 < 0.3% 且金额 > 1M，或距离 < 0.1%
        - WARNING:   距离 < 0.8% 且金额 > 500K，或距离 < 0.3%
        - INFO:      距离 < 1%
        """
        if distance_pct < self.LIQUIDATION_EMERGENCY_THRESHOLD:
            if amount >= self.HUGE_LIQUIDATION:
                return AlertSeverity.EMERGENCY
            return AlertSeverity.CRITICAL
        
        if distance_pct < self.LIQUIDATION_CRITICAL_THRESHOLD:
            if amount >= self.LARGE_LIQUIDATION:
                return AlertSeverity.CRITICAL
            return AlertSeverity.WARNING
        
        if distance_pct < self.LIQUIDATION_WARNING_THRESHOLD:
            return AlertSeverity.WARNING
        
        if distance_pct < self.LIQUIDATION_INFO_THRESHOLD:
            return AlertSeverity.INFO
        
        return AlertSeverity.INFO
    
    def check_liquidation(
        self,
        current_price: float,
        liquidation_zones: List[Dict],
    ) -> List[Alert]:
        """
        检查清算墙预警
        
        级别规则：
        - INFO:     距离 0.8%~1%
        - WARNING:  距离 0.3%~0.8%，或金额>500K时0.8%~1%
        - CRITICAL: 距离 0.1%~0.3% 且金额>1M
        - EMERGENCY: 距离 < 0.1% 且金额 > 5M
        """
        alerts = []
        
        for liq in liquidation_zones[:5]:
            liq_price = liq.get('price', 0)
            liq_dir = liq.get('direction', 'unknown')
            liq_amount = liq.get('amount', 0)
            
            distance_pct = abs(current_price - liq_price) / current_price
            
            if distance_pct < self.LIQUIDATION_INFO_THRESHOLD:
                severity = self._determine_liquidation_severity(distance_pct, liq_amount)
                
                # 级别图标和行动建议
                if severity == AlertSeverity.EMERGENCY:
                    icon = "🚨"
                    action = "立即检查仓位，考虑止损/减仓！"
                elif severity == AlertSeverity.CRITICAL:
                    icon = "🔴"
                    action = "密切关注，准备应对清算连锁反应"
                elif severity == AlertSeverity.WARNING:
                    icon = "⚠️"
                    action = "关注清算触发后的波动"
                else:
                    icon = "📍"
                    action = "留意清算墙位置"
                
                # 格式化金额
                if liq_amount >= 1_000_000:
                    amount_str = f"${liq_amount/1_000_000:.1f}M"
                else:
                    amount_str = f"${liq_amount/1_000:.0f}K"
                
                alert = Alert(
                    alert_type=AlertType.LIQUIDATION,
                    severity=severity,
                    title=f"{icon} 清算墙预警",
                    message=f"价格接近{liq_dir}头清算墙 ${liq_price:,.0f} ({amount_str}, 距离: {distance_pct*100:.2f}%)",
                    details={
                        "liquidation_price": liq_price,
                        "direction": liq_dir,
                        "amount": liq_amount,
                        "distance_pct": distance_pct * 100,
                        "action_hint": action,
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
        divergence_strength: float = 1.0,  # 背离强度 0-1
    ) -> Optional[Alert]:
        """
        检查CVD背离
        
        级别规则：
        - CRITICAL: 强背离 (divergence_strength > 0.7)
        - WARNING:  中等背离 (divergence_strength 0.4-0.7)
        - INFO:     弱背离 (divergence_strength < 0.4)
        """
        alert = None
        
        # 价格上涨但CVD看跌 = 顶背离
        if price_trend == "up" and cvd_trend == "bearish":
            # 根据强度确定级别
            if divergence_strength > 0.7:
                severity = AlertSeverity.CRITICAL
                icon = "🔴"
                action = "强烈顶背离，考虑减仓止盈"
            elif divergence_strength > 0.4:
                severity = AlertSeverity.WARNING
                icon = "⚠️"
                action = "顶背离确认，关注反转信号"
            else:
                severity = AlertSeverity.INFO
                icon = "📍"
                action = "轻度顶背离，持续观察"
            
            alert = Alert(
                alert_type=AlertType.DIVERGENCE,
                severity=severity,
                title=f"{icon} 顶背离预警",
                message=f"价格上涨但CVD走弱 ({cvd_value:.1f})，强度: {divergence_strength*100:.0f}%",
                details={
                    "price_trend": price_trend,
                    "cvd_trend": cvd_trend,
                    "cvd_value": cvd_value,
                    "divergence_type": "bearish",
                    "divergence_strength": divergence_strength,
                    "action_hint": action,
                }
            )
        
        # 价格下跌但CVD看涨 = 底背离
        elif price_trend == "down" and cvd_trend == "bullish":
            # 根据强度确定级别
            if divergence_strength > 0.7:
                severity = AlertSeverity.CRITICAL
                icon = "🔴"
                action = "强烈底背离，关注做多机会"
            elif divergence_strength > 0.4:
                severity = AlertSeverity.WARNING
                icon = "⚠️"
                action = "底背离确认，等待反转确认"
            else:
                severity = AlertSeverity.INFO
                icon = "📍"
                action = "轻度底背离，持续观察"
            
            alert = Alert(
                alert_type=AlertType.DIVERGENCE,
                severity=severity,
                title=f"{icon} 底背离预警",
                message=f"价格下跌但CVD走强 ({cvd_value:.1f})，强度: {divergence_strength*100:.0f}%",
                details={
                    "price_trend": price_trend,
                    "cvd_trend": cvd_trend,
                    "cvd_value": cvd_value,
                    "divergence_type": "bullish",
                    "divergence_strength": divergence_strength,
                    "action_hint": action,
                }
            )
        
        if alert:
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
        
        级别规则：
        - EMERGENCY: 5分钟波动 > 5%
        - CRITICAL:  5分钟波动 3%~5%
        - WARNING:   5分钟波动 2%~3%
        """
        abs_change = abs(price_change_pct)
        
        # 判断级别
        if abs_change > self.BLACK_SWAN_VOLATILITY * 100:  # > 5%
            severity = AlertSeverity.EMERGENCY
            icon = "🚨"
            action = "立即暂停交易！极端行情，等待市场稳定"
        elif abs_change > 3:  # 3%~5%
            severity = AlertSeverity.CRITICAL
            icon = "🔴"
            action = "大幅波动，谨慎操作，减小仓位"
        elif abs_change > 2:  # 2%~3%
            severity = AlertSeverity.WARNING
            icon = "⚠️"
            action = "波动加剧，注意风险管理"
        else:
            return None
        
        direction = "暴涨" if price_change_pct > 0 else "暴跌"
        
        alert = Alert(
            alert_type=AlertType.BLACK_SWAN,
            severity=severity,
            title=f"{icon} 黑天鹅预警: {direction}",
            message=f"5分钟{direction} {abs_change:.2f}%，波动率: {volatility:.2f}%",
            details={
                "price_change_pct": price_change_pct,
                "volatility": volatility,
                "action_hint": action,
                "action_recommended": "pause_trading" if severity == AlertSeverity.EMERGENCY else "reduce_risk",
            }
        )
        self._add_alert(alert)
        return alert
    
    def check_breakout(
        self,
        current_price: float,
        previous_price: float,
        range_high: float,
        range_low: float,
        cvd_value: float,
        cvd_previous: float,
        volume_ratio: float = 1.0,
    ) -> Optional[Alert]:
        """
        检查区间突破
        
        当价格收盘突破震荡区间且CVD同步放大时触发
        
        Args:
            current_price: 当前价格
            previous_price: 前一价格（用于判断突破方向）
            range_high: 震荡区间上轨
            range_low: 震荡区间下轨
            cvd_value: 当前CVD
            cvd_previous: 前一CVD
            volume_ratio: 成交量比率（相对平均成交量）
        
        级别规则：
        - CRITICAL: 突破 + CVD放大 > 50% + 放量 > 1.5倍
        - WARNING:  突破 + CVD放大 > 20%
        - INFO:     突破确认
        """
        alert = None
        
        # 计算区间宽度
        range_width = (range_high - range_low) / range_low * 100
        
        # 计算CVD变化
        cvd_change = (cvd_value - cvd_previous) / abs(cvd_previous) if cvd_previous != 0 else 0
        
        # 向上突破
        if previous_price <= range_high and current_price > range_high:
            # 判断突破强度
            breakout_strength = (current_price - range_high) / (range_high - range_low)
            cvd_confirm = cvd_change > 0.2  # CVD放大20%以上
            volume_confirm = volume_ratio > 1.2  # 放量20%以上
            
            if cvd_confirm and volume_confirm and cvd_change > 0.5:
                severity = AlertSeverity.CRITICAL
                icon = "🚀"
                action = f"强势向上突破！CVD放大{cvd_change*100:.0f}%，放量{volume_ratio:.1f}x，可考虑做多"
            elif cvd_confirm:
                severity = AlertSeverity.WARNING
                icon = "📈"
                action = f"向上突破确认，CVD放大{cvd_change*100:.0f}%，关注追多机会"
            else:
                severity = AlertSeverity.INFO
                icon = "📍"
                action = "价格向上突破区间，但CVD未同步放大，谨防假突破"
            
            alert = Alert(
                alert_type=AlertType.BREAKOUT,
                severity=severity,
                title=f"{icon} 向上突破预警",
                message=f"价格突破 ${range_high:,.0f} 至 ${current_price:,.0f}，区间宽度{range_width:.1f}%",
                details={
                    "breakout_type": "up",
                    "breakout_price": current_price,
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_width_pct": range_width,
                    "cvd_change_pct": cvd_change * 100,
                    "volume_ratio": volume_ratio,
                    "action_hint": action,
                    "stop_loss_hint": f"止损参考: ${range_low:,.0f} (区间下轨)",
                }
            )
        
        # 向下突破
        elif previous_price >= range_low and current_price < range_low:
            breakout_strength = (range_low - current_price) / (range_high - range_low)
            cvd_confirm = cvd_change < -0.2  # CVD反向放大（卖出压力）
            volume_confirm = volume_ratio > 1.2
            
            if cvd_confirm and volume_confirm and abs(cvd_change) > 0.5:
                severity = AlertSeverity.CRITICAL
                icon = "📉"
                action = f"强势向下突破！卖压增加{abs(cvd_change)*100:.0f}%，放量{volume_ratio:.1f}x，可考虑做空"
            elif cvd_confirm:
                severity = AlertSeverity.WARNING
                icon = "📉"
                action = f"向下突破确认，卖压增加{abs(cvd_change)*100:.0f}%，关注追空机会"
            else:
                severity = AlertSeverity.INFO
                icon = "📍"
                action = "价格向下突破区间，但卖压不足，谨防假突破"
            
            alert = Alert(
                alert_type=AlertType.BREAKOUT,
                severity=severity,
                title=f"{icon} 向下突破预警",
                message=f"价格跌破 ${range_low:,.0f} 至 ${current_price:,.0f}，区间宽度{range_width:.1f}%",
                details={
                    "breakout_type": "down",
                    "breakout_price": current_price,
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_width_pct": range_width,
                    "cvd_change_pct": cvd_change * 100,
                    "volume_ratio": volume_ratio,
                    "action_hint": action,
                    "stop_loss_hint": f"止损参考: ${range_high:,.0f} (区间上轨)",
                }
            )
        
        if alert:
            self._add_alert(alert)
        return alert
    
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
        divergence_strength: float = 1.0,
    ) -> List[Alert]:
        """
        执行所有检查
        
        Args:
            divergence_strength: 背离强度 0-1，默认1.0
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
        div_alert = self.check_divergence(
            price_trend, cvd_trend, cvd_value, divergence_strength
        )
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
    
    def get_severity_rules(self) -> Dict:
        """获取预警级别规则说明"""
        return {
            "INFO": {
                "icon": "📍",
                "description": "信息提示，无需紧急行动",
                "examples": [
                    "价格距离关键位 1.5%~2%",
                    "清算墙距离 0.8%~1%",
                    "轻度背离 (强度<40%)",
                ]
            },
            "WARNING": {
                "icon": "⚠️",
                "description": "需要关注，可能需要行动",
                "examples": [
                    "价格距离关键位 0.5%~1.5%",
                    "清算墙距离 0.3%~0.8%",
                    "中度背离 (强度40%~70%)",
                    "5分钟波动 2%~3%",
                ]
            },
            "CRITICAL": {
                "icon": "🔴",
                "description": "需要立即关注，建议采取行动",
                "examples": [
                    "价格距离关键位 < 0.5%",
                    "清算墙距离 < 0.3% 且金额 > 100万",
                    "强背离 (强度>70%)",
                    "5分钟波动 3%~5%",
                ]
            },
            "EMERGENCY": {
                "icon": "🚨",
                "description": "紧急情况，必须立即行动",
                "examples": [
                    "黑天鹅事件 (5分钟波动 > 5%)",
                    "清算墙距离 < 0.1% 且金额 > 500万",
                ]
            }
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
