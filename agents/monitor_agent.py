"""
监控代理
========

风控与监控模块：
- 仓位监控
- 风险管理
- 异常报警
- 日志记录
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from core.base import BaseModule


class RiskLevel(Enum):
    """风险等级"""
    LOW = 'low'
    NORMAL = 'normal'
    HIGH = 'high'
    CRITICAL = 'critical'


class AlertType(Enum):
    """警报类型"""
    POSITION_LIMIT = 'position_limit'       # 仓位超限
    LOSS_LIMIT = 'loss_limit'               # 亏损超限
    PRICE_ANOMALY = 'price_anomaly'         # 价格异常
    VOLATILITY_SPIKE = 'volatility_spike'   # 波动异常
    ORDER_FAILED = 'order_failed'           # 订单失败
    CONNECTION_LOST = 'connection_lost'     # 连接断开
    MARGIN_CALL = 'margin_call'             # 保证金不足
    LIQUIDATION_RISK = 'liquidation_risk'   # 爆仓风险


@dataclass
class Alert:
    """警报"""
    alert_type: AlertType
    level: RiskLevel
    message: str
    symbol: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'alert_type': self.alert_type.value,
            'level': self.level.value,
            'message': self.message,
            'symbol': self.symbol,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
        }


@dataclass
class RiskMetrics:
    """风险指标"""
    total_equity: float = 0
    total_position_value: float = 0
    position_ratio: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    max_drawdown: float = 0
    daily_pnl: float = 0
    daily_pnl_pct: float = 0
    margin_used: float = 0
    margin_ratio: float = 0
    leverage: float = 1
    
    def to_dict(self) -> Dict:
        return {
            'total_equity': self.total_equity,
            'total_position_value': self.total_position_value,
            'position_ratio': self.position_ratio,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'max_drawdown': self.max_drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl_pct,
            'margin_used': self.margin_used,
            'margin_ratio': self.margin_ratio,
            'leverage': self.leverage,
        }


class MonitorAgent(BaseModule):
    """
    监控代理
    
    功能：
    1. 风控管理 - 仓位、亏损、杠杆限制
    2. 异常检测 - 价格异常、波动异常
    3. 警报通知 - 多渠道警报
    4. 日志记录 - 完整审计日志
    5. 自动保护 - 触发风控自动平仓
    """
    
    # 风控阈值
    RISK_THRESHOLDS = {
        'max_position_ratio': 0.50,      # 最大总仓位 50%
        'max_single_position': 0.30,     # 最大单仓 30%
        'max_loss_ratio': 0.15,          # 最大亏损 15%
        'max_daily_loss': 0.05,          # 最大日亏损 5%
        'max_leverage': 5,               # 最大杠杆 5x
        'margin_call_ratio': 0.8,        # 追保比例 80%
        'liquidation_ratio': 0.9,        # 爆仓比例 90%
        'price_change_threshold': 0.05,  # 价格异常阈值 5%
        'volatility_threshold': 0.10,    # 波动异常阈值 10%
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('monitor_agent', config)
        
        # 风控阈值（可覆盖默认值）
        self._thresholds = {**self.RISK_THRESHOLDS, **self.config.get('thresholds', {})}
        
        # 风险指标
        self._metrics = RiskMetrics()
        
        # 警报
        self._alerts: List[Alert] = []
        self._alert_callbacks: List[Callable] = []
        
        # 仓位追踪
        self._positions: Dict[str, Dict] = {}
        self._equity_history: List[Dict] = []
        
        # 自动保护开关
        self._auto_protection = self.config.get('auto_protection', True)
        
        # 统计
        self._stats = {
            'total_checks': 0,
            'alerts_triggered': 0,
            'auto_closures': 0,
            'max_drawdown_reached': 0,
        }
        
        # 回调
        self._on_alert_callbacks: List[Callable] = []
        self._on_risk_breach: Optional[Callable] = None
    
    async def initialize(self) -> bool:
        """初始化"""
        self.logger.info("Initializing monitor agent...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """启动"""
        self._running = True
        self._start_time = datetime.now()
        self.logger.info("Monitor agent started")
        return True
    
    async def stop(self) -> bool:
        """停止"""
        self._running = False
        self.logger.info("Monitor agent stopped")
        return True
    
    # ==================== 风控检查 ====================
    
    async def check_risk(self) -> Dict[str, Any]:
        """
        全面风控检查
        
        Returns:
            风控检查结果
        """
        self._stats['total_checks'] += 1
        
        results = {
            'passed': True,
            'risk_level': RiskLevel.NORMAL,
            'checks': {},
            'alerts': [],
        }
        
        # 1. 仓位检查
        position_check = self._check_position_limit()
        results['checks']['position'] = position_check
        if not position_check['passed']:
            results['passed'] = False
            results['alerts'].append(position_check['alert'])
        
        # 2. 亏损检查
        loss_check = self._check_loss_limit()
        results['checks']['loss'] = loss_check
        if not loss_check['passed']:
            results['passed'] = False
            results['alerts'].append(loss_check['alert'])
        
        # 3. 杠杆检查
        leverage_check = self._check_leverage()
        results['checks']['leverage'] = leverage_check
        if not leverage_check['passed']:
            results['alerts'].append(leverage_check['alert'])
        
        # 4. 保证金检查
        margin_check = self._check_margin()
        results['checks']['margin'] = margin_check
        if not margin_check['passed']:
            results['alerts'].append(margin_check['alert'])
        
        # 确定风险等级
        if results['alerts']:
            critical_count = sum(1 for a in results['alerts'] if a.level == RiskLevel.CRITICAL)
            high_count = sum(1 for a in results['alerts'] if a.level == RiskLevel.HIGH)
            
            if critical_count > 0:
                results['risk_level'] = RiskLevel.CRITICAL
            elif high_count > 0:
                results['risk_level'] = RiskLevel.HIGH
            else:
                results['risk_level'] = RiskLevel.NORMAL
        
        # 触发警报
        for alert in results['alerts']:
            await self._trigger_alert(alert)
        
        # 自动保护
        if self._auto_protection and results['risk_level'] == RiskLevel.CRITICAL:
            await self._auto_protection_action(results)
        
        return results
    
    def _check_position_limit(self) -> Dict[str, Any]:
        """检查仓位限制"""
        check = {
            'passed': True,
            'alert': None,
            'details': {},
        }
        
        # 检查总仓位
        if self._metrics.position_ratio > self._thresholds['max_position_ratio']:
            check['passed'] = False
            check['alert'] = Alert(
                alert_type=AlertType.POSITION_LIMIT,
                level=RiskLevel.HIGH,
                message=f"总仓位超限: {self._metrics.position_ratio*100:.1f}% > {self._thresholds['max_position_ratio']*100:.0f}%",
                data={'current': self._metrics.position_ratio, 'limit': self._thresholds['max_position_ratio']},
            )
        
        # 检查单仓
        for symbol, pos in self._positions.items():
            position_pct = pos.get('position_pct', 0)
            if position_pct > self._thresholds['max_single_position']:
                check['passed'] = False
                check['alert'] = Alert(
                    alert_type=AlertType.POSITION_LIMIT,
                    level=RiskLevel.HIGH,
                    message=f"单仓超限 {symbol}: {position_pct*100:.1f}%",
                    symbol=symbol,
                    data={'current': position_pct, 'limit': self._thresholds['max_single_position']},
                )
                break
        
        return check
    
    def _check_loss_limit(self) -> Dict[str, Any]:
        """检查亏损限制"""
        check = {
            'passed': True,
            'alert': None,
            'details': {},
        }
        
        # 总亏损检查
        total_loss_pct = abs(self._metrics.realized_pnl) / self._metrics.total_equity if self._metrics.total_equity > 0 else 0
        
        if total_loss_pct > self._thresholds['max_loss_ratio']:
            check['passed'] = False
            check['alert'] = Alert(
                alert_type=AlertType.LOSS_LIMIT,
                level=RiskLevel.CRITICAL,
                message=f"累计亏损超限: {total_loss_pct*100:.1f}% > {self._thresholds['max_loss_ratio']*100:.0f}%",
                data={'current': total_loss_pct, 'limit': self._thresholds['max_loss_ratio']},
            )
        
        # 日亏损检查
        elif self._metrics.daily_pnl_pct < -self._thresholds['max_daily_loss']:
            check['passed'] = False
            check['alert'] = Alert(
                alert_type=AlertType.LOSS_LIMIT,
                level=RiskLevel.HIGH,
                message=f"日亏损超限: {self._metrics.daily_pnl_pct*100:.1f}%",
                data={'current': self._metrics.daily_pnl_pct, 'limit': -self._thresholds['max_daily_loss']},
            )
        
        return check
    
    def _check_leverage(self) -> Dict[str, Any]:
        """检查杠杆"""
        check = {
            'passed': True,
            'alert': None,
            'details': {},
        }
        
        if self._metrics.leverage > self._thresholds['max_leverage']:
            check['passed'] = False
            check['alert'] = Alert(
                alert_type=AlertType.POSITION_LIMIT,
                level=RiskLevel.HIGH,
                message=f"杠杆过高: {self._metrics.leverage:.1f}x > {self._thresholds['max_leverage']}x",
                data={'current': self._metrics.leverage, 'limit': self._thresholds['max_leverage']},
            )
        
        return check
    
    def _check_margin(self) -> Dict[str, Any]:
        """检查保证金"""
        check = {
            'passed': True,
            'alert': None,
            'details': {},
        }
        
        if self._metrics.margin_ratio >= self._thresholds['liquidation_ratio']:
            check['passed'] = False
            check['alert'] = Alert(
                alert_type=AlertType.LIQUIDATION_RISK,
                level=RiskLevel.CRITICAL,
                message=f"爆仓风险: 保证金比例 {self._metrics.margin_ratio*100:.1f}%",
                data={'margin_ratio': self._metrics.margin_ratio},
            )
        
        elif self._metrics.margin_ratio >= self._thresholds['margin_call_ratio']:
            check['passed'] = False
            check['alert'] = Alert(
                alert_type=AlertType.MARGIN_CALL,
                level=RiskLevel.HIGH,
                message=f"保证金不足: 比例 {self._metrics.margin_ratio*100:.1f}%",
                data={'margin_ratio': self._metrics.margin_ratio},
            )
        
        return check
    
    # ==================== 价格异常检测 ====================
    
    async def check_price_anomaly(
        self, 
        symbol: str, 
        current_price: float, 
        previous_price: float
    ) -> Optional[Alert]:
        """检查价格异常"""
        if previous_price <= 0:
            return None
        
        change_pct = abs(current_price - previous_price) / previous_price
        
        if change_pct > self._thresholds['price_change_threshold']:
            alert = Alert(
                alert_type=AlertType.PRICE_ANOMALY,
                level=RiskLevel.HIGH,
                message=f"价格异常 {symbol}: 变动 {change_pct*100:.1f}%",
                symbol=symbol,
                data={
                    'current': current_price,
                    'previous': previous_price,
                    'change_pct': change_pct,
                },
            )
            await self._trigger_alert(alert)
            return alert
        
        return None
    
    async def check_volatility(
        self, 
        symbol: str, 
        volatility: float
    ) -> Optional[Alert]:
        """检查波动异常"""
        if volatility > self._thresholds['volatility_threshold']:
            alert = Alert(
                alert_type=AlertType.VOLATILITY_SPIKE,
                level=RiskLevel.HIGH,
                message=f"波动异常 {symbol}: {volatility*100:.1f}%",
                symbol=symbol,
                data={'volatility': volatility},
            )
            await self._trigger_alert(alert)
            return alert
        
        return None
    
    # ==================== 警报管理 ====================
    
    async def _trigger_alert(self, alert: Alert):
        """触发警报"""
        self._alerts.append(alert)
        self._stats['alerts_triggered'] += 1
        
        # 限制警报数量
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-50:]
        
        # 记录日志
        self.logger.warning(f"ALERT [{alert.level.value}]: {alert.message}")
        
        # 通知回调
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """确认警报"""
        if 0 <= alert_index < len(self._alerts):
            self._alerts[alert_index].acknowledged = True
            return True
        return False
    
    def add_alert_callback(self, callback: Callable):
        """添加警报回调"""
        self._alert_callbacks.append(callback)
    
    # ==================== 自动保护 ====================
    
    async def _auto_protection_action(self, risk_result: Dict[str, Any]):
        """自动保护动作"""
        self._stats['auto_closures'] += 1
        
        self.logger.critical("触发自动保护机制！")
        
        # 通知风控违规回调
        if self._on_risk_breach:
            try:
                if asyncio.iscoroutinefunction(self._on_risk_breach):
                    await self._on_risk_breach(risk_result)
                else:
                    self._on_risk_breach(risk_result)
            except Exception as e:
                self.logger.error(f"Risk breach callback error: {e}")
    
    def set_risk_breach_callback(self, callback: Callable):
        """设置风控违规回调"""
        self._on_risk_breach = callback
    
    # ==================== 数据更新 ====================
    
    async def update_metrics(
        self,
        total_equity: float,
        positions: Dict[str, Dict],
        daily_pnl: float = 0,
        realized_pnl: float = 0,
        margin_used: float = 0,
        leverage: float = 1
    ):
        """更新风险指标"""
        self._positions = positions
        self._metrics.total_equity = total_equity
        self._metrics.daily_pnl = daily_pnl
        self._metrics.daily_pnl_pct = daily_pnl / total_equity if total_equity > 0 else 0
        self._metrics.realized_pnl = realized_pnl
        self._metrics.margin_used = margin_used
        self._metrics.leverage = leverage
        
        # 计算总仓位
        total_position_value = sum(
            pos.get('value', 0) for pos in positions.values()
        )
        self._metrics.total_position_value = total_position_value
        self._metrics.position_ratio = total_position_value / total_equity if total_equity > 0 else 0
        
        # 计算未实现盈亏
        unrealized_pnl = sum(
            pos.get('unrealized_pnl', 0) for pos in positions.values()
        )
        self._metrics.unrealized_pnl = unrealized_pnl
        
        # 保证金比例
        self._metrics.margin_ratio = margin_used / total_equity if total_equity > 0 else 0
        
        # 记录权益历史
        self._equity_history.append({
            'timestamp': datetime.now().isoformat(),
            'equity': total_equity,
            'position_value': total_position_value,
            'unrealized_pnl': unrealized_pnl,
        })
        
        # 更新最大回撤
        if len(self._equity_history) > 1:
            peak = max(h['equity'] for h in self._equity_history)
            if peak > 0:
                drawdown = (peak - total_equity) / peak
                self._metrics.max_drawdown = max(self._metrics.max_drawdown, drawdown)
        
        # 保持历史在合理范围
        if len(self._equity_history) > 1000:
            self._equity_history = self._equity_history[-500:]
    
    # ==================== 状态查询 ====================
    
    def get_metrics(self) -> RiskMetrics:
        """获取风险指标"""
        return self._metrics
    
    def get_alerts(self, limit: int = 20, unacknowledged_only: bool = False) -> List[Alert]:
        """获取警报"""
        alerts = self._alerts[-limit:]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        return alerts
    
    def get_risk_level(self) -> RiskLevel:
        """获取当前风险等级"""
        # 根据各项指标判断
        if (self._metrics.position_ratio > self._thresholds['max_position_ratio'] or
            self._metrics.margin_ratio >= self._thresholds['liquidation_ratio']):
            return RiskLevel.CRITICAL
        
        if (self._metrics.daily_pnl_pct < -self._thresholds['max_daily_loss'] or
            self._metrics.margin_ratio >= self._thresholds['margin_call_ratio']):
            return RiskLevel.HIGH
        
        if self._metrics.position_ratio > self._thresholds['max_position_ratio'] * 0.8:
            return RiskLevel.LOW
        
        return RiskLevel.NORMAL
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'running': self._running,
            'auto_protection': self._auto_protection,
            'risk_level': self.get_risk_level().value,
            'metrics': self._metrics.to_dict(),
            'thresholds': self._thresholds,
            'unacknowledged_alerts': len([a for a in self._alerts if not a.acknowledged]),
            'stats': self._stats,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'healthy': self._running,
            'risk_level': self.get_risk_level().value,
            'alerts_today': self._stats['alerts_triggered'],
            'auto_closures': self._stats['auto_closures'],
        }
    
    def update_thresholds(self, thresholds: Dict[str, float]):
        """更新风控阈值"""
        self._thresholds.update(thresholds)
        self.logger.info(f"Risk thresholds updated: {thresholds}")


# 导出
__all__ = [
    'MonitorAgent',
    'RiskLevel',
    'AlertType',
    'Alert',
    'RiskMetrics'
]
