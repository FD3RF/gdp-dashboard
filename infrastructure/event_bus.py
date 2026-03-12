"""
事件驱动架构模块
实现发布-订阅模式，支持事件触发策略
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型"""
    # 市场事件
    PRICE_UPDATE = "price_update"
    KLINE_CLOSE = "kline_close"
    ORDERBOOK_CHANGE = "orderbook_change"
    VOLUME_SPIKE = "volume_spike"
    
    # 特征事件
    HURST_CHANGE = "hurst_change"
    FRACTAL_CHANGE = "fractal_change"
    FUNDING_EXTREME = "funding_extreme"
    LIQUIDITY_TRAP = "liquidity_trap"
    
    # 链上事件
    WHALE_ALERT = "whale_alert"
    EXCHANGE_FLOW = "exchange_flow"
    
    # 社交事件
    SENTIMENT_SPIKE = "sentiment_spike"
    TRENDING_TOPIC = "trending_topic"
    
    # 风险事件
    RISK_WARNING = "risk_warning"
    CIRCUIT_BREAKER = "circuit_breaker"
    
    # 策略事件
    SIGNAL_GENERATED = "signal_generated"
    TRADE_EXECUTED = "trade_executed"
    POSITION_CLOSED = "position_closed"
    
    # 系统事件
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
    DATA_STALE = "data_stale"
    SYSTEM_ERROR = "system_error"


@dataclass
class Event:
    """事件数据结构"""
    event_type: EventType
    timestamp: datetime
    source: str  # 事件来源模块
    data: Dict[str, Any]
    priority: int = 0  # 优先级，数字越大越优先
    processed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "priority": self.priority,
            "processed": self.processed,
        }


class EventBus:
    """
    事件总线
    实现发布-订阅模式
    """
    
    def __init__(self):
        # 订阅者: {event_type: [callbacks]}
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # 事件队列
        self.event_queue: List[Event] = []
        self.max_queue_size = 1000
        
        # 事件历史
        self.event_history: List[Dict] = []
        self.max_history = 500
        
        # 统计
        self.stats = {
            "total_published": 0,
            "total_processed": 0,
            "by_type": defaultdict(int),
        }
    
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ) -> None:
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        self.subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")
    
    def subscribe_multiple(
        self,
        event_types: List[EventType],
        callback: Callable[[Event], None]
    ) -> None:
        """订阅多种事件类型"""
        for event_type in event_types:
            self.subscribe(event_type, callback)
    
    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable
    ) -> bool:
        """取消订阅"""
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            return True
        return False
    
    def publish(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any],
        priority: int = 0
    ) -> Event:
        """
        发布事件
        
        Args:
            event_type: 事件类型
            source: 事件来源
            data: 事件数据
            priority: 优先级
        
        Returns:
            发布的事件对象
        """
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            data=data,
            priority=priority,
        )
        
        # 添加到队列
        self.event_queue.append(event)
        if len(self.event_queue) > self.max_queue_size:
            self.event_queue = self.event_queue[-self.max_queue_size:]
        
        # 更新统计
        self.stats["total_published"] += 1
        self.stats["by_type"][event_type.value] += 1
        
        logger.debug(f"Published {event_type.value} from {source}")
        
        return event
    
    def process_events(self, batch_size: int = 100) -> int:
        """
        处理事件队列
        
        Returns:
            处理的事件数量
        """
        if not self.event_queue:
            return 0
        
        # 按优先级排序
        self.event_queue.sort(key=lambda e: e.priority, reverse=True)
        
        processed = 0
        batch = self.event_queue[:batch_size]
        
        for event in batch:
            callbacks = self.subscribers.get(event.event_type, [])
            
            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
            
            event.processed = True
            processed += 1
            
            # 记录历史
            self.event_history.append(event.to_dict())
        
        # 清理已处理事件
        self.event_queue = [e for e in self.event_queue if not e.processed]
        
        # 限制历史长度
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        self.stats["total_processed"] += processed
        
        return processed
    
    def get_pending_count(self) -> int:
        """获取待处理事件数量"""
        return len(self.event_queue)
    
    def get_recent_events(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 20
    ) -> List[Dict]:
        """获取最近事件"""
        history = self.event_history[-limit:]
        
        if event_type:
            history = [e for e in history if e["event_type"] == event_type.value]
        
        return list(reversed(history))


class EventHandler:
    """
    事件处理器基类
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """设置事件处理器，子类实现"""
        pass


class RiskEventHandler(EventHandler):
    """
    风险事件处理器
    处理风险预警和熔断事件
    """
    
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self.risk_warnings: List[Dict] = []
        self.circuit_breaker_active = False
    
    def _setup_handlers(self) -> None:
        self.event_bus.subscribe(EventType.RISK_WARNING, self._handle_risk_warning)
        self.event_bus.subscribe(EventType.CIRCUIT_BREAKER, self._handle_circuit_breaker)
    
    def _handle_risk_warning(self, event: Event) -> None:
        """处理风险预警"""
        warning = {
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            **event.data
        }
        self.risk_warnings.append(warning)
        
        # 保留最近50条
        self.risk_warnings = self.risk_warnings[-50:]
        
        logger.warning(f"Risk warning: {event.data.get('message', 'Unknown')}")
    
    def _handle_circuit_breaker(self, event: Event) -> None:
        """处理熔断"""
        self.circuit_breaker_active = event.data.get("active", True)
        
        if self.circuit_breaker_active:
            logger.critical(f"CIRCUIT BREAKER ACTIVATED: {event.data.get('reason', 'Unknown')}")
        else:
            logger.info("Circuit breaker deactivated")


class WhaleAlertHandler(EventHandler):
    """
    巨鲸警报处理器
    """
    
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self.recent_alerts: List[Dict] = []
    
    def _setup_handlers(self) -> None:
        self.event_bus.subscribe(EventType.WHALE_ALERT, self._handle_whale_alert)
        self.event_bus.subscribe(EventType.EXCHANGE_FLOW, self._handle_exchange_flow)
    
    def _handle_whale_alert(self, event: Event) -> None:
        """处理巨鲸警报"""
        alert = event.data
        
        # 判断影响
        impact = alert.get("impact_score", 0)
        if impact >= 70:
            # 发布风险预警
            self.event_bus.publish(
                EventType.RISK_WARNING,
                "whale_monitor",
                {
                    "message": f"高影响巨鲸转账: {alert.get('amount_eth', 0):.1f} ETH",
                    "severity": "high",
                    "details": alert,
                },
                priority=10
            )
        
        self.recent_alerts.append({
            "timestamp": event.timestamp.isoformat(),
            **alert
        })
        self.recent_alerts = self.recent_alerts[-20:]
    
    def _handle_exchange_flow(self, event: Event) -> None:
        """处理交易所流入流出"""
        flow = event.data
        
        if flow.get("net_flow_eth", 0) < -500:
            # 大量流出 = 看涨
            self.event_bus.publish(
                EventType.SENTIMENT_SPIKE,
                "exchange_flow",
                {
                    "direction": "bullish",
                    "magnitude": abs(flow.get("net_flow_eth", 0)),
                    "message": f"交易所净流出 {abs(flow.get('net_flow_eth', 0)):.0f} ETH",
                }
            )


class StrategyTrigger:
    """
    策略触发器
    基于事件组合触发策略
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.conditions: Dict[str, Dict] = {}  # 条件配置
        self.trigger_history: List[Dict] = []
        
        # 订阅相关事件
        self.event_bus.subscribe_multiple(
            [
                EventType.PRICE_UPDATE,
                EventType.WHALE_ALERT,
                EventType.FUNDING_EXTREME,
                EventType.SENTIMENT_SPIKE,
                EventType.HURST_CHANGE,
            ],
            self._check_conditions
        )
    
    def add_trigger(
        self,
        name: str,
        conditions: Dict[str, Any],
        action: Callable
    ) -> None:
        """
        添加触发条件
        
        Args:
            name: 触发器名称
            conditions: 条件配置
                {
                    "events": [EventType.WHALE_ALERT, EventType.FUNDING_EXTREME],
                    "time_window": 300,  # 秒
                    "min_events": 2,
                    "filters": {...}
                }
            action: 触发动作
        """
        self.conditions[name] = {
            "conditions": conditions,
            "action": action,
            "pending_events": [],
        }
    
    def _check_conditions(self, event: Event) -> None:
        """检查是否满足触发条件"""
        now = datetime.now()
        
        for name, config in self.conditions.items():
            cond = config["conditions"]
            
            # 检查事件类型是否匹配
            if event.event_type not in cond.get("events", []):
                continue
            
            # 添加到待处理事件
            config["pending_events"].append(event)
            
            # 清理过期事件
            time_window = cond.get("time_window", 300)
            cutoff = now - __import__('datetime').timedelta(seconds=time_window)
            config["pending_events"] = [
                e for e in config["pending_events"]
                if e.timestamp >= cutoff
            ]
            
            # 检查是否满足最小事件数
            if len(config["pending_events"]) >= cond.get("min_events", 1):
                # 执行动作
                try:
                    result = config["action"](config["pending_events"])
                    
                    self.trigger_history.append({
                        "name": name,
                        "timestamp": now.isoformat(),
                        "events_count": len(config["pending_events"]),
                        "result": result,
                    })
                    
                    # 清空待处理事件
                    config["pending_events"] = []
                    
                    logger.info(f"Trigger '{name}' executed")
                except Exception as e:
                    logger.error(f"Trigger '{name}' failed: {e}")


# 全局事件总线实例
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def publish_event(
    event_type: EventType,
    source: str,
    data: Dict[str, Any],
    priority: int = 0
) -> Event:
    """
    发布事件（便捷函数）
    """
    return get_event_bus().publish(event_type, source, data, priority)
