"""
任务调度器
==========

核心调度中心，负责：
- 定时任务调度
- 心跳控制
- Agent 协调
- 数据同步触发
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 1  # 关键任务（风控、止损）
    HIGH = 2      # 高优先级（信号生成、下单）
    NORMAL = 3    # 普通优先级（数据同步）
    LOW = 4       # 低优先级（日志、报告）


class TaskStatus(Enum):
    """任务状态"""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass
class ScheduledTask:
    """计划任务"""
    name: str
    callback: Callable
    interval: float  # 秒
    priority: TaskPriority = TaskPriority.NORMAL
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    error_count: int = 0
    max_errors: int = 3
    enabled: bool = True
    
    def __post_init__(self):
        if self.next_run is None:
            self.next_run = datetime.now()


class Scheduler:
    """
    任务调度器
    
    功能：
    1. 定时任务调度
    2. 优先级队列
    3. 错误重试
    4. 任务监控
    5. 心跳控制
    """
    
    # 预定义任务间隔
    INTERVALS = {
        'tick': 1,           # 实时行情 (1秒)
        'signal': 5,         # 信号生成 (5秒)
        'order': 1,          # 订单检查 (1秒)
        'balance': 10,       # 余额同步 (10秒)
        'position': 5,       # 持仓同步 (5秒)
        'risk_check': 3,     # 风控检查 (3秒)
        'kline_1m': 60,      # 1分钟K线
        'kline_15m': 900,    # 15分钟K线
        'kline_1h': 3600,    # 1小时K线
        'heartbeat': 25,     # 心跳检测 (25秒)
        'health_check': 30,  # 健康检查 (30秒)
        'report': 3600,      # 报告生成 (1小时)
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("Scheduler")
        
        # 任务队列
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # 统计
        self._stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_errors': 0,
        }
        
        # Agent 引用
        self._agents: Dict[str, Any] = {}
        
        # 回调
        self._on_task_complete: Optional[Callable] = None
        self._on_task_error: Optional[Callable] = None
    
    # ==================== Agent 注册 ====================
    
    def register_agent(self, name: str, agent: Any):
        """注册 Agent"""
        self._agents[name] = agent
        self.logger.info(f"Agent registered: {name}")
    
    def get_agent(self, name: str) -> Optional[Any]:
        """获取 Agent"""
        return self._agents.get(name)
    
    # ==================== 任务管理 ====================
    
    def add_task(
        self,
        name: str,
        callback: Callable,
        interval: float,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_errors: int = 3
    ) -> bool:
        """添加定时任务"""
        if name in self._tasks:
            self.logger.warning(f"Task already exists: {name}")
            return False
        
        task = ScheduledTask(
            name=name,
            callback=callback,
            interval=interval,
            priority=priority,
            max_errors=max_errors,
            next_run=datetime.now()
        )
        
        self._tasks[name] = task
        self.logger.info(f"Task added: {name}, interval={interval}s, priority={priority.name}")
        return True
    
    def remove_task(self, name: str) -> bool:
        """移除任务"""
        if name in self._tasks:
            del self._tasks[name]
            self.logger.info(f"Task removed: {name}")
            return True
        return False
    
    def enable_task(self, name: str, enabled: bool = True):
        """启用/禁用任务"""
        if name in self._tasks:
            self._tasks[name].enabled = enabled
    
    def update_interval(self, name: str, interval: float):
        """更新任务间隔"""
        if name in self._tasks:
            self._tasks[name].interval = interval
            self.logger.info(f"Task interval updated: {name} -> {interval}s")
    
    # ==================== 核心调度 ====================
    
    async def start(self):
        """启动调度器"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._run_loop())
        self.logger.info("Scheduler started")
    
    async def stop(self):
        """停止调度器"""
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Scheduler stopped")
    
    async def _run_loop(self):
        """主调度循环"""
        while self._running:
            try:
                now = datetime.now()
                
                # 获取需要执行的任务
                ready_tasks = self._get_ready_tasks(now)
                
                # 按优先级排序
                ready_tasks.sort(key=lambda t: t.priority.value)
                
                # 执行任务
                for task in ready_tasks:
                    if not self._running:
                        break
                    
                    await self._execute_task(task, now)
                
                # 短暂休眠
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1)
    
    def _get_ready_tasks(self, now: datetime) -> List[ScheduledTask]:
        """获取就绪任务"""
        ready = []
        
        for task in self._tasks.values():
            if not task.enabled:
                continue
            
            if task.status == TaskStatus.RUNNING:
                continue
            
            if task.next_run and now >= task.next_run:
                ready.append(task)
        
        return ready
    
    async def _execute_task(self, task: ScheduledTask, now: datetime):
        """执行任务"""
        task.status = TaskStatus.RUNNING
        task.last_run = now
        
        try:
            self._stats['total_runs'] += 1
            
            # 执行回调
            if asyncio.iscoroutinefunction(task.callback):
                result = await task.callback()
            else:
                result = task.callback()
            
            # 成功
            task.status = TaskStatus.COMPLETED
            task.error_count = 0
            self._stats['successful_runs'] += 1
            
            # 更新下次执行时间
            task.next_run = now + timedelta(seconds=task.interval)
            
            # 回调通知
            if self._on_task_complete:
                await self._safe_callback(self._on_task_complete, task.name, result)
            
        except Exception as e:
            # 失败
            task.status = TaskStatus.FAILED
            task.error_count += 1
            self._stats['failed_runs'] += 1
            self._stats['total_errors'] += 1
            
            self.logger.error(f"Task failed: {task.name}, error: {e}")
            
            # 检查是否禁用
            if task.error_count >= task.max_errors:
                task.enabled = False
                self.logger.warning(f"Task disabled due to max errors: {task.name}")
            
            # 回调通知
            if self._on_task_error:
                await self._safe_callback(self._on_task_error, task.name, str(e))
    
    async def _safe_callback(self, callback: Callable, *args):
        """安全执行回调"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            self.logger.error(f"Callback error: {e}")
    
    # ==================== 预定义任务注册 ====================
    
    def setup_default_tasks(self):
        """设置默认任务"""
        # 心跳检测
        self.add_task(
            'heartbeat',
            self._heartbeat,
            self.INTERVALS['heartbeat'],
            TaskPriority.CRITICAL
        )
        
        # 健康检查
        self.add_task(
            'health_check',
            self._health_check,
            self.INTERVALS['health_check'],
            TaskPriority.HIGH
        )
    
    async def _heartbeat(self):
        """心跳检测"""
        for name, agent in self._agents.items():
            if hasattr(agent, 'heartbeat'):
                try:
                    await agent.heartbeat()
                except Exception as e:
                    self.logger.error(f"Agent heartbeat failed: {name}, error: {e}")
    
    async def _health_check(self):
        """健康检查"""
        health_status = {}
        
        for name, agent in self._agents.items():
            if hasattr(agent, 'health_check'):
                try:
                    status = await agent.health_check()
                    health_status[name] = status
                except Exception as e:
                    health_status[name] = {'healthy': False, 'error': str(e)}
        
        return health_status
    
    # ==================== 手动触发 ====================
    
    async def run_once(self, name: str) -> Any:
        """立即执行一次任务"""
        task = self._tasks.get(name)
        if not task:
            raise ValueError(f"Task not found: {name}")
        
        result = None
        
        if asyncio.iscoroutinefunction(task.callback):
            result = await task.callback()
        else:
            result = task.callback()
        
        return result
    
    async def run_all(self, priority: Optional[TaskPriority] = None):
        """立即执行所有任务（或指定优先级）"""
        results = {}
        
        tasks = list(self._tasks.values())
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        
        tasks.sort(key=lambda t: t.priority.value)
        
        for task in tasks:
            try:
                if asyncio.iscoroutinefunction(task.callback):
                    results[task.name] = await task.callback()
                else:
                    results[task.name] = task.callback()
            except Exception as e:
                results[task.name] = {'error': str(e)}
        
        return results
    
    # ==================== 状态查询 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            'running': self._running,
            'agents': list(self._agents.keys()),
            'tasks': {
                name: {
                    'interval': task.interval,
                    'priority': task.priority.name,
                    'status': task.status.value,
                    'enabled': task.enabled,
                    'error_count': task.error_count,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'next_run': task.next_run.isoformat() if task.next_run else None,
                }
                for name, task in self._tasks.items()
            },
            'stats': self._stats,
        }
    
    def get_task_status(self, name: str) -> Optional[Dict[str, Any]]:
        """获取单个任务状态"""
        task = self._tasks.get(name)
        if not task:
            return None
        
        return {
            'name': task.name,
            'interval': task.interval,
            'priority': task.priority.name,
            'status': task.status.value,
            'enabled': task.enabled,
            'error_count': task.error_count,
            'last_run': task.last_run.isoformat() if task.last_run else None,
            'next_run': task.next_run.isoformat() if task.next_run else None,
        }
    
    # ==================== 回调设置 ====================
    
    def on_task_complete(self, callback: Callable):
        """设置任务完成回调"""
        self._on_task_complete = callback
    
    def on_task_error(self, callback: Callable):
        """设置任务错误回调"""
        self._on_task_error = callback


# 导出
__all__ = [
    'Scheduler',
    'ScheduledTask',
    'TaskPriority',
    'TaskStatus'
]
