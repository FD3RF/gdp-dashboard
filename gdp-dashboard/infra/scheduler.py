"""
Scheduler module for task scheduling and cron jobs.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import croniter
from core.base import BaseModule


class ScheduleType(Enum):
    """Schedule type enumeration."""
    ONCE = 'once'
    INTERVAL = 'interval'
    CRON = 'cron'
    DAILY = 'daily'
    HOURLY = 'hourly'


@dataclass
class ScheduledTask:
    """Represents a scheduled task."""
    id: str
    name: str
    func: Callable
    schedule_type: ScheduleType
    next_run: datetime
    interval: Optional[int] = None  # seconds
    cron_expression: Optional[str] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    
    def update_next_run(self) -> None:
        """Update next run time based on schedule type."""
        now = datetime.now()
        
        if self.schedule_type == ScheduleType.INTERVAL:
            self.next_run = now + timedelta(seconds=self.interval)
        elif self.schedule_type == ScheduleType.CRON and self.cron_expression:
            cron = croniter.croniter(self.cron_expression, now)
            self.next_run = cron.get_next(datetime)
        elif self.schedule_type == ScheduleType.HOURLY:
            self.next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif self.schedule_type == ScheduleType.DAILY:
            self.next_run = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    def is_due(self) -> bool:
        """Check if task is due to run."""
        if not self.enabled:
            return False
        if self.max_runs and self.run_count >= self.max_runs:
            return False
        return datetime.now() >= self.next_run


class Scheduler(BaseModule):
    """
    Async task scheduler supporting cron, interval, and one-time tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('scheduler', config)
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._check_interval = self.config.get('check_interval', 1.0)
    
    async def initialize(self) -> bool:
        """Initialize the scheduler."""
        self.logger.info("Initializing scheduler...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the scheduler."""
        if self._running:
            return True
        
        self.logger.info("Starting scheduler...")
        self._running = True
        self._start_time = datetime.now()
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        return True
    
    async def stop(self) -> bool:
        """Stop the scheduler."""
        self.logger.info("Stopping scheduler...")
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        return True
    
    async def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_run_tasks()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(self._check_interval)
    
    async def _check_and_run_tasks(self) -> None:
        """Check for due tasks and execute them."""
        tasks_to_run = [
            task for task in self._tasks.values()
            if task.is_due()
        ]
        
        for task in tasks_to_run:
            try:
                self.logger.debug(f"Executing task: {task.name}")
                await self._execute_task(task)
            except Exception as e:
                self.logger.error(f"Error executing task {task.name}: {e}")
    
    async def _execute_task(self, task: ScheduledTask) -> Any:
        """Execute a scheduled task."""
        task.last_run = datetime.now()
        task.run_count += 1
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: task.func(*task.args, **task.kwargs)
                )
            
            # Update next run time for recurring tasks
            if task.schedule_type != ScheduleType.ONCE:
                task.update_next_run()
            else:
                task.enabled = False
            
            return result
        except Exception as e:
            self.logger.error(f"Task execution failed: {task.name} - {e}")
            raise
    
    def schedule(
        self,
        name: str,
        func: Callable,
        schedule_type: ScheduleType,
        **kwargs
    ) -> str:
        """
        Schedule a new task.
        
        Args:
            name: Task name
            func: Function to execute
            schedule_type: Type of schedule
            **kwargs: Additional schedule parameters
        
        Returns:
            Task ID
        """
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        # Calculate initial next run time
        now = datetime.now()
        next_run = kwargs.get('start_time', now)
        
        if schedule_type == ScheduleType.INTERVAL:
            interval = kwargs.get('interval', 60)
            if next_run == now:
                next_run = now + timedelta(seconds=interval)
        elif schedule_type == ScheduleType.CRON:
            cron_expr = kwargs.get('cron_expression', '* * * * *')
            cron = croniter.croniter(cron_expr, now)
            next_run = cron.get_next(datetime)
        elif schedule_type == ScheduleType.HOURLY:
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif schedule_type == ScheduleType.DAILY:
            next_run = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        task = ScheduledTask(
            id=task_id,
            name=name,
            func=func,
            schedule_type=schedule_type,
            next_run=next_run,
            interval=kwargs.get('interval'),
            cron_expression=kwargs.get('cron_expression'),
            args=kwargs.get('args', ()),
            kwargs=kwargs.get('kwargs', {}),
            enabled=kwargs.get('enabled', True),
            max_runs=kwargs.get('max_runs')
        )
        
        self._tasks[task_id] = task
        self.logger.info(f"Scheduled task: {name} (ID: {task_id})")
        return task_id
    
    def schedule_interval(
        self,
        name: str,
        func: Callable,
        interval: int,
        **kwargs
    ) -> str:
        """Schedule a task to run at fixed intervals."""
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.INTERVAL,
            interval=interval,
            **kwargs
        )
    
    def schedule_cron(
        self,
        name: str,
        func: Callable,
        cron_expression: str,
        **kwargs
    ) -> str:
        """Schedule a task using cron expression."""
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.CRON,
            cron_expression=cron_expression,
            **kwargs
        )
    
    def schedule_once(
        self,
        name: str,
        func: Callable,
        run_at: datetime,
        **kwargs
    ) -> str:
        """Schedule a task to run once at a specific time."""
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.ONCE,
            start_time=run_at,
            **kwargs
        )
    
    def schedule_daily(
        self,
        name: str,
        func: Callable,
        hour: int = 0,
        minute: int = 0,
        **kwargs
    ) -> str:
        """Schedule a task to run daily."""
        run_at = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        if run_at <= datetime.now():
            run_at += timedelta(days=1)
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.DAILY,
            start_time=run_at,
            **kwargs
        )
    
    def schedule_hourly(
        self,
        name: str,
        func: Callable,
        **kwargs
    ) -> str:
        """Schedule a task to run hourly."""
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.HOURLY,
            **kwargs
        )
    
    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            self.logger.info(f"Cancelled task: {task_id}")
            return True
        return False
    
    def pause(self, task_id: str) -> bool:
        """Pause a scheduled task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False
            return True
        return False
    
    def resume(self, task_id: str) -> bool:
        """Resume a paused task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = True
            return True
        return False
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a scheduled task by ID."""
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[ScheduledTask]:
        """Get all scheduled tasks."""
        return list(self._tasks.values())
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            **super().get_status(),
            'task_count': len(self._tasks),
            'enabled_tasks': sum(1 for t in self._tasks.values() if t.enabled),
            'tasks': [
                {
                    'id': t.id,
                    'name': t.name,
                    'type': t.schedule_type.value,
                    'enabled': t.enabled,
                    'next_run': t.next_run.isoformat(),
                    'last_run': t.last_run.isoformat() if t.last_run else None,
                    'run_count': t.run_count
                }
                for t in self._tasks.values()
            ]
        }
